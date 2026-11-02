#!/usr/bin/env python3
"""Trim HoI! preparation frames, refine fixed RGB-D poses and seed initial COLMAP with Leica geometry."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from convert_hoi_dataset import export_endpoint, write_json, write_ply
from mask_colmap_models import read_model
from read_write_model import Point3D, write_model
from standardize_hoi_cameras import standardize_scene


def read_leica(path):
    with path.open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline()
            if not line:raise ValueError('Missing Leica PLY header')
            lines.append(line.decode().strip())
            if lines[-1]=='end_header':break
        count=int(next(line.split()[-1] for line in lines if line.startswith('element vertex ')))
        props=[line.split()[1:] for line in lines if line.startswith('property ')]
        if [p[1] for p in props]!=['x','y','z']:raise ValueError('Expected XYZ-only annotated Leica PLY')
        if 'format ascii 1.0' in lines:xyz=np.loadtxt(stream,max_rows=count,ndmin=2)[:,:3]
        else:
            kind={'double':'f8','float':'f4'}
            endian='<' if 'format binary_little_endian 1.0' in lines else '>'
            dtype=np.dtype([(name,endian+kind[t]) for t,name in props])
            raw=np.frombuffer(stream.read(),dtype=dtype,count=count)
            xyz=np.column_stack([raw[a] for a in 'xyz'])
    if len(xyz)!=count or not np.isfinite(xyz).all():raise ValueError('Invalid Leica cloud')
    return xyz


def trim_scene(scene,start):
    ledger=scene/'metadata/hoi_start_frame.json'
    if ledger.exists():
        if json.loads(ledger.read_text())['original_start_frame']!=start:raise ValueError('Scene already trimmed with another start frame')
        return
    if start<0:raise ValueError('Start frame must be nonnegative')
    conversion=json.loads((scene/'metadata/hoi_conversion.json').read_text())
    old_count=conversion['num_frames']
    if old_count-start<2:raise ValueError('Trimming leaves fewer than two frames')
    streams=json.loads((scene/'metadata/camera_streams.json').read_text())
    poses={s['camera']:json.loads((scene/s['poses_file']).read_text()) for s in streams['cameras']}
    for name,data in poses.items():
        indices={r['frame_index'] for r in data['frames']}
        if not {start,old_count-1}<=indices:raise ValueError(f'{name}: no pose at requested new initial/final frame')
    backup=scene/f'metadata/before_start_frame_{start:06d}'
    if backup.exists():raise ValueError(f'An interrupted preparation exists: {backup}; inspect before retrying')
    backup.mkdir()
    # Keep the original metadata and discarded frames inside the output scene.
    shutil.copytree(scene/'metadata',backup/'metadata',ignore=shutil.ignore_patterns(backup.name,'.runtime'))
    for folder in ['rgb','depth_npy','depth_png','mask']:
        for camera in sorted((scene/folder).glob('cam_*')):
            files=sorted(camera.glob('frame_*'))
            saved=backup/folder/camera.name;saved.mkdir(parents=True,exist_ok=True)
            for path in files:path.rename(saved/path.name)
            for path in sorted(saved.iterdir()):
                index=int(path.stem.removeprefix('frame_'))
                if index>=start:path.rename(camera/f'frame_{index-start:06d}{path.suffix}')
    new_count=old_count-start
    for info in streams['cameras']:
        data=poses[info['camera']];rows=[]
        for row in data['frames']:
            old=row['frame_index']
            if old<start:continue
            row.update(original_frame_index=old,frame_index=old-start)
            for key in ['rgb_path','depth_path']:
                path=Path(row[key]);row[key]=str(path.with_name(f'frame_{old-start:06d}{path.suffix}'))
            rows.append(row)
        info['num_frames']=len(rows)
        info['missing_frame_indices']=sorted(set(range(new_count))-{r['frame_index'] for r in rows})
        data.update(info,frames=rows);write_json(scene/info['poses_file'],data)
    write_json(scene/'metadata/camera_streams.json',streams)
    timestamp=scene/'metadata/frame_timestamps.csv'
    with timestamp.open() as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if int(r['frame_index'])>=start]
    origin=min(int(r['target_timestamp_ns']) for r in rows)
    for row in rows:
        row['original_frame_index']=int(row['frame_index']);row['frame_index']=int(row['frame_index'])-start
        row['timestamp_s']=(int(row['target_timestamp_ns'])-origin)/1e9
    with timestamp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+['original_frame_index']);writer.writeheader();writer.writerows(rows)
    conversion.update(num_frames=new_count,output_frames=list(range(new_count)),original_start_frame=start,
                      original_frame_count=old_count,initial_frame_rule=f'output frame 0 is original frame {start}')
    write_json(scene/'metadata/hoi_conversion.json',conversion)
    mask_manifest=scene/'metadata/mask_generation.json'
    if mask_manifest.exists():mask_manifest.unlink()  # Original prompt records are preserved in the metadata backup.
    write_json(ledger,dict(original_start_frame=start,original_frame_count=old_count,num_frames=new_count,
                           new_initial_frame=0,backup=str(backup)))


def refresh_frame_metadata(scene):
    streams=json.loads((scene/'metadata/camera_streams.json').read_text())['cameras']
    moving=[s for s in streams if s['camera_motion']=='moving']
    sets=[];missing={}
    for info in streams:
        poses=json.loads((scene/info['poses_file']).read_text())
        indices={row['frame_index'] for row in poses['frames']};sets.append(indices)
        missing[info['camera']]=info.get('missing_frame_indices',[])
    write_json(scene/'metadata/synchronized_frames.json',dict(frame_indices=sorted(set.intersection(*sets)),
        missing_by_camera=missing,rule='Use this intersection when a training sample requires every camera; absent frames are not duplicated'))
    for relative in ['metadata/hoi_conversion.json','metadata/hoi_auxiliary_export.json']:
        path=scene/relative
        if path.exists():
            data=json.loads(path.read_text())
            if 'ego_views' in data:data['ego_views']=moving
            data['frame_selection_metadata']='metadata/hoi_start_frame.json'
            write_json(path,data)


def depth_world(scene,camera,stride=2,frame=0):
    name=camera['name'];z=np.load(scene/f'depth_npy/{name}/frame_{frame:06d}.npy',allow_pickle=False)
    rgb=cv2.imread(str(scene/f'rgb/{name}/frame_{frame:06d}.png'))[:,:,::-1]
    yy,xx=np.mgrid[0:z.shape[0]:stride,0:z.shape[1]:stride]
    depth=z[yy,xx];valid=np.isfinite(depth)&(depth>0)
    x,y,depth=xx[valid],yy[valid],depth[valid]
    local=np.column_stack([(x-camera['cx'])/camera['fx']*depth,(y-camera['cy'])/camera['fy']*depth,depth])
    pose=np.asarray(camera['pose_matrix_4x4'])
    return local@pose[:3,:3].T+pose[:3,3],rgb[y,x]


def register_pose(source,target,max_distance=.10):
    """Regularized robust point-to-plane ICP; planar directions remain prior-constrained."""
    _,unique=np.unique(np.floor(target/.008).astype(np.int64),axis=0,return_index=True)
    target=target[np.sort(unique)]
    if len(target)<30:raise ValueError('Too few Leica reference points')
    tree=cKDTree(target);neighbors=target[tree.query(target,k=min(20,len(target)))[1]]
    centered=neighbors-neighbors.mean(axis=1,keepdims=True)
    covariance=np.einsum('nki,nkj->nij',centered,centered)/neighbors.shape[1]
    _,eigenvectors=np.linalg.eigh(covariance);normals=eigenvectors[:,:,0]
    distances,_=tree.query(source)
    selected=source[distances<max_distance]
    if len(selected)>12000:selected=selected[::int(np.ceil(len(selected)/12000))]
    if len(selected)<100:return np.eye(4),dict(status='insufficient_overlap',correspondences=len(selected),applied=False)
    before=float(np.median(tree.query(selected)[0]))
    parameters=np.zeros(6);clipped=False;limit=np.r_[np.full(3,np.deg2rad(5)),np.full(3,.10)]
    for _ in range(12):
        transformed=Rotation.from_rotvec(parameters[:3]).apply(selected)+parameters[3:]
        distance,index=tree.query(transformed)
        keep=distance<=min(max_distance,max(.015,float(np.quantile(distance,.85)))+1e-9)
        if keep.sum()<100:break
        points=selected[keep];matches=target[index[keep]];directions=normals[index[keep]]
        def residual(values):
            moved=Rotation.from_rotvec(values[:3]).apply(points)+values[3:]
            geometry=np.sum((moved-matches)*directions,axis=1)/np.sqrt(len(points))
            return np.r_[geometry,values[:3]*.015,values[3:]*.10]
        fit=least_squares(residual,parameters,bounds=(-limit,limit),loss='soft_l1',f_scale=.002,max_nfev=40)
        candidate=fit.x
        for part,bound in [(slice(0,3),np.deg2rad(5)),(slice(3,6),.10)]:
            norm=np.linalg.norm(candidate[part])
            if norm>=bound:candidate[part]*=bound*.999/norm;clipped=True
        change=np.linalg.norm(candidate-parameters);parameters=candidate
        if change<1e-6:break
    transformed=Rotation.from_rotvec(parameters[:3]).apply(selected)+parameters[3:]
    after=float(np.median(tree.query(transformed)[0]));angle=float(np.linalg.norm(parameters[:3]));translation=float(np.linalg.norm(parameters[3:]))
    accepted=after<before and angle<np.deg2rad(5) and translation<.10
    delta=np.eye(4)
    if accepted:delta[:3,:3]=Rotation.from_rotvec(parameters[:3]).as_matrix();delta[:3,3]=parameters[3:]
    return delta,dict(status='accepted' if accepted else 'rejected_no_safe_improvement',applied=bool(accepted),
        correspondences=len(selected),median_distance_before_m=before,median_distance_after_m=after if accepted else before,candidate_median_distance_m=after,correction_clipped_to_bounds=clipped,
        rotation_degrees=float(np.rad2deg(angle)),translation_m=translation,correction_world_4x4=delta.tolist(),
        method='regularized robust point-to-plane ICP against annotated Leica part; not a joint/GT error',
        planar_warning='A flat part does not constrain all six pose dimensions; calibration prior and bounds constrain weak directions')


def fuse_leica(scene,xyz):
    folder=scene/'colmap/sparse_unmasked/0';cameras,images,points=read_model(folder)
    observations={iid:(list(image.xys),list(image.point3D_ids)) for iid,image in images.items()}
    depths={iid:np.load(scene/'depth_npy'/Path(image.name).with_suffix('.npy')) for iid,image in images.items()}
    colors={iid:cv2.imread(str(scene/'rgb'/image.name))[:,:,::-1] for iid,image in images.items()}
    pose_metadata=json.loads((scene/'metadata/cameras.json').read_text())['cameras']
    transforms={c['name']:np.linalg.inv(np.asarray(c['pose_matrix_4x4'])) for c in pose_metadata if c.get('camera_motion','static')!='moving'}
    added=0;pid=max(points,default=0)+1
    for point in xyz:
        ids=[];indices=[];rgb=[]
        for iid,image in images.items():
            camera=cameras[image.camera_id];fx,fy,cx,cy=camera.params
            local=transforms[image.name.split('/')[0]]@np.r_[point,1]
            if local[2]<=0:continue
            uv=np.array([fx*local[0]/local[2]+cx,fy*local[1]/local[2]+cy]);x,y=np.rint(uv).astype(int)
            if not(0<=x<camera.width and 0<=y<camera.height):continue
            measured=depths[iid][y,x]
            if not np.isfinite(measured) or measured<=0 or abs(float(measured)-local[2])>.05:continue
            ids.append(iid);indices.append(len(observations[iid][0]));observations[iid][0].append(uv);observations[iid][1].append(pid)
            rgb.append(colors[iid][y,x])
        if ids:
            points[pid]=Point3D(pid,point,np.rint(np.mean(rgb,axis=0)).astype(np.uint8),0.,np.asarray(ids,np.int32),np.asarray(indices,np.int32))
            added+=1;pid+=1
    images={iid:image._replace(xys=np.asarray(observations[iid][0]).reshape(-1,2),point3D_ids=np.asarray(observations[iid][1],np.int64)) for iid,image in images.items()}
    for extension in ['.txt','.bin']:write_model(cameras,images,points,str(folder),extension)
    write_ply(scene/'pointcloud/points3D.ply',np.array([p.xyz for p in points.values()]),np.array([p.rgb for p in points.values()]))
    for path in [scene/'colmap/export_summary.json',scene/'metadata/colmap_export.json']:
        summary=json.loads(path.read_text());summary.update(num_points3D=len(points),leica_points_added=added)
        write_json(path,summary)
    return dict(leica_points_total=len(xyz),leica_points_with_depth_supported_observations=added,
                observation_provenance='Leica projections with RGB-D Z agreement within 5cm; projected observations are not measured SfM keypoints',
                final_state_leica_fused=False)


def prepare_scene(scene,start=22):
    scene=Path(scene).resolve();trim_scene(scene,start);refresh_frame_metadata(scene)
    report_file=scene/'metadata/hoi_leica_alignment.json'
    if report_file.exists():
        print(f'{scene.name}: already aligned; preserved existing correction',flush=True);return
    progress=scene/'metadata/hoi_alignment_in_progress.json'
    if progress.exists():raise ValueError(f'Interrupted alignment; inspect {progress} and its backup before retrying')
    write_json(progress,dict(start_frame=start))
    reference=read_leica(scene/'pointcloud/leica_moving_part.ply')
    metadata=json.loads((scene/'metadata/cameras.json').read_text());reports=[];sensors=[]
    for camera in metadata['cameras']:
        if camera.get('camera_motion','static')=='moving':continue
        source,_=depth_world(scene,camera)
        correction,report=register_pose(source,reference);report['camera']=camera['name'];reports.append(report)
        pose=correction@np.asarray(camera['pose_matrix_4x4']);camera['pose_matrix_4x4']=pose.tolist()
        path=scene/f'metadata/poses/{camera["name"]}.json';data=json.loads(path.read_text())
        for row in data['frames']:
            row['pose_matrix_4x4']=pose.tolist();row['world_to_camera_matrix_4x4']=np.linalg.inv(pose).tolist()
        data['pose_source']='fixed optical calibration refined against Leica moving-part reference; see hoi_leica_alignment.json'
        write_json(path,data)
        k=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]])
        sensors.append(dict(pose=pose,target_k=k))
        print(f'{scene.name}/{camera["name"]}: {report["status"]}; median distances {report.get("median_distance_before_m")} -> {report.get("median_distance_after_m")}',flush=True)
    write_json(scene/'metadata/cameras.json',metadata)
    backup=scene/f'metadata/before_start_frame_{start:06d}'
    for filename in ['points3D.ply','points3D_final.ply']:
        if (scene/'pointcloud'/filename).exists():
            (backup/'pointcloud').mkdir(exist_ok=True);shutil.copyfile(scene/'pointcloud'/filename,backup/'pointcloud'/filename)
    for model in ['colmap','colmap_final']:
        if (scene/model).exists():shutil.move(scene/model,backup/model)
    count=json.loads((scene/'metadata/hoi_conversion.json').read_text())['num_frames']
    export_endpoint(scene,sensors,0,'start',4,.005)
    export_endpoint(scene,sensors,count-1,'end',4,.005)
    fusion=fuse_leica(scene,reference)
    standardize_scene(scene)
    write_json(report_file,dict(start_frame_original=start,initial_frame=0,cameras=reports,fusion=fusion,
        reference='pointcloud/leica_moving_part.ply',coordinate_frame='hoi_leica_world',
        initial_leica_state_assumption='Static Leica part is the initial reference, as requested; final cloud uses final RGB-D only'))
    progress.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,default=22)
    args=parser.parse_args()
    for scene in args.input:prepare_scene(scene,args.start_frame)

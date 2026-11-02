#!/usr/bin/env python3
"""Convert confirmed HoI! object interactions from static iPhones to standard RGB-D/COLMAP scenes.

Read-only sources, ZIP or extracted streams. Masks are a separate step; no fake
foreground, joint axes, fixed Aria poses, or Leica/video mask correspondences.
Moving ego RGB/relative-depth and optical poses are exported in a separate namespace.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import copy
import os
import tempfile
import sys
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile
from collections import Counter
from contextlib import ExitStack

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from read_write_model import Camera, Image, Point3D, write_model
from hoi_ego import prepare_ego, export_ego, leica_part_path, export_leica
from standardize_hoi_cameras import standardize_scene

MANIFEST_URL = 'https://hoi-dataset.ethz.ch/static/data/manifest.json'
FILE_URL = 'https://bonndata.uni-bonn.de/api/access/datafile/{}'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Metadata:
    def __init__(self, source, output, overlay=None, fetch=False, dry_run=False):
        self.source, self.output, self.overlay = source, output, overlay
        self.fetch, self.dry_run = fetch, dry_run
        self.index, self.cache, self.provenance = None, {}, {}

    def read(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe metadata path')
        key = relative.as_posix()
        if key in self.cache:
            return self.cache[key]
        candidates = [self.source/relative]
        if self.overlay:
            candidates.append(self.overlay/relative)
        candidates.append(self.output/'.hoi_metadata'/relative)
        found = next((p for p in candidates if p.is_file()), None)
        origin = str(found) if found else None
        if found:
            raw = found.read_bytes()
        elif self.fetch:
            if self.index is None:
                self.index = json.load(urllib.request.urlopen(MANIFEST_URL, timeout=90))
            entries = [e for e in self.index if e['path'] == key]
            if len(entries) != 1 or not 0 < entries[0]['size'] <= 16*1024*1024:
                raise ValueError(f'No unique small metadata file in official index: {key}')
            origin = FILE_URL.format(entries[0]['id'])
            raw = urllib.request.urlopen(origin, timeout=90).read(16*1024*1024+1)
            if len(raw) > 16*1024*1024:
                raise ValueError(f'Metadata response too large: {key}')
        else:
            raise FileNotFoundError(f'Missing HoI! metadata: {key}. Supply --metadata-root or --fetch-missing-metadata; calibration/windows will not be guessed.')
        value = json.loads(raw)
        self.cache[key] = value
        self.provenance[key] = dict(origin=origin, sha256=hashlib.sha256(raw).hexdigest())
        if not found and not self.dry_run:
            target=self.output/'.hoi_metadata'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        return value


class Stream:
    def __init__(self, root, label, suffix):
        self.root, self.label, self.suffix = root, label, suffix
        self.archive = None
        self.members = {}
        if (root/label).is_dir():
            self.members.update({int(p.stem): p for p in (root/label).glob('*'+suffix)})
        if (root/(label+'.zip')).is_file():
            self.archive = zipfile.ZipFile(root/(label+'.zip'))
            for name in self.archive.namelist():
                path=PurePosixPath(name)
                if path.parent == PurePosixPath(label) and path.suffix == suffix and path.stem.isdecimal():
                    self.members.setdefault(int(path.stem),name)
        if not self.members:
            self.close()
            raise ValueError(f'Empty/missing {label} stream: {root}')

    def read(self, stamp):
        member=self.members[int(stamp)]
        return member.read_bytes() if isinstance(member,Path) else self.archive.read(member)

    def close(self):
        if self.archive:
            self.archive.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.close()


def csv_rows(root, label):
    file=root/label/'data.csv'
    if file.is_file():
        return list(csv.DictReader(io.StringIO(file.read_text())))
    with zipfile.ZipFile(root/(label+'.zip')) as archive:
        return list(csv.DictReader(io.StringIO(archive.read(label+'/data.csv').decode())))


def nearest_indices(available, targets):
    right=np.searchsorted(available,targets).clip(0,len(available)-1)
    left=(right-1).clip(0,len(available)-1)
    return np.where(np.abs(available[left]-targets)<=np.abs(available[right]-targets),left,right)


def validate_pose(pose):
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-5):
        raise ValueError('Invalid optical camera-to-world transform')
    return pose


def sensor_info(root, source, metadata, name, size):
    relative=root.relative_to(source)
    info=metadata.read(relative/'metadata')
    sync=metadata.read(relative/'time_sync_info.json')
    if sync.get('timestamps_aligned') is not True:
        raise ValueError(f'{root}: source timestamps are not aligned; run official time alignment first')
    width,height=int(info['w']),int(info['h'])
    k=np.array(info['K'],dtype=float).reshape(3,3).T  # Record3D stores column-major K.
    if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError(f'{root}: invalid intrinsics')
    coeffs=np.asarray(info.get('perFrameIntrinsicCoeffs',[]),dtype=float)
    source_times=np.rint(np.asarray(info['frameTimestamps'],dtype=float)*1e9).astype(np.int64)+int(sync['delta'])
    if len(source_times)<2 or np.any(np.diff(source_times)<=0):
        raise ValueError('Metadata frame timestamps must strictly increase')
    if coeffs.size and (coeffs.shape!=(len(source_times),4) or not np.isfinite(coeffs).all() or np.any(coeffs[:,:2]<=0)):
        raise ValueError('Invalid per-frame intrinsic coefficients')
    rows=csv_rows(root,'poses_aligned')
    pose_times=np.array([int(row['timestamp']) for row in rows],dtype=np.int64)
    if len(set(pose_times))!=len(pose_times) or np.any(np.diff(pose_times)<=0):
        raise ValueError('Aligned pose timestamps must strictly increase')
    translations=np.array([[float(row[k]) for k in ['tx_world_cam','ty_world_cam','tz_world_cam']] for row in rows])
    quats=np.array([[float(row[k]) for k in ['qx_world_cam','qy_world_cam','qz_world_cam','qw_world_cam']] for row in rows])
    rotations=Rotation.from_quat(quats).as_matrix()
    if np.max(np.linalg.norm(translations-translations[0],axis=1))>1e-5 or not np.allclose(rotations,rotations[0],atol=1e-5):
        raise ValueError(f'{root}: moving camera; fixed-view export supports static iPhones only')
    pose=np.eye(4);pose[:3,:3]=rotations[0];pose[:3,3]=translations[0]
    target_k=k.copy();target_k[0]*=size[0]/width;target_k[1]*=size[1]/height
    target_k[0,2]=size[0]/2;target_k[1,2]=size[1]/2
    return dict(root=root,name=name,info=info,k=k,target_k=target_k,coeffs=coeffs,
                source_times=source_times,pose_times=pose_times,pose=validate_pose(pose))


def plan_frames(sensors, streams, window, fps, tolerance_ms):
    start,end=int(window['start_ns']),int(window['end_ns'])
    if not 0<=start<end:
        raise ValueError('Invalid confirmed interaction window')
    targets=np.arange(start,end+1,int(round(1e9/fps)),dtype=np.int64)
    valid=np.ones(len(targets),dtype=bool);chosen=[]
    tolerance=int(round(tolerance_ms*1e6))
    for sensor,(rgb,depth) in zip(sensors,streams):
        stamps=np.array(sorted(set(rgb.members)&set(depth.members)),dtype=np.int64)
        stamps=stamps[(stamps>=start-tolerance)&(stamps<=end+tolerance)]
        if not len(stamps):raise ValueError('No RGB/depth pairs inside object window')
        picks=stamps[nearest_indices(stamps,targets)]
        metadata_indices=nearest_indices(sensor['source_times'],picks)
        if np.any(np.abs(sensor['source_times'][metadata_indices]-picks)>2000):
            raise ValueError('Frame timestamps do not match metadata after time-sync delta; intrinsics cannot be assigned safely')
        pose_indices=nearest_indices(sensor['pose_times'],picks)
        valid &= np.abs(picks-targets)<=tolerance
        valid &= np.abs(sensor['pose_times'][pose_indices]-picks)<=tolerance
        chosen.append((picks,metadata_indices))
    rows=[];previous=[None]*len(sensors)
    for i in np.flatnonzero(valid):
        picked=[int(picks[i]) for picks,_ in chosen]
        if any(current==old for current,old in zip(picked,previous)):continue
        if max(picked)-min(picked)>tolerance:continue
        rows.append(dict(target_ns=int(targets[i]),source_ns=picked,metadata_indices=[int(idx[i]) for _,idx in chosen]))
        previous=picked
    if len(rows)<2:raise ValueError('Fewer than two synchronized frames within the tolerance')
    return rows


def rectify(rgb_bytes, depth_bytes, sensor, metadata_index, size):
    image=cv2.imdecode(np.frombuffer(rgb_bytes,dtype=np.uint8),cv2.IMREAD_COLOR)
    raw=np.load(io.BytesIO(depth_bytes),allow_pickle=False)
    info=sensor['info'];w,h=int(info['w']),int(info['h'])
    if image is None or image.shape[:2]!=(h,w) or raw.shape!=(int(info['dh']),int(info['dw'])) or raw.dtype!=np.float32:
        raise ValueError('RGB/depth dimensions or depth dtype differ from iPhone metadata')
    k=sensor['k'].copy()
    if sensor['coeffs'].size:
        fx,fy,cx,cy=sensor['coeffs'][metadata_index];k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    yy,xx=np.mgrid[:size[1],:size[0]]
    target=sensor['target_k']
    mapx=((xx-target[0,2])/target[0,0]*k[0,0]+k[0,2]).astype(np.float32)
    mapy=((yy-target[1,2])/target[1,1]*k[1,1]+k[1,2]).astype(np.float32)
    rgb=cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    clean=np.where(np.isfinite(raw)&(raw>0),raw,0).astype(np.float32)
    # Depth is already registered to RGB at a lower resolution, with the same field of view.
    depth=cv2.remap(clean,mapx*(raw.shape[1]/w),mapy*(raw.shape[0]/h),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    depth[(mapx<0)|(mapx>w-1)|(mapy<0)|(mapy>h-1)]=0
    if not np.any(depth>0):raise ValueError('No positive depth after rectification')
    return rgb,depth


def save_png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):raise OSError(path)


def write_ply(path,xyz,rgb):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point,color in zip(xyz,rgb):stream.write(' '.join(map(str,[*point,0,0,0,*color]))+'\n')


def export_endpoint(output,sensors,frame,state,stride,voxel):
    cameras,images,xyzs,rgbs,observations={}, {}, [], [], []
    for cid,sensor in enumerate(sensors,1):
        name=f'cam_{cid-1:03d}/frame_{frame:06d}.png'
        depth=np.load(output/'depth_npy'/Path(name).with_suffix('.npy'),allow_pickle=False)
        color=cv2.imread(str(output/'rgb'/name))[:,:,::-1]
        yy,xx=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
        z=depth[yy,xx];valid=np.isfinite(z)&(z>0);x,y,z=xx[valid],yy[valid],z[valid]
        k=sensor['target_k'];pose=sensor['pose'];r=pose[:3,:3]
        local=np.column_stack([(x-k[0,2])/k[0,0]*z,(y-k[1,2])/k[1,1]*z,z])
        world=local@r.T+pose[:3,3];colors=color[y,x]
        xyzs.append(world);rgbs.append(colors);observations.extend((cid,int(u),int(v)) for u,v in zip(x,y))
        q=Rotation.from_matrix(r.T).as_quat()[[3,0,1,2]]
        cameras[cid]=Camera(cid,'PINHOLE',depth.shape[1],depth.shape[0],np.array([k[0,0],k[1,1],k[0,2],k[1,2]]))
        images[cid]=Image(cid,q,-r.T@pose[:3,3],cid,name,np.empty((0,2)),np.empty(0,dtype=np.int64))
        write_ply(output/f'pointcloud/views/{state}/cam_{cid-1:03d}.ply',world,colors)
    xyz,rgb=np.concatenate(xyzs),np.concatenate(rgbs)
    if not len(xyz):raise ValueError('No endpoint depth points')
    _,inverse=np.unique(np.floor(xyz/voxel).astype(np.int64),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    fused=np.column_stack([np.bincount(inverse,weights=xyz[:,axis])/counts for axis in range(3)])
    colors=np.rint(np.column_stack([np.bincount(inverse,weights=rgb[:,axis])/counts for axis in range(3)])).clip(0,255).astype(np.uint8)
    tracks=[{} for _ in counts]
    for index,group in enumerate(inverse):
        cid,_,_=observations[index]
        previous=tracks[group].get(cid)
        if previous is None or np.linalg.norm(xyz[index]-fused[group])<np.linalg.norm(xyz[previous]-fused[group]):tracks[group][cid]=index
    points={};image_xy={cid:[] for cid in images};image_pids={cid:[] for cid in images}
    for group,members in enumerate(tracks):
        pid=group+1;iids=[];indices=[];errors=[]
        for cid,index in members.items():
            _,x,y=observations[index];iids.append(cid);indices.append(len(image_xy[cid]))
            image_xy[cid].append((x,y));image_pids[cid].append(pid)
            pose=sensors[cid-1]['pose'];local=pose[:3,:3].T@(fused[group]-pose[:3,3]);uv=sensors[cid-1]['target_k']@local
            if local[2]<=0:raise ValueError('Voxel fusion crossed the optical plane; reduce voxel size')
            errors.append(np.sum((uv[:2]/uv[2]-[x,y])**2))
        points[pid]=Point3D(pid,fused[group],colors[group],float(np.sqrt(np.mean(errors))),np.array(iids,dtype=np.int32),np.array(indices,dtype=np.int32))
    images={cid:image._replace(xys=np.array(image_xy[cid],dtype=float).reshape(-1,2),point3D_ids=np.array(image_pids[cid],dtype=np.int64)) for cid,image in images.items()}
    model='colmap' if state=='start' else 'colmap_final';folder=output/model/'sparse_unmasked/0';folder.mkdir(parents=True)
    for ext in ['.txt','.bin']:write_model(cameras,images,points,str(folder),ext)
    for image in images.values():
        save_png(output/model/'images'/image.name,cv2.imread(str(output/'rgb'/image.name)))
    ply_name='points3D.ply' if state=='start' else 'points3D_final.ply'
    write_ply(output/'pointcloud'/ply_name,fused,colors)
    summary=dict(endpoint_frame=frame,num_images=len(images),num_views=len(images),num_cameras=len(cameras),num_points3D=len(points),
                 sparse_dir=None,sparse_unmasked_dir=str(folder),masking_pending=True,pointcloud_ply=str(output/'pointcloud'/ply_name),
                 source_points=len(xyz),voxel_size_m=voxel,part_labels_available=False,joint_axis_gt_available=False)
    write_json(output/model/'export_summary.json',summary)
    write_json(output/'metadata'/('colmap_export.json' if state=='start' else 'colmap_export_final.json'),summary)


def upgrade_scene(plan,metadata,args,output):
    """Add auxiliary streams using the original export's frame IDs and calibration."""
    if (output/'conversion_in_progress.json').exists():
        raise ValueError('Existing static conversion is incomplete; upgrade requires a completed scene')
    previous=json.loads((output/'metadata/hoi_conversion.json').read_text())
    for key in ['location','session','object_index','recording_type']:
        expected=args.recording_type if key=='recording_type' else plan[key]
        if previous.get(key)!=expected:raise ValueError(f'Existing scene {key} differs from source plan')
    for key in ['start_ns','end_ns','index','single_word_description']:
        if previous['window'].get(key)!=plan['window'].get(key):raise ValueError('Existing interaction window differs from source plan')
    if previous['view_modules']!=[name for name,_ in plan['sensors']]:
        raise ValueError('Existing camera selection differs; use the original --sensors selection')
    cameras=[c for c in json.loads((output/'metadata/cameras.json').read_text())['cameras'] if c.get('camera_motion','static')!='moving']
    if len(cameras)!=len(previous['view_modules']) or [c['source_camera_name'] for c in cameras]!=previous['view_modules']:
        raise ValueError('Existing camera metadata is inconsistent')
    sizes={(int(c['width']),int(c['height'])) for c in cameras}
    if len(sizes)!=1:raise ValueError('Existing cameras have different export sizes')
    auxiliary_args=copy.copy(args);auxiliary_args.size=list(sizes.pop())
    with (output/'metadata/frame_timestamps.csv').open() as stream:rows=list(csv.DictReader(stream))
    grouped={c['name']:[] for c in cameras}
    for row in rows:
        grouped[row['camera']].append(row)
    frame_count=int(previous['num_frames'])
    targets=None
    for camera in cameras:
        items=grouped[camera['name']]
        if [int(row['frame_index']) for row in items]!=list(range(frame_count)):
            raise ValueError('Existing static frame indices are incomplete or reordered')
        times=[int(row['target_timestamp_ns']) for row in items]
        if targets is not None and times!=targets:raise ValueError('Existing static target timestamps disagree across cameras')
        targets=times
        for number in range(frame_count):
            for folder,suffix in [('rgb','.png'),('depth_npy','.npy'),('depth_png','.png')]:
                if not (output/folder/camera['name']/f'frame_{number:06d}{suffix}').is_file():
                    raise ValueError(f'Incomplete existing {folder}/{camera["name"]}, frame {number}')
    if frame_count<2 or any(b<=a for a,b in zip(targets,targets[1:])):raise ValueError('Invalid existing target timeline')
    frames=[dict(target_ns=stamp) for stamp in targets]
    leica_source,leica_count=leica_part_path(plan,auxiliary_args)
    with ExitStack() as stack:
        ego=prepare_ego(plan,auxiliary_args,metadata,frames,stack,Stream,validate_pose)
        for sensor in ego:sensor['total_frames']=frame_count
        counts=', '.join(f"{s['module']}:{len(s['rows'])}" for s in ego) or 'none'
        print(f'{plan["name"]}: upgrade {frame_count} existing frames; ego [{counts}]; Leica {leica_count} points; static files preserved',flush=True)
        result=dict(scene=plan['name'],frames=frame_count,views=len(cameras),ego_views=len(ego),leica_part_points=leica_count,upgraded=True,output=str(output))
        if args.dry_run:return result
        # Stage only new auxiliary data on the target drive, never in the input tree.
        with tempfile.TemporaryDirectory(prefix='.hoi_aux_',dir=output) as temporary:
            staged=Path(temporary);static=[]
            for camera in cameras:
                name=camera['name'];pose=validate_pose(np.asarray(camera['pose_matrix_4x4'],dtype=float))
                pose_rows=[]
                for row in grouped[name]:
                    number=int(row['frame_index']);filename=f'frame_{number:06d}'
                    pose_rows.append(dict(frame_index=number,target_timestamp_ns=int(row['target_timestamp_ns']),
                        source_timestamp_ns=int(row['source_timestamp_ns']),matching_error_ns=int(row['matching_error_ns']),
                        source_metadata_index=int(row['source_metadata_index']),rgb_path=f'rgb/{name}/{filename}.png',
                        depth_path=f'depth_npy/{name}/{filename}.npy',pose_matrix_4x4=pose.tolist(),world_to_camera_matrix_4x4=np.linalg.inv(pose).tolist()))
                k=[[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]]
                info=dict(camera=name,source_camera_name=camera['source_camera_name'],camera_motion='static',
                    width=camera['width'],height=camera['height'],intrinsics_matrix_3x3=k,coordinate_frame='hoi_leica_world',
                    pose_convention='opencv_optical_camera_to_world',pose_is_estimated=True,pose_source='preserved existing optical calibration',
                    depth_units='meters, optical camera Z; zero missing',depth_is_metric=True,depth_projection_to_world_valid=True,
                    num_frames=frame_count,poses_file=f'metadata/poses/{name}.json')
                write_json(staged/info['poses_file'],dict(info,frames=pose_rows));static.append(info)
            egos=export_ego(ego,staged,auxiliary_args.size,len(cameras),write_json,save_png)
            leica=export_leica(plan,staged,leica_source,leica_count,write_json)
            write_json(staged/'metadata/camera_streams.json',dict(coordinate_frame='hoi_leica_world',cameras=static+egos))
            write_json(staged/'metadata/hoi_auxiliary_export.json',dict(ego_views=egos,leica_moving_part=leica,
                static_conversion_preserved=True,metadata_files=metadata.provenance))
            # Replace only generated auxiliary paths; roll them back if publication fails.
            paths=['metadata/poses','pointcloud/leica_moving_part.ply','metadata/leica_moving_part.json',
                   'metadata/camera_streams.json','metadata/hoi_auxiliary_export.json']
            if ego:paths[0:0]=['ego/rgb','ego/depth_relative_npy']
            published=[];backed_up=[]
            try:
                for relative in paths:
                    target=output/relative;backup=staged/'backup'/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        backup.parent.mkdir(parents=True,exist_ok=True);os.replace(target,backup);backed_up.append(relative)
                    os.replace(staged/relative,target);published.append(relative)
            except BaseException:
                for relative in reversed(published):os.replace(output/relative,staged/relative)
                for relative in reversed(backed_up):os.replace(staged/'backup'/relative,output/relative)
                raise
    standardize_scene(output)
    if args.prepare_start_frame is not None:
        from refine_hoi_geometry import prepare_scene
        prepare_scene(output,args.prepare_start_frame)
    return result


def convert_scene(plan,metadata,args):
    if '/' in plan['name'] or '\\' in plan['name'] or plan['name'] in ['.','..']:
        raise ValueError('Unsafe output scene name')
    output=args.output_root/plan['name']
    if output.exists():
        if args.upgrade_existing:return upgrade_scene(plan,metadata,args,output)
        raise FileExistsError(f'Output exists: {output}. Use --upgrade-existing to add ego/pose/Leica exports while preserving static data.')
    with ExitStack() as stack:
        sensors=[];streams=[]
        for module,root in plan['sensors']:
            sensors.append(sensor_info(root,args.input,metadata,module,args.size))
            streams.append((stack.enter_context(Stream(root,'camera_rgb','.jpg')),stack.enter_context(Stream(root,'camera_depth','.npy'))))
        frames=plan_frames(sensors,streams,plan['window'],args.fps,args.tolerance_ms)
        full_frame_count=len(frames)
        if args.max_frames:frames=frames[:args.max_frames]
        if len(frames)<2:raise ValueError('At least two frames required for endpoints')
        leica_source,leica_count=leica_part_path(plan,args)
        ego_sensors=prepare_ego(plan,args,metadata,frames,stack,Stream,validate_pose)
        for sensor in ego_sensors:sensor['total_frames']=len(frames)
        ego_counts=', '.join(f"{s['module']}:{len(s['rows'])}" for s in ego_sensors) or 'none'
        print(f'{plan["name"]}: {len(frames)} frames x {len(sensors)} static views; ego frames [{ego_counts}]; Leica part {leica_count} points; {plan["description"]}',flush=True)
        if args.dry_run:return dict(scene=plan['name'],frames=len(frames),views=len(sensors),ego_views=[dict(module=s['module'],frames=len(s['rows'])) for s in ego_sensors],leica_part_points=leica_count)
        output.mkdir(parents=True)
        write_json(output/'conversion_in_progress.json',dict(source=str(args.input),object_index=plan['object_index']))
        cameras=[];timestamp_rows=[];static_streams=[]
        for cid,(sensor,(rgb,depth)) in enumerate(zip(sensors,streams)):
            camera=f'cam_{cid:03d}';k=sensor['target_k']
            for folder in ['rgb','depth_npy','depth_png','mask']:(output/folder/camera).mkdir(parents=True)
            cameras.append(dict(id=cid+1,name=camera,source_camera_name=sensor['name'],width=args.size[0],height=args.size[1],
                fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],pose_matrix_4x4=sensor['pose'].tolist(),
                pose_convention='opencv_optical_camera_to_world',coordinate_frame='hoi_leica_world'))
            pose_rows=[]
            for number,row in enumerate(frames):
                stamp=row['source_ns'][cid];image,z=rectify(rgb.read(stamp),depth.read(stamp),sensor,row['metadata_indices'][cid],args.size)
                filename=f'frame_{number:06d}'
                save_png(output/f'rgb/{camera}/{filename}.png',image)
                np.save(output/f'depth_npy/{camera}/{filename}.npy',z,allow_pickle=False)
                save_png(output/f'depth_png/{camera}/{filename}.png',np.rint(z*1000).clip(0,65535).astype(np.uint16))
                timestamp_rows.append((number,camera,(row['target_ns']-frames[0]['target_ns'])/1e9,row['target_ns'],stamp,stamp-row['target_ns'],row['metadata_indices'][cid]))
                pose_rows.append(dict(frame_index=number,target_timestamp_ns=row['target_ns'],source_timestamp_ns=stamp,
                    matching_error_ns=stamp-row['target_ns'],source_metadata_index=row['metadata_indices'][cid],
                    rgb_path=f'rgb/{camera}/{filename}.png',depth_path=f'depth_npy/{camera}/{filename}.npy',
                    pose_matrix_4x4=sensor['pose'].tolist(),world_to_camera_matrix_4x4=np.linalg.inv(sensor['pose']).tolist()))
            stream=dict(camera=camera,source_camera_name=sensor['name'],camera_motion='static',
                width=args.size[0],height=args.size[1],intrinsics_matrix_3x3=k.tolist(),
                coordinate_frame='hoi_leica_world',pose_convention='opencv_optical_camera_to_world',
                depth_units='meters, optical camera Z; zero missing',depth_is_metric=True,depth_projection_to_world_valid=True,
                pose_is_estimated=True,pose_source='source aligned optical poses',num_frames=len(pose_rows),
                poses_file=f'metadata/poses/{camera}.json')
            write_json(output/stream['poses_file'],dict(stream,frames=pose_rows))
            static_streams.append(stream)
        ego_streams=export_ego(ego_sensors,output,args.size,len(sensors),write_json,save_png)
        leica_summary=export_leica(plan,output,leica_source,leica_count,write_json)
        write_json(output/'metadata/camera_streams.json',dict(coordinate_frame='hoi_leica_world',cameras=static_streams+ego_streams))
        write_json(output/'metadata/cameras.json',dict(cameras=cameras,coordinate_frame='hoi_leica_world',camera_motion='static'))
        with (output/'metadata/frame_timestamps.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['frame_index','camera','timestamp_s','target_timestamp_ns','source_timestamp_ns','matching_error_ns','source_metadata_index']);writer.writerows(timestamp_rows)
        write_json(output/'metadata/hoi_object.json',plan['annotation'])
        write_json(output/'metadata/hoi_conversion.json',dict(source=str(args.input),location=plan['location'],object_index=plan['object_index'],
            recording_type=args.recording_type,session=plan['session'],window=plan['window'],window_name=plan['window_name'],
            window_duration_s=(int(plan['window']['end_ns'])-int(plan['window']['start_ns']))/1e9,
            fps=args.fps,num_frames=len(frames),full_window_frame_count=full_frame_count,smoke_test_truncated=len(frames)<full_frame_count,output_frames=list(range(len(frames))),frame_naming='consecutive_zero_based',
            ego_views=ego_streams,leica_moving_part=leica_summary,
            view_modules=[s['name'] for s in sensors],max_matching_error_ns=max(abs(row[5]) for row in timestamp_rows),
            metadata_files=metadata.provenance,masks_generated=False,joint_type_gt_available=bool(plan['annotation'].get('class')),
            joint_axis_gt_available=False,depth_unit='meters, optical camera Z; zero missing',
            calibration='source per-frame intrinsics rectified to a constant pinhole K per view; source aligned static optical camera poses',
            synchronization='nearest timestamps with tolerance; time-aligned sources are not hardware synchronized'))
        for endpoint,state in [(0,'start'),(len(frames)-1,'end')]:export_endpoint(output,sensors,endpoint,state,args.point_stride,args.voxel_size)
        standardize_scene(output)
        if args.prepare_start_frame is not None:
            from refine_hoi_geometry import prepare_scene
            prepare_scene(output,args.prepare_start_frame)
        (output/'README.txt').write_text('HoI! object interaction: consecutive frame IDs, two static iPhone views.\nInitial COLMAP contains only frame 0 for all cameras; final COLMAP contains only the final frame, using per-frame ego poses.\nUnmasked models: sparse_unmasked/0. Generate object masks separately to publish sparse/0.\nAria cameras follow iPhones in shared rgb/, depth_npy/, depth_png/ and mask/; per-frame optical poses for every exported camera are in metadata/poses/.\nAria depth is nonmetric inverse depth: do not project as meters. Leica selected moving-part scan: pointcloud/leica_moving_part.ply.\nNo invented joint axes, white placeholder masks, or Leica video-mask projections.\nDepth NPY is metric Z; PNG is uint16 millimeters (saturated above 65.535m).\n')
        (output/'conversion_in_progress.json').unlink()
        return dict(scene=plan['name'],frames=len(frames),views=len(sensors),ego_views=len(ego_streams),leica_part_points=leica_count,output=str(output))


def discover_plans(args,metadata):
    locations=[args.input/name for name in args.locations] if args.locations else sorted(p for p in args.input.iterdir() if p.is_dir())
    plans=[]
    for location in locations:
        base=location/args.recording_type
        if not base.is_dir():continue
        modules={}
        for module in args.sensors:
            candidates=sorted(d for d in base.iterdir() if d.is_dir() and d.name.split(' ')[0]==module)
            if len(candidates)!=1:raise ValueError(f'{location.name}: expected one {module} module')
            sessions={d.name:d for d in candidates[0].iterdir() if d.is_dir() and ((d/'camera_rgb.zip').exists() or (d/'camera_rgb').is_dir())}
            modules[module]=sessions
        shared=set.intersection(*(set(sessions) for sessions in modules.values()))
        if not shared:raise ValueError(f'{location.name}: no shared iPhone recording sessions')
        annotation_path=location/'leica/001/instance_annotations_3d/instances.json'
        annotations={int(item['index']):item for item in json.loads(annotation_path.read_text())['items']} if annotation_path.exists() else {}
        for session in sorted(shared):
            prefix=location.name+'_';suffix='_'+args.recording_type
            if not session.startswith(prefix) or not session.endswith(suffix):raise ValueError(f'Unrecognized session name: {session}')
            interaction_range=session[len(prefix):-len(suffix)]
            windows=metadata.read(base.relative_to(args.input)/f'interaction_splitting_info_{interaction_range}_confirmed.json')
            for name,window in windows.items():
                if window.get('single_word_description')!=args.action:continue
                index=window.get('index')
                if type(index) is not int or index<1:raise ValueError(f'{name}: missing confirmed object index')
                if args.object_ids and index not in args.object_ids:continue
                annotation=annotations.get(index,dict(index=index))
                plans.append(dict(name=f'{location.name}_obj{index:03d}',location=location.name,session=session,window_name=name,window=window,
                    object_index=index,annotation=annotation,description=annotation.get('description','unlabelled object'),
                    sensors=[(module,modules[module][session]) for module in args.sensors]))
    counts=Counter(p['name'] for p in plans)
    for plan in plans:
        if counts[plan['name']]>1:plan['name']+='_'+plan['session']+'_'+plan['window_name']
    if not plans:raise ValueError('No confirmed object interactions match the requested locations/action/IDs')
    return plans


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='HoI! root containing location folders')
    parser.add_argument('--output-root',type=Path,required=True,help='New standard object scene folders and optional metadata cache go here')
    parser.add_argument('--locations',nargs='+')
    parser.add_argument('--recording-type',choices=['gripper','hand','wrist','umi'],default='gripper')
    parser.add_argument('--action',choices=['open','close'],default='open')
    parser.add_argument('--object-ids',type=int,nargs='+')
    parser.add_argument('--sensors',nargs='+',choices=['iphone_1','iphone_2'],default=['iphone_1','iphone_2'])
    parser.add_argument('--metadata-root',type=Path,help='Read-only overlay with missing metadata, preserving source-relative paths')
    parser.add_argument('--fetch-missing-metadata',action='store_true',help='Fetch small official metadata only; cache under output root; never edit input')
    parser.add_argument('--size',type=int,nargs=2,default=[640,480],metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--fps',type=float,default=30.)
    parser.add_argument('--tolerance-ms',type=float,default=20.)
    parser.add_argument('--point-stride',type=int,default=8)
    parser.add_argument('--voxel-size',type=float,default=.005)
    parser.add_argument('--max-frames',type=int,default=0,help='Smoke-test limit per interaction; zero means full confirmed interval')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-start-frame',type=int,help='After conversion/upgrade, trim to this original frame, refine fixed poses against Leica and fuse initial Leica/RGB-D geometry (e.g. 22)')
    parser.add_argument('--upgrade-existing',action='store_true',help='Refresh ego/pose/Leica auxiliary exports in completed scenes; preserve static RGB-D, masks and COLMAP; reuse saved frame times and resolution')
    args=parser.parse_args(argv)
    args.input=args.input.expanduser().resolve();args.output_root=args.output_root.expanduser().resolve()
    if args.metadata_root:args.metadata_root=args.metadata_root.expanduser().resolve()
    if args.output_root==args.input or args.output_root.is_relative_to(args.input) or args.input.is_relative_to(args.output_root):parser.error('Input/output must be separate non-nested directories')
    if min(args.size)<=0 or args.point_stride<=0 or not all(np.isfinite(v) and v>0 for v in [args.fps,args.tolerance_ms,args.voxel_size]) or args.max_frames<0 or args.max_frames==1 or len(set(args.sensors))!=len(args.sensors):parser.error('Invalid size, sampling, timing or sensor selection')
    metadata=Metadata(args.input,args.output_root,args.metadata_root,args.fetch_missing_metadata,args.dry_run)
    plans=discover_plans(args,metadata)
    results=[];failures=[]
    for plan in plans:
        try:results.append(convert_scene(plan,metadata,args))
        except Exception as error:
            failures.append(dict(scene=plan['name'],error=str(error)));print(f'FAILED {plan["name"]}: {error}',flush=True)
    if not args.dry_run and (results or not (args.output_root/'hoi_batch_summary.json').exists()):
        write_json(args.output_root/'hoi_batch_summary.json',dict(results=results,failures=failures))
    print(f'{"Validated" if args.dry_run else "Converted"} {len(results)} object interactions; {len(failures)} failures',flush=True)
    return int(bool(failures))


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(2)

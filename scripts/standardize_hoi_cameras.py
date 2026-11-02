#!/usr/bin/env python3
"""Move exported HoI! ego views into the common RGB/depth/camera interface."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mask_colmap_models import read_model
from read_write_model import Camera, Image, write_model


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def move_camera(source, target):
    if not source.exists():
        if not target.is_dir():raise FileNotFoundError(target)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        sources={p.name:p for p in source.iterdir()}
        targets={p.name:p for p in target.iterdir()}
        if sources.keys()!=targets.keys() or any(not p.is_file() or not targets[n].is_file() or
                hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(targets[n].read_bytes()).digest() for n,p in sources.items()):
            raise ValueError(f'Conflicting common and legacy camera files: {target}')
        shutil.rmtree(source)  # Only byte-identical legacy duplicates are removed.
    else:
        source.rename(target)


def add_endpoint_views(scene, streams, poses, frame, model):
    root=scene/model
    if not root.is_dir():return
    for variant in ['sparse_unmasked/0','sparse/0']:
        folder=root/variant
        if not folder.is_dir():continue
        cameras,images,points=read_model(folder)
        for info in streams:
            if info['camera_motion']!='moving':continue
            name=info['camera'];filename=f'{name}/frame_{frame:06d}.png'
            row=next((r for r in poses[name]['frames'] if r['frame_index']==frame),None)
            if row is None:raise ValueError(f'{name} has no pose/image for endpoint frame {frame}')
            previous=next((image for image in images.values() if image.name==filename),None)
            cid=previous.camera_id if previous else max(cameras,default=0)+1
            iid=previous.id if previous else max(images,default=0)+1
            k=np.asarray(info['intrinsics_matrix_3x3'])
            cameras[cid]=Camera(cid,'PINHOLE',info['width'],info['height'],np.array([k[0,0],k[1,1],k[0,2],k[1,2]]))
            w2c=np.asarray(row['world_to_camera_matrix_4x4'])
            q=Rotation.from_matrix(w2c[:3,:3]).as_quat()[[3,0,1,2]]
            # Preserve any existing observations; no nonmetric-depth points are invented.
            xy=previous.xys if previous else np.empty((0,2))
            ids=previous.point3D_ids if previous else np.empty(0,dtype=np.int64)
            images[iid]=Image(iid,q,w2c[:3,3],cid,filename,xy,ids)
            target=root/'images'/filename;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(scene/row['rgb_path'],target)
        for suffix in ['.txt','.bin']:write_model(cameras,images,points,str(folder),suffix)
    for path in [root/'export_summary.json',scene/'metadata'/('colmap_export.json' if model=='colmap' else 'colmap_export_final.json')]:
        if path.exists():
            data=json.loads(path.read_text());data.update(num_images=len(images),num_views=len(images),num_cameras=len(cameras),
                moving_views_included=True,ego_depth_fused=False)
            write_json(path,data)


def standardize_scene(scene):
    scene=Path(scene).resolve()
    manifest_path=scene/'metadata/camera_streams.json'
    manifest=json.loads(manifest_path.read_text());streams=manifest['cameras']
    conversion=json.loads((scene/'metadata/hoi_conversion.json').read_text())
    last=int(conversion['num_frames'])-1
    poses={s['camera']:json.loads((scene/s['poses_file']).read_text()) for s in streams}
    # Check endpoints before moving files or changing calibration metadata.
    for info in streams:
        if info['camera_motion']=='moving':
            indices={r['frame_index'] for r in poses[info['camera']]['frames']}
            if not {0,last}<=indices:raise ValueError(f'{info["camera"]}: missing initial/final pose; cannot create complete endpoint models')
    cameras_path=scene/'metadata/cameras.json'
    original=json.loads(cameras_path.read_text())
    cameras=[c for c in original['cameras'] if c.get('camera_motion','static')!='moving']
    for info in streams:
        if info['camera_motion']!='moving':continue
        name=info['camera'];meta=poses[name]
        move_camera(scene/'ego/rgb'/name,scene/'rgb'/name)
        move_camera(scene/'ego/depth_relative_npy'/name,scene/'depth_npy'/name)
        (scene/'mask'/name).mkdir(parents=True,exist_ok=True)
        preview=scene/'depth_png'/name;preview.mkdir(parents=True,exist_ok=True)
        for row in meta['frames']:
            filename=f'frame_{row["frame_index"]:06d}'
            row.update(rgb_path=f'rgb/{name}/{filename}.png',depth_path=f'depth_npy/{name}/{filename}.npy')
            depth=np.load(scene/row['depth_path'],allow_pickle=False)
            valid=np.isfinite(depth)&(depth>0);image=np.zeros(depth.shape,np.uint16)
            if valid.any():
                low,high=float(depth[valid].min()),float(depth[valid].max())
                if high>low:image[valid]=np.rint((depth[valid]-low)/(high-low)*65535).astype(np.uint16)
            if not cv2.imwrite(str(preview/(filename+'.png')),image):raise OSError(preview)
        info.update(depth_npy_directory=f'depth_npy/{name}',rgb_directory=f'rgb/{name}',
                    depth_png_encoding='per-frame normalized relative inverse-depth preview, uint16; not millimeters')
        meta.update(info);write_json(scene/info['poses_file'],meta)
        k=np.asarray(info['intrinsics_matrix_3x3'])
        cameras.append(dict(id=len(cameras)+1,name=name,source_camera_name=info['source_camera_name'],
            width=info['width'],height=info['height'],fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),
            camera_motion='moving',poses_file=info['poses_file'],pose_convention=info['pose_convention'],
            coordinate_frame='hoi_leica_world',depth_is_metric=False,depth_units='relative inverse depth'))
    add_endpoint_views(scene,streams,poses,0,'colmap')
    add_endpoint_views(scene,streams,poses,last,'colmap_final')
    original.update(cameras=cameras,camera_motion='mixed' if any(s['camera_motion']=='moving' for s in streams) else 'static',
                    moving_camera_pose_rule='Read metadata/poses/<camera>.json for every frame; never reuse one pose for moving views')
    write_json(cameras_path,original);write_json(manifest_path,manifest)
    for folder in [scene/'ego/rgb',scene/'ego/depth_relative_npy',scene/'ego']:
        if folder.is_dir() and not any(folder.iterdir()):folder.rmdir()
    print(f'{scene.name}: common RGB/depth/masks and endpoint COLMAP for {len(cameras)} cameras',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True,help='Existing HoI! scene folders')
    args=parser.parse_args()
    for scene in args.input:standardize_scene(scene)

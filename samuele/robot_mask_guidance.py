"""Project recorded robot tool poses into RGB views for SAM background prompts."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


def read_parquet_rows(parquet,python,columns):
    reader='import polars as pl,json,sys; print(json.dumps(pl.read_parquet(sys.argv[1],columns=json.loads(sys.argv[2])).to_dicts()))'
    result=subprocess.run([str(python),'-B','-c',reader,str(parquet),json.dumps(columns)],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'Cannot read Parquet with {python}: {result.stderr.strip()}')
    return json.loads(result.stdout)


def prepare(scene, parquet_python, max_offset_m=.15):
    scene=Path(scene)
    metadata=json.loads((scene/'metadata/real_conversion.json').read_text())
    cameras_path=scene/'metadata/cameras.json'
    cameras=json.loads(cameras_path.read_text())['cameras']
    parquet=scene/f'metadata/raw/episode_{metadata["episode"]:06d}.parquet'
    rows=read_parquet_rows(parquet,parquet_python,['frame_index','observation.state'])
    lookup={int(row['frame_index']):row['observation.state'][:6] for row in rows}
    poses=np.array([lookup[frame] for frame in metadata['source_frames']],dtype=float)
    if poses.shape!=(len(metadata['source_frames']),6) or not np.isfinite(poses).all():
        raise ValueError(f'Missing or invalid recorded metric robot poses: {parquet}')
    return project(poses,cameras,np.asarray(metadata['rectified_K'],dtype=float),dict(
        parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        camera_metadata_sha256=hashlib.sha256(cameras_path.read_bytes()).hexdigest(),
        source_frames=metadata['source_frames']),max_offset_m)


def project(poses,cameras,intrinsics,provenance,max_offset_m=.15):
    # Stop short of the observed tool tip so contact points on the object are not labelled background.
    if not np.isfinite(max_offset_m) or not 0<=max_offset_m<=.15:
        raise ValueError('Robot background offset extent must be between 0 and .15 meters')
    offsets=np.array([0.,.03,.06,.09,.12,.15])
    offsets=offsets[offsets<=max_offset_m]
    rotation=Rotation.from_rotvec(poses[:,3:6]).as_matrix()
    world=poses[:,:3,None]+rotation[:,:,2,None]*offsets[None,None,:]
    world=world.transpose(0,2,1)
    payload=dict(coordinate_system='image_pixels_xy',source='recorded observation.state tool poses; calibrated projection',
        position_unit='meters',orientation_unit='rotation_vector_radians',tool_axis_offsets_m=offsets.tolist(),
        provenance=provenance,cameras={})
    for camera in cameras:
        pose=np.linalg.inv(np.asarray(camera['pose_matrix_4x4'],dtype=float))
        local=world@pose[:3,:3].T+pose[:3,3]
        image=local@intrinsics.T
        uv=image[:,:,:2]/np.where(image[:,:,2:]>0,image[:,:,2:],np.nan)
        frames=[]
        for points,depth in zip(uv,local[:,:,2]):
            valid=np.isfinite(points).all(axis=1)&(depth>0)&(points[:,0]>=0)&(points[:,0]<camera['width'])&(points[:,1]>=0)&(points[:,1]<camera['height'])
            frames.append(points[valid].tolist())
        payload['cameras'][camera['name']]=frames
    return payload


def guidance_path(scene, root=None):
    return Path(root)/Path(scene).name/'robot_mask_guidance.json' if root else Path(scene)/'metadata/robot_mask_guidance.json'


def load_cached(scene,max_offset_m=.15):
    scene=Path(scene);file=guidance_path(scene)
    if not file.is_file(): return None
    payload=json.loads(file.read_text())
    metadata=json.loads((scene/'metadata/real_conversion.json').read_text())
    parquet=scene/f'metadata/raw/episode_{metadata["episode"]:06d}.parquet'
    expected=dict(parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        camera_metadata_sha256=hashlib.sha256((scene/'metadata/cameras.json').read_bytes()).hexdigest(),
        source_frames=metadata['source_frames'])
    offsets=[x for x in [0.,.03,.06,.09,.12,.15] if x<=max_offset_m]
    if payload.get('provenance')==expected and payload.get('tool_axis_offsets_m')==offsets:
        return payload
    return None

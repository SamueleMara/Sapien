"""Disambiguate text detections using a HoI! annotated part in Leica world coordinates."""
import json
from pathlib import Path

import numpy as np
import cv2


def large_qr_occluder(rgb):
    detector=cv2.QRCodeDetector()
    found,points=detector.detectMulti(rgb)
    if not found:found,points=detector.detect(rgb)
    if found:
        for polygon in points:
            area=abs(cv2.contourArea(np.asarray(polygon,dtype=np.float32)))
            if area>rgb.shape[0]*rgb.shape[1]*.08:return polygon.tolist()
    return None


def read_xyz(path):
    types={'float':'f4','float32':'f4','double':'f8','float64':'f8',
           'uchar':'u1','uint8':'u1','char':'i1','int8':'i1','int':'i4','uint':'u4',
           'short':'i2','ushort':'u2'}
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'ply':raise ValueError('Invalid Leica PLY')
        properties=[];count=None;vertex=False;encoding=None
        while True:
            raw=stream.readline()
            if not raw:raise ValueError('Missing PLY end_header')
            words=raw.decode('ascii').split()
            if not words:continue
            if words[0]=='end_header':break
            if words[0]=='format':encoding=words[1]
            if words[0]=='element':
                vertex=words[1]=='vertex'
                if vertex:count=int(words[2])
            if words[0]=='property' and vertex:
                if words[1] not in types:raise ValueError('Unsupported Leica PLY vertex property')
                properties.append((words[2],types[words[1]]))
        if count is None or count<=0 or not {'x','y','z'}<={p[0] for p in properties}:
            raise ValueError('Missing Leica XYZ vertices')
        if encoding=='ascii':
            values=np.loadtxt(stream,max_rows=count,ndmin=2)
            names=[p[0] for p in properties]
            xyz=values[:,[names.index(axis) for axis in 'xyz']]
        elif encoding in ['binary_little_endian','binary_big_endian']:
            endian='<' if encoding=='binary_little_endian' else '>'
            dtype=np.dtype([(name,endian+kind) for name,kind in properties])
            values=np.frombuffer(stream.read(count*dtype.itemsize),dtype=dtype,count=count)
            xyz=np.column_stack([values[axis] for axis in 'xyz']).astype(float)
        else:raise ValueError('Unsupported Leica PLY encoding')
    if len(xyz)!=count or not np.isfinite(xyz).all():raise ValueError('Invalid Leica part vertices')
    return xyz


class LeicaGuidance:
    def __init__(self):
        self.scenes={};self.poses={}

    def select(self,image_path,boxes):
        image_path=Path(image_path);scene=image_path.parents[2];camera=image_path.parent.name
        if scene not in self.scenes:
            metadata=json.loads((scene/'metadata/leica_moving_part.json').read_text())
            if metadata['coordinate_frame']!='hoi_leica_world' or metadata['units']!='meters':
                raise ValueError('Leica guidance needs metric points in the camera world frame')
            path=(scene/metadata['path']).resolve()
            if not path.is_relative_to(scene.resolve()):raise ValueError('Leica guidance cloud escapes scene')
            xyz=read_xyz(path);negative=np.empty((0,3))
            profile=scene/'metadata/leica_appliance_guidance.json'
            if profile.exists():
                appliance=json.loads(profile.read_text())
                xyz=np.asarray(appliance['positive_world_xyz'],dtype=float)
                negative=np.asarray(appliance['negative_world_xyz'],dtype=float).reshape(-1,3)
            self.scenes[scene]=(np.column_stack([xyz,np.ones(len(xyz))]),metadata,
                                np.column_stack([negative,np.ones(len(negative))]))
        key=(scene,camera)
        if key not in self.poses:
            data=json.loads((scene/f'metadata/poses/{camera}.json').read_text())
            if data['coordinate_frame']!='hoi_leica_world':raise ValueError('Pose and Leica coordinate frames differ')
            self.poses[key]=(data,{row['frame_index']:row for row in data['frames']})
        data,frames=self.poses[key];frame=int(image_path.stem.removeprefix('frame_'))
        row=frames[frame];world,metadata,negative=self.scenes[scene]
        def project(points):
            local=points@np.asarray(row['world_to_camera_matrix_4x4'],dtype=float).T
            projected=local[local[:,2]>1e-5,:3]@np.asarray(data['intrinsics_matrix_3x3'],dtype=float).T
            uv=projected[:,:2]/projected[:,2:3]
            uv=np.rint(uv).astype(int)
            return np.unique(uv[(uv[:,0]>=0)&(uv[:,0]<data['width'])&(uv[:,1]>=0)&(uv[:,1]<data['height'])],axis=0)
        uv=project(world);negative_uv=project(negative)
        details=dict(selection='leica_annotated_appliance',object_index=metadata['object_index'],
                     reference_projected_pixels=len(uv),reference_frame='static Leica scan',
                     selection_note='Spatial detection guidance only; not a per-frame moving-part mask or depth GT')
        if len(uv)<20:
            return None,dict(details,selection_status='insufficient_visible_reference')
        def prompt_points(pixels):
            points=[]
            if len(pixels):
                for quantile in [.15,.4,.65,.85]:
                    y=np.quantile(pixels[:,1],quantile)
                    near=pixels[np.abs(pixels[:,1]-y)<=max(2,np.ptp(pixels[:,1])*.06)]
                    if len(near):points.append(near[np.argmin(np.linalg.norm(near-np.median(near,axis=0),axis=1))].tolist())
            return points
        details.update(sam_positive_points=prompt_points(uv),sam_negative_points=prompt_points(negative_uv))
        supports=[]
        for x0,y0,x1,y1 in boxes:
            inside=(uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1)
            supports.append(int(inside.sum()))
        negative_support=[]
        for x0,y0,x1,y1 in boxes:
            negative_support.append(float(np.mean((negative_uv[:,0]>=x0)&(negative_uv[:,0]<=x1)&(negative_uv[:,1]>=y0)&(negative_uv[:,1]<=y1))) if len(negative_uv) else 0.)
        quality=[count/len(uv)-1.5*negative_support[i] for i,count in enumerate(supports)]
        eligible=[i for i,count in enumerate(supports) if count>=20 and count/len(uv)>=.30 and quality[i]>.10]
        if not eligible:
            lower=np.quantile(uv,.01,axis=0);upper=np.quantile(uv,.99,axis=0);margin=(upper-lower)*.12
            box=np.r_[np.maximum(0,lower-margin),np.minimum([data['width']-1,data['height']-1],upper+margin)]
            return None,dict(details,selection_status='leica_box_fallback',fallback_box=box.tolist(),requires_detector_fallback=True,
                             selection_note='No text detection supports this appliance; SAM uses the projected Leica appliance box and positive/negative points')
        # Tighter boxes win only when they cover essentially the same reference pixels.
        best=max(quality[i] for i in eligible)
        eligible=[i for i in eligible if quality[i]>=best-.05]
        index=min(eligible,key=lambda i:(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]))
        return index,dict(details,selection_status='selected',selected_box=np.asarray(boxes[index]).tolist(),
                          selected_reference_pixels=supports[index],reference_coverage=supports[index]/len(uv),
                          negative_reference_coverage=negative_support[index],requires_detector_fallback=negative_support[index]>.05)

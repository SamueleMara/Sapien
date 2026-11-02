import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import convert_hoi_dataset as hoi
from mask_colmap_models import read_model, apply_scene


def fixture(root):
    source=root/'raw';loc=source/'fridge_1';base=loc/'gripper'
    base.mkdir(parents=True)
    windows={'window_0':dict(start_ns=1_200_000_000,end_ns=1_500_000_000,index=1,single_word_description='open'),
             'window_1':dict(start_ns=1_600_000_000,end_ns=1_900_000_000,index=2,single_word_description='open')}
    (base/'interaction_splitting_info_1-2_confirmed.json').write_text(json.dumps(windows))
    annotation=loc/'leica/001/instance_annotations_3d/instances.json';annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(dict(items=[dict(index=1,description='door',**{'class':'revolute'}),dict(index=2,description='drawer',**{'class':'prismatic'})])))
    points=annotation.parent/'points';points.mkdir()
    for index in [1,2]:
        hoi.write_ply(points/f'{index:03d}.ply',np.array([[0.,0.,2.]]),np.array([[0,0,0]],dtype=np.uint8))
    for sensor in range(2):
        r=base/f'iphone_{sensor+1}/fridge_1_1-2_gripper';r.mkdir(parents=True)
        delta=1_000_000_000+sensor*5_000_000
        meta=dict(w=8,h=6,dw=4,dh=3,K=[8,0,0,0,8,0,4,3,1],frameTimestamps=[i/10 for i in range(11)],
                  perFrameIntrinsicCoeffs=[[8+i*.01,8+i*.01,4,3] for i in range(11)])
        (r/'metadata').write_text(json.dumps(meta));(r/'time_sync_info.json').write_text(json.dumps(dict(timestamps_aligned=True,delta=delta)))
        (r/'poses_aligned').mkdir();stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(['timestamp','tx_world_cam','ty_world_cam','tz_world_cam','qx_world_cam','qy_world_cam','qz_world_cam','qw_world_cam'])
        with zipfile.ZipFile(r/'camera_rgb.zip','w') as zr,zipfile.ZipFile(r/'camera_depth.zip','w') as zd:
            for i in range(2,11):  # Cropped frames must retain metadata indices 2..10.
                stamp=delta+i*100_000_000
                writer.writerow([stamp,sensor,0,0,0,0,0,1])
                image=np.full((6,8,3),i*15,dtype=np.uint8)
                zr.writestr(f'camera_rgb/{stamp}.jpg',cv2.imencode('.jpg',image)[1].tobytes())
                data=io.BytesIO();np.save(data,np.full((3,4),2+i*.1,dtype=np.float32),allow_pickle=False)
                zd.writestr(f'camera_depth/{stamp}.npy',data.getvalue())
        (r/'poses_aligned/data.csv').write_text(stream.getvalue())
    return source


class HoiConversionTests(unittest.TestCase):
    def command(self,source,output):
        return ['--input',str(source),'--output-root',str(output),'--fps','10','--size','8','6','--point-stride','1','--voxel-size','.001']

    def test_two_objects_consecutive_frames_endpoints_tracks_units_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'standard'
            before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            self.assertEqual(hoi.main(self.command(source,output)),0)
            for object_id,start,end in [(1,2.2,2.5),(2,2.6,2.9)]:
                scene=output/f'fridge_1_obj{object_id:03d}'
                data=json.loads((scene/'metadata/hoi_conversion.json').read_text())
                self.assertEqual(data['num_frames'],4);self.assertFalse(data['joint_axis_gt_available'])
                for cam in ['cam_000','cam_001']:
                    self.assertEqual(sorted(p.name for p in (scene/'rgb'/cam).iterdir()),[f'frame_{i:06d}.png' for i in range(4)])
                    self.assertFalse(any((scene/'mask'/cam).iterdir()))
                    z=np.load(scene/f'depth_npy/{cam}/frame_000000.npy')
                    self.assertEqual(z.dtype,np.float32);self.assertAlmostEqual(float(z[3,4]),start,places=5)
                    self.assertEqual(int(cv2.imread(str(scene/f'depth_png/{cam}/frame_000000.png'),-1)[3,4]),round(start*1000))
                for model,frame,depth in [('colmap',0,start),('colmap_final',3,end)]:
                    self.assertFalse((scene/model/'sparse').exists())
                    cameras,images,points=read_model(scene/model/'sparse_unmasked/0')
                    self.assertEqual(len(images),2);self.assertEqual(len(cameras),2)
                    self.assertEqual({i.name for i in images.values()},{f'cam_000/frame_{frame:06d}.png',f'cam_001/frame_{frame:06d}.png'})
                    self.assertEqual(len(list((scene/model/'images').glob('*/*.png'))),2)
                    for pid,point in points.items():
                        self.assertAlmostEqual(float(point.xyz[2]),depth,places=5)
                        for iid,index in zip(point.image_ids,point.point2D_idxs):self.assertEqual(images[iid].point3D_ids[index],pid)
                rows=list(csv.DictReader(io.StringIO((scene/'metadata/frame_timestamps.csv').read_text())))
                self.assertEqual(len(rows),8)
                self.assertEqual(int(rows[0]['source_metadata_index']),2 if object_id==1 else 6)
                self.assertEqual([int(row['source_timestamp_ns'])-int(row['target_timestamp_ns']) for row in rows[4:]],[5_000_000]*4)
            for scene in output.glob('fridge_1_obj*'):
                for rgb in (scene/'rgb').glob('*/*.png'):
                    # Fake masks belong only to this test fixture, never to conversion output.
                    cv2.imwrite(str(scene/'mask'/rgb.relative_to(scene/'rgb')),np.full((6,8),255,dtype=np.uint8))
                apply_scene(scene)
                for model in ['colmap','colmap_final']:
                    self.assertEqual(len(read_model(scene/model/'sparse/0')[1]),2)
                    self.assertEqual(len(read_model(scene/model/'sparse_unmasked/0')[1]),2)
            for p,value in before.items():self.assertEqual(p.read_bytes(),value)
            self.assertEqual(hoi.main(self.command(source,output)),1)  # Never overwrite scenes.

    def test_dry_run_is_read_only_and_missing_metadata_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'none'
            self.assertEqual(hoi.main(self.command(source,output)+['--dry-run']),0)
            self.assertFalse(output.exists())
            next(source.glob('fridge_1/gripper/interaction*_confirmed.json')).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'fetch-missing-metadata'):hoi.main(self.command(source,output))
            self.assertFalse(output.exists())

    def test_moving_or_unaligned_cameras_fail_without_scene_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'none'
            sync=next(source.glob('fridge_1/gripper/iphone_1/*/time_sync_info.json'))
            value=json.loads(sync.read_text());value['timestamps_aligned']=False;sync.write_text(json.dumps(value))
            self.assertEqual(hoi.main(self.command(source,output)),1)
            self.assertFalse((output/'fridge_1_obj001').exists())
            value['timestamps_aligned']=True;sync.write_text(json.dumps(value))
            poses=next(source.glob('fridge_1/gripper/iphone_1/*/poses_aligned/data.csv'))
            lines=poses.read_text().splitlines();parts=lines[-1].split(',');parts[1]='.1';lines[-1]=','.join(parts);poses.write_text('\n'.join(lines)+'\n')
            self.assertEqual(hoi.main(self.command(source,output)),1)
            self.assertFalse((output/'fridge_1_obj001').exists())

    def test_rectification_uses_selected_frame_intrinsics(self):
        image=np.zeros((6,8,3),dtype=np.uint8)
        image[:]=np.arange(8,dtype=np.uint8)[None,:,None]*20
        data=io.BytesIO();np.save(data,np.full((3,4),2,dtype=np.float32),allow_pickle=False)
        k=np.array([[8.,0,4],[0,8,3],[0,0,1]])
        sensor=dict(info=dict(w=8,h=6,dw=4,dh=3),k=k,target_k=k,coeffs=np.array([[8,8,4,3],[8,8,3,3]]))
        rgb,z=hoi.rectify(cv2.imencode('.png',image)[1].tobytes(),data.getvalue(),sensor,1,[8,6])
        self.assertEqual(int(rgb[3,4,0]),60);self.assertEqual(z[3,4],2)

    def test_metadata_fetch_stays_under_output_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'raw';source.mkdir();output=root/'out'
            relative=Path('location/gripper/interaction_confirmed.json')
            index=[dict(path=relative.as_posix(),id=123,size=20)]
            def fake_open(url,timeout):
                return io.BytesIO(json.dumps(index if url==hoi.MANIFEST_URL else dict(example=True)).encode())
            with patch.object(hoi.urllib.request,'urlopen',side_effect=fake_open):
                dry=hoi.Metadata(source,output,fetch=True,dry_run=True)
                self.assertEqual(dry.read(relative),dict(example=True));self.assertFalse(output.exists())
                normal=hoi.Metadata(source,output,fetch=True)
                self.assertEqual(normal.read(relative),dict(example=True))
                self.assertTrue((output/'.hoi_metadata'/relative).is_file())
                self.assertFalse((source/relative).exists())
                with self.assertRaises(ValueError):normal.read('../outside')

    def test_extracted_streams_and_selected_object(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'out'
            for path in source.rglob('camera_*.zip'):
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        target=path.parent/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
                path.unlink()
            self.assertEqual(hoi.main(self.command(source,output)+['--object-ids','2']),0)
            self.assertFalse((output/'fridge_1_obj001').exists());self.assertTrue((output/'fridge_1_obj002/colmap_final/sparse_unmasked/0/images.bin').is_file())

    def test_gap_tolerance_does_not_duplicate_or_assume_ordinal_synchronization(self):
        sensors=[dict(source_times=np.array([100,200,300])*1_000_000,pose_times=np.array([100,200,300])*1_000_000)]*2
        class FakeStream:
            def __init__(self,times):self.members=dict.fromkeys(times)
        a=FakeStream([100_000_000,200_000_000,300_000_000]);b=FakeStream([100_000_000,300_000_000])
        rows=hoi.plan_frames(sensors,[(a,a),(b,b)],dict(start_ns=100_000_000,end_ns=300_000_000),10,20)
        self.assertEqual(len(rows),2);self.assertEqual([r['source_ns'] for r in rows],[[100_000_000]*2,[300_000_000]*2])


if __name__=='__main__':unittest.main()

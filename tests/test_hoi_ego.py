import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from test_convert_hoi_dataset import fixture, hoi


def ego_fixture(source, module='aria_human', missing=()):
    root = source/'fridge_1/gripper'/module/'fridge_1_1-2_gripper_vrs'
    root.mkdir(parents=True)
    (root/'time_sync_info.json').write_text(json.dumps(dict(timestamps_aligned=True,delta=0)))
    raw_pose = np.eye(4)
    raw_pose[:3,3] = [.1,.2,.3]
    rect_pose = np.eye(4)
    rect_pose[:3,:3] = Rotation.from_euler('z',-90,degrees=True).as_matrix()
    calib = dict(PINHOLE=dict(K=[[8,0,4],[0,8,3],[0,0,1]],h=6,w=8,
                             model='PINHOLE',distortion=[0]*5,
                             T_device_camera=raw_pose.tolist(),pinhole_T_device_camera=rect_pose.tolist()))
    with zipfile.ZipFile(root/'calib.zip','w') as archive:
        archive.writestr('calib/calib.json',json.dumps(calib))
    trajectory=io.StringIO()
    writer=csv.writer(trajectory)
    writer.writerow(['timestamp','tx_world_device','ty_world_device','tz_world_device',
                     'qx_world_device','qy_world_device','qz_world_device','qw_world_device','quality_score'])
    for tick in range(15,96):
        seconds=tick/100
        quat=Rotation.from_euler('z',seconds*20,degrees=True).as_quat()
        writer.writerow([1_000_000_000+tick*10_000_000,seconds,seconds*2,0,*quat,1])
    with zipfile.ZipFile(root/'slam.zip','w') as archive:
        archive.writestr('slam/closed_loop_trajectory_aligned/data.csv',trajectory.getvalue())
    with zipfile.ZipFile(root/'camera_rgb.zip','w') as rgb,zipfile.ZipFile(root/'camera_depth.zip','w') as depth:
        for tick in range(2,10):
            if tick in missing:continue
            stamp=1_000_000_000+tick*100_000_000+5_000_000
            image=np.full((6,8,3),tick*20,np.uint8)
            rgb.writestr(f'camera_rgb/{stamp}.jpg',cv2.imencode('.jpg',image)[1].tobytes())
            data=io.BytesIO();np.save(data,np.full((3,4),tick+20,np.float32))
            depth.writestr(f'camera_depth/{stamp}.npy',data.getvalue())
    return root,raw_pose@rect_pose


class EgoExportTests(unittest.TestCase):
    def command(self,source,output):
        return ['--input',str(source),'--output-root',str(output),'--fps','10',
                '--size','8','6','--point-stride','1','--object-ids','1']

    def test_two_moving_views_pose_interpolation_shared_frame_ids_and_exact_leica_part(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=fixture(base);output=base/'out'
            _,extrinsic=ego_fixture(source,missing=(3,))
            ego_fixture(source,'aria_gripper')
            before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            self.assertEqual(hoi.main(self.command(source,output)),0)
            scene=output/'fridge_1_obj001'
            streams=json.loads((scene/'metadata/camera_streams.json').read_text())['cameras']
            self.assertEqual([s['camera_motion'] for s in streams],['static','static','moving','moving'])
            for camera,indices in [('cam_002',[0,2,3]),('cam_003',[0,1,2,3])]:
                meta=json.loads((scene/f'metadata/poses/{camera}.json').read_text())
                self.assertFalse(meta['depth_is_metric'])
                self.assertFalse(meta['depth_projection_to_world_valid'])
                self.assertEqual([r['frame_index'] for r in meta['frames']],indices)
                self.assertEqual(meta['missing_frame_indices'],[1] if camera=='cam_002' else [])
                self.assertEqual(sorted(p.name for p in (scene/f'rgb/{camera}').glob('*.png')),
                                 [f'frame_{index:06d}.png' for index in indices])
                for frame in meta['frames']:
                    seconds=(frame['source_timestamp_ns']-1_000_000_000)/1e9
                    device=np.eye(4)
                    device[:3,:3]=Rotation.from_euler('z',seconds*20,degrees=True).as_matrix()
                    device[:3,3]=[seconds,seconds*2,0]
                    expected=device@extrinsic
                    np.testing.assert_allclose(frame['pose_matrix_4x4'],expected,atol=1e-10)
                    np.testing.assert_allclose(np.asarray(frame['world_to_camera_matrix_4x4'])@expected,np.eye(4),atol=1e-10)
                    # A metric optical point round-trips through the saved world frame and intrinsics.
                    point=expected@np.array([0,0,2,1.])
                    local=np.asarray(frame['world_to_camera_matrix_4x4'])@point
                    uv=np.asarray(meta['intrinsics_matrix_3x3'])@local[:3]
                    np.testing.assert_allclose(uv[:2]/uv[2],[4,3])
                    depth=np.load(scene/frame['depth_path'],allow_pickle=False)
                    self.assertEqual(depth.dtype,np.float32)
                    self.assertEqual(float(depth[2,3]),frame['frame_index']+22)
                self.assertTrue((scene/f'depth_npy/{camera}').exists())
            for cam in ['cam_000','cam_001']:
                meta=json.loads((scene/f'metadata/poses/{cam}.json').read_text())
                self.assertEqual(len(meta['frames']),4);self.assertTrue(meta['depth_is_metric'])
            leica=source/'fridge_1/leica/001/instance_annotations_3d/points/001.ply'
            self.assertEqual((scene/'pointcloud/leica_moving_part.ply').read_bytes(),leica.read_bytes())
            meta=json.loads((scene/'metadata/leica_moving_part.json').read_text())
            self.assertEqual(meta['object_index'],1);self.assertEqual(meta['coordinate_frame'],'hoi_leica_world')
            for p,data in before.items():self.assertEqual(p.read_bytes(),data)
            # Ego export must not add time frames or moving-camera streams to static endpoint COLMAP.
            from mask_colmap_models import read_model
            for model in ['colmap','colmap_final']:
                self.assertEqual(len(read_model(scene/model/'sparse_unmasked/0')[1]),4)

    def test_dry_run_and_missing_leica_or_unaligned_ego_leave_no_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'out'
            ego,_=ego_fixture(source)
            self.assertEqual(hoi.main(self.command(source,output)+['--dry-run']),0)
            self.assertFalse(output.exists())
            (ego/'time_sync_info.json').write_text(json.dumps(dict(timestamps_aligned=False)))
            self.assertEqual(hoi.main(self.command(source,output)),1)
            self.assertFalse((output/'fridge_1_obj001').exists())

            (source/'fridge_1/leica/001/instance_annotations_3d/points/001.ply').unlink()
            self.assertEqual(hoi.main(self.command(source,output)),1)
            self.assertFalse((output/'fridge_1_obj001').exists())

    def test_upgrade_preserves_existing_static_files_and_uses_saved_timeline_and_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'out'
            self.assertEqual(hoi.main(self.command(source,output)),0)
            scene=output/'fridge_1_obj001'
            # These stand in for existing downstream masks and training outputs.
            (scene/'mask/cam_000/custom_mask.txt').write_text('keep mask')
            (scene/'colmap/training.txt').write_text('keep trained geometry')
            preserved={p.relative_to(scene):p.read_bytes() for folder in ['rgb','depth_npy','depth_png','mask']
                       for p in (scene/folder).rglob('*') if p.is_file()}
            for name in ['metadata/hoi_conversion.json','metadata/frame_timestamps.csv','colmap/training.txt']:
                preserved[Path(name)]=(scene/name).read_bytes()
            ego_fixture(source)
            command=self.command(source,output)+['--upgrade-existing','--fps','5','--size','32','24']
            before={p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()}
            self.assertEqual(hoi.main(command+['--dry-run']),0)
            self.assertEqual({p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()},before)
            self.assertEqual(hoi.main(command),0)
            for name,value in preserved.items():self.assertEqual((scene/name).read_bytes(),value)
            meta=json.loads((scene/'metadata/poses/cam_002.json').read_text())
            self.assertEqual((meta['width'],meta['height'],meta['num_frames']),(8,6,4))
            self.assertTrue((scene/'metadata/hoi_auxiliary_export.json').is_file())
            # Regeneration replaces generated auxiliary folders without touching static data.
            (scene/'mask/cam_002/keep.txt').write_text('keep ego masks')
            self.assertEqual(hoi.main(command),0)
            self.assertEqual((scene/'mask/cam_002/keep.txt').read_text(),'keep ego masks')
            for name,value in preserved.items():self.assertEqual((scene/name).read_bytes(),value)
            self.assertFalse(list(scene.glob('.hoi_aux_*')))
            auxiliary={p.relative_to(scene):p.read_bytes() for folder in ['rgb/cam_002','depth_npy/cam_002','metadata/poses'] for p in (scene/folder).rglob('*') if p.is_file()}
            all_before_failure={p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()}
            original=hoi.os.replace
            def fail_publish(src,dst):
                if '/.hoi_aux_' in str(src) and '/backup/' not in str(src) and Path(dst)==scene/'pointcloud/leica_moving_part.ply':
                    raise OSError('simulated publication failure')
                return original(src,dst)
            with patch.object(hoi.os,'replace',side_effect=fail_publish):
                self.assertEqual(hoi.main(command),1)
            for name,value in {**preserved,**auxiliary}.items():self.assertEqual((scene/name).read_bytes(),value)
            self.assertEqual({p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()},all_before_failure)
            self.assertFalse(list(scene.glob('.hoi_aux_*')))


if __name__=='__main__':unittest.main()

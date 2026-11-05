"""Coverage validity, complete-view alignment, and final pipeline-step checks."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'samuele'), str(ROOT/'src')]
import choose_rgbd_direction as direction


def make_scene(scene):
    (scene/'metadata').mkdir(parents=True)
    cameras=['cam_000','cam_001']
    (scene/'metadata/cameras.json').write_text(json.dumps(dict(cameras=[dict(name=c) for c in cameras])))
    for camera in cameras:
        for kind in ['rgb','mask','depth_npy','depth_png']:(scene/kind/camera).mkdir(parents=True)
        for frame,count in enumerate([3,5,9,12]):
            name=f'frame_{frame:06d}'
            mask=np.zeros((3,4),dtype=np.uint8);mask.flat[:count]=255
            Image.fromarray(mask).save(scene/'mask'/camera/(name+'.png'))
            Image.fromarray(np.full((3,4,3),100,dtype=np.uint8)).save(scene/'rgb'/camera/(name+'.png'))
            np.save(scene/'depth_npy'/camera/(name+'.npy'),np.ones((3,4),dtype=np.float32))
            Image.fromarray(np.zeros((3,4),dtype=np.uint16)).save(scene/'depth_png'/camera/(name+'.png'))
    return scene


class DirectionTests(unittest.TestCase):
    def test_decisions_ties_zero_depth_and_no_nan_json(self):
        self.assertEqual(direction.decide([[8,2],[5,1]])['direction'],'FORWARD')
        self.assertEqual(direction.decide([[2,8],[1,5]])['direction'],'BACKWARD')
        self.assertEqual(direction.decide([[5,5]])['direction'],'TIE')
        self.assertEqual(direction.decide([[0,0]])['direction'],'UNAVAILABLE')
        self.assertIsNone(direction.decide([[0,2]])['forward_initial_coverage_advantage_pct'])
        json.dumps(direction.decide([[0,2]]),allow_nan=False)
        for values,fraction in [([], .1),([[1,-2]],.1),([[1,np.nan]],.1),([[1,2]],0),([[1,2]],.6)]:
            with self.assertRaises(ValueError):direction.decide(values,fraction)
        robust=direction.decide([[0,10,10,5,5,100]],.5)
        self.assertEqual(robust['first_window_total'],10)
        self.assertEqual(robust['last_window_total'],5)

    def test_metric_npy_preferred_counts_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene=make_scene(Path(tmp)/'scene')
            original={p:p.read_bytes() for p in scene.rglob('*') if p.is_file()}
            depth=np.ones((3,4),np.float32);depth.flat[:3]=[np.nan,np.inf,-1]
            np.save(scene/'depth_npy/cam_000/frame_000000.npy',depth)
            original[scene/'depth_npy/cam_000/frame_000000.npy']=(scene/'depth_npy/cam_000/frame_000000.npy').read_bytes()
            result=direction.analyze_scene(scene)
            self.assertEqual(result['recommended_direction'],'BACKWARD')
            self.assertFalse(result['sequence_reversed'])
            self.assertEqual(set(result['depth_source_by_camera'].values()),{'depth_npy'})
            self.assertEqual(result['mask_and_valid_depth']['first_window_total'],3)
            self.assertEqual(result['mask_and_valid_depth']['last_window_total'],24)
            with (scene/'metadata/direction/per_frame.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['valid_depth_total'],'3')
            self.assertTrue((scene/'metadata/direction/coverage.png').is_file())
            for p,data in original.items():self.assertEqual(p.read_bytes(),data)

    def test_reject_partial_masks_missing_cameras_and_partial_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene=make_scene(Path(tmp)/'scene')
            file=scene/'mask/cam_001/frame_000003.png';data=file.read_bytes();file.unlink()
            with self.assertRaisesRegex(ValueError,'frame set differs'):direction.analyze_scene(scene,dry_run=True)
            self.assertFalse((scene/'metadata/direction').exists());file.write_bytes(data)
            file=scene/'depth_npy/cam_001/frame_000003.npy';file.unlink()
            with self.assertRaisesRegex(ValueError,'missing or stale depth'):direction.analyze_scene(scene,dry_run=True)
            for p in (scene/'mask/cam_001').iterdir():p.unlink()
            with self.assertRaisesRegex(ValueError,'Mask cameras differ'):direction.analyze_scene(scene,dry_run=True)

    def test_resume_invalidates_changed_inputs_and_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene=make_scene(Path(tmp)/'scene');first=direction.analyze_scene(scene)
            with patch.object(direction.np,'load',side_effect=AssertionError('matching resume must not count pixels')):
                reused=direction.analyze_scene(scene,resume=True)
                self.assertEqual(first,reused)
            path=scene/'mask/cam_000/frame_000000.png'
            Image.fromarray(np.full((3,4),255,dtype=np.uint8)).save(path)
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            changed=direction.analyze_scene(scene,resume=True)
            self.assertNotEqual(first['input_fingerprint'],changed['input_fingerprint'])
            changed=direction.analyze_scene(scene,fraction=.5,resume=True)
            self.assertEqual(changed['mask_and_valid_depth']['window_frames'],2)

    def test_zip_png_fallback_tie_and_unavailable_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);scene=make_scene(tmp/'scene');archive=tmp/'scene.zip'
            for camera in ['cam_000','cam_001']:
                for path in (scene/'depth_npy'/camera).glob('*.npy'):path.unlink()
                for path in (scene/'depth_png'/camera).glob('*.png'):Image.fromarray(np.ones((3,4),np.uint16)).save(path)
            with ZipFile(archive,'w') as zipfile:
                for path in scene.rglob('*'):
                    if path.is_file():zipfile.write(path,path.relative_to(scene).as_posix())
            result=direction.analyze_scene(archive,out=tmp/'report')
            self.assertEqual(result['recommended_direction'],'BACKWARD')
            self.assertEqual(set(result['depth_source_by_camera'].values()),{'depth_png'})
            for camera in ['cam_000','cam_001']:
                for path in (scene/'mask'/camera).glob('*.png'):Image.fromarray(np.ones((3,4),np.uint8)).save(path)
            result=direction.analyze_scene(scene)
            self.assertEqual(result['mask_and_valid_depth']['direction'],'TIE')
            self.assertEqual(result['recommended_direction'],'FORWARD')
            for camera in ['cam_000','cam_001']:
                for path in (scene/'depth_png'/camera).glob('*.png'):Image.fromarray(np.zeros((3,4),np.uint16)).save(path)
            result=direction.analyze_scene(scene)
            self.assertIsNone(result['recommended_direction'])
            with self.assertRaisesRegex(ValueError,'No valid foreground'):direction.finalize_scene(scene)

    def test_discover_all_profiles_and_only_real_final_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=['preparation_8cams/linear/object','preparation_100cams/object','real/final_black_box_episode_000003','real/black_box_episode_000003','real/not_final_episode_000000','hoi/object']
            for name in names:
                folder=root/name/'metadata';folder.mkdir(parents=True)
                (folder/('job_config_resolved.yaml' if name.startswith('preparation') else 'real_conversion.json')).write_text('{}')
            selected=direction.discover_scenes(root,['synthetic','real-final'])
            self.assertEqual({p.relative_to(root).as_posix() for p in selected},set(names[:3]))
            self.assertEqual(len(direction.discover_scenes(root,['real-final'])),1)

    def test_batch_continues_after_invalid_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['final_a_episode_000000','final_b_episode_000000']:
                scene=make_scene(root/name);(scene/'metadata/real_conversion.json').write_text('{}')
            (root/'final_a_episode_000000/mask/cam_000/frame_000003.png').unlink()
            self.assertEqual(direction.main(['--root',str(root),'--dry-run']),1)
            self.assertFalse((root/'final_b_episode_000000/metadata/direction').exists())

    def test_real_bridge_decides_after_colmap_has_completed(self):
        spec=importlib.util.spec_from_file_location('direction_bridge',ROOT/'samuele/colmap_masking.py')
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        with tempfile.TemporaryDirectory() as tmp:
            scene=Path(tmp);(scene/'metadata').mkdir();(scene/'metadata/real_conversion.json').write_text('{}')
            events=[]
            fake=SimpleNamespace(apply_scene=lambda *args:events.append('colmap') or True)
            with patch.dict(sys.modules,{'dense_real_colmap':fake}),patch.object(direction,'finalize_scene',side_effect=lambda scene:events.append('direction')):
                self.assertTrue(bridge.apply_scene_masks(scene))
            self.assertEqual(events,['colmap','direction'])
            with patch.dict(sys.modules,{'dense_real_colmap':SimpleNamespace(apply_scene=lambda *args:(_ for _ in ()).throw(ValueError('failed export')))}),patch.object(direction,'finalize_scene') as finalize:
                with self.assertRaises(ValueError):bridge.apply_scene_masks(scene)
                finalize.assert_not_called()

    def test_render_without_colmap_finishes_with_direction_command(self):
        import subprocess
        spec=importlib.util.spec_from_file_location('direction_runner',ROOT/'scripts/run.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);urdf=root/'assets';urdf.mkdir()
            (urdf/'object.urdf').write_text('<robot name="object"><link name="base"/></robot>')
            commands=[]
            def run(command,dry_run=False):
                commands.append(command)
                if 'render' in command:Path(command[command.index('--output-dir')+1]).mkdir(parents=True)
                return subprocess.CompletedProcess(command,0,stdout='')
            def joints(command,dry_run=False):
                return subprocess.CompletedProcess(command,0,stdout=json.dumps([{'name':'joint','dof':1,'limits':[[0,1]]}]))
            argv=['run.py','--urdf-dir',str(urdf),'--output-root',str(root/'output'),'--num-cams','8','--num-frames','4','--no-export-colmap','--skip-renderer-preflight']
            with patch.object(sys,'argv',argv),patch.object(runner,'run_command',side_effect=run),patch.object(runner,'run_list_joints_command',side_effect=joints):
                self.assertEqual(runner.main(),0)
            self.assertEqual(Path(commands[-1][1]).name,'choose_rgbd_direction.py')
            self.assertIn('--resume',commands[-1])
            self.assertTrue(any('render' in cmd for cmd in commands[:-1]))

    def test_synthetic_export_decides_after_both_endpoints(self):
        spec=importlib.util.spec_from_file_location('direction_export',ROOT/'scripts/export_colmap_dataset.py')
        exporter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=exporter;spec.loader.exec_module(exporter)
        args=argparse.Namespace(sample_stride=1,frame_step=1,max_points_per_image=0,max_frames=1,urdf=None,urdf_dir=None,no_final_frame=False,output_name='colmap',ply_name='points.ply',output_root=None)
        events=[];scene=Path('/tmp/direction-export-fixture')
        stats=SimpleNamespace(num_images=2,num_points3D=3,colmap_dir='colmap',pointcloud_ply='points.ply')
        def export(**kwargs):events.append('final' if hasattr(kwargs['args'],'start_frame') else 'initial');return stats
        with patch.object(exporter,'parse_args',return_value=args),patch.object(exporter,'_discover_model_outputs',return_value=[scene]),patch.object(exporter,'_build_urdf_map',return_value={}),patch.object(exporter,'_resolve_urdf_for_model',return_value=None),patch.object(exporter,'export_model',side_effect=export),patch.object(exporter,'asdict',return_value={}),patch.object(exporter,'_load_render_config',return_value=SimpleNamespace(resolved_num_frames=lambda:10)),patch.object(direction,'finalize_scene',side_effect=lambda scene:events.append('direction')):
            self.assertEqual(exporter.main(),0)
        self.assertEqual(events,['initial','final','direction'])


if __name__=='__main__':unittest.main()

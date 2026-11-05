import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
from robot_mask_guidance import project, load_cached
import correct_real_masks as correction


class RealMaskCorrectionTests(unittest.TestCase):
    def test_recorded_pose_projection_uses_world_to_camera_and_tool_length(self):
        poses=np.array([[.2,.3,1,0,0,0],[.2,.3,-1,0,0,0]])
        pose=np.eye(4);pose[0,3]=.1
        camera=dict(name='cam_000',width=200,height=200,pose_matrix_4x4=pose.tolist())
        k=np.array([[100,0,50],[0,100,50],[0,0,1]])
        data=project(poses,[camera],k,{},.03)
        points=data['cameras']['cam_000']
        np.testing.assert_allclose(points[0],[[60,80],[50+10/1.03,50+30/1.03]])
        self.assertEqual(points[1],[])
        self.assertEqual(data['tool_axis_offsets_m'],[0,.03])
        with self.assertRaises(ValueError):project(poses,[camera],k,{},1.)

    def test_correction_plan_selects_only_requested_converted_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profiles=root/'profiles';profiles.mkdir()
            scene=root/'final_matchbox_episode_000000';(scene/'metadata/raw').mkdir(parents=True)
            (scene/'metadata/real_conversion.json').write_text('{"episode":0}')
            (scene/'metadata/raw/episode_000000.parquet').write_bytes(b'fixture')
            (scene/'rgb/cam_000').mkdir(parents=True)
            Image.new('RGB',(16,16)).save(scene/'rgb/cam_000/frame_000000.png')
            spec=dict(coordinate_system='image_pixels_xy',cameras={'cam_000':dict(image_size=[16,16],box=[0,0,16,16],positive=[[5,5]],negative=[])})
            (profiles/'final_matchbox.json').write_text(json.dumps(spec))
            jobs=correction.plan(root,['matchbox'],profiles)
            self.assertEqual([row[0] for row in jobs],[scene])
            (scene/'conversion_in_progress.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'incomplete'):correction.plan(root,['matchbox'],profiles)

    def test_pose_cache_rejects_changed_calibration_or_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene=Path(tmp);(scene/'metadata/raw').mkdir(parents=True)
            parquet=scene/'metadata/raw/episode_000003.parquet';parquet.write_bytes(b'poses')
            cameras=scene/'metadata/cameras.json';cameras.write_text('{"cameras":[]}')
            (scene/'metadata/real_conversion.json').write_text('{"episode":3,"source_frames":[0,2]}')
            provenance=dict(parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),camera_metadata_sha256=hashlib.sha256(cameras.read_bytes()).hexdigest(),source_frames=[0,2])
            payload=dict(provenance=provenance,tool_axis_offsets_m=[0,.03])
            (scene/'metadata/robot_mask_guidance.json').write_text(json.dumps(payload))
            self.assertEqual(load_cached(scene,.03),payload)
            self.assertIsNone(load_cached(scene,.15))
            cameras.write_text('{"cameras":[1]}')
            self.assertIsNone(load_cached(scene,.03))

    def test_black_box_group_keeps_new_and_old_recording_profiles_separate(self):
        profiles=Path(__file__).resolve().parents[1]/'configs/masks'
        prefixes=['final_black_box','black_box_20261021_161154']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for prefix in prefixes+['final_light_blue_box']:
                scene=root/(prefix+'_episode_000000');(scene/'metadata/raw').mkdir(parents=True)
                (scene/'metadata/real_conversion.json').write_text('{"episode":0}')
                (scene/'metadata/raw/episode_000000.parquet').write_bytes(b'fixture')
                for cam in ['cam_000','cam_001','cam_002']:
                    rgb=scene/'rgb'/cam;rgb.mkdir(parents=True)
                    Image.new('RGB',(640,480)).save(rgb/'frame_000000.png')
            jobs=correction.plan(root,['black_box'],profiles)
            self.assertEqual({profile.stem for _,profile,_ in jobs},set(prefixes))
            self.assertEqual(len(jobs),2)

    def test_suitcase_group_matches_native_spelling_and_checks_three_camera_profiles(self):
        profiles=Path(__file__).resolve().parents[1]/'configs/masks'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for prefix in ['final_orange_suitecase','final_black_box']:
                scene=root/(prefix+'_episode_000000');(scene/'metadata/raw').mkdir(parents=True)
                (scene/'metadata/real_conversion.json').write_text('{"episode":0}')
                (scene/'metadata/raw/episode_000000.parquet').write_bytes(b'fixture')
                for cam in ['cam_000','cam_001','cam_002']:
                    rgb=scene/'rgb'/cam;rgb.mkdir(parents=True)
                    Image.new('RGB',(640,480)).save(rgb/'frame_000000.png')
            jobs=correction.plan(root,['orange_suitcase'],profiles)
            self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0][0].name,'final_orange_suitecase_episode_000000')
            self.assertEqual(jobs[0][2],'orange suitcase')

    def test_shared_sam_backend_and_scene_failure_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[]
            for i in range(2):
                scene=root/f'scene_{i}';(scene/'metadata').mkdir(parents=True)
                profile=root/f'profile_{i}.json';profile.write_text('{}')
                jobs.append((scene,profile,'target'))
            def generate(args,segment):
                segment(args.input/'rgb/cam_000/frame_000000.png','target')
                if args.input==jobs[0][0]:raise ValueError('bad view')
                return dict(empty=0,completed=3)
            backend=lambda *args:(np.ones((2,2),dtype=np.uint8)*255,{})
            with patch.object(correction,'plan',return_value=jobs),patch.object(correction,'prepare',return_value={}),patch.object(correction.masker,'generate',side_effect=generate),patch.object(correction.masker,'load_segmenter',return_value=backend) as load:
                result=correction.main(['--root',str(root),'--overwrite'])
            self.assertEqual(result,1);load.assert_called_once()
            report=json.loads((root/'mask_correction_report.json').read_text())
            self.assertEqual([row['status'] for row in report],['failed','complete'])


if __name__=='__main__':unittest.main()

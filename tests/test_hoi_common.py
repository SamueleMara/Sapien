import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from test_convert_hoi_dataset import fixture, hoi
from test_hoi_ego import ego_fixture
from standardize_hoi_cameras import standardize_scene, move_camera
from mask_colmap_models import read_model, apply_scene
from read_write_model import qvec2rotmat


class CommonCameraTests(unittest.TestCase):
    def test_all_views_depth_metadata_endpoint_poses_and_mask_filtering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'out'
            ego_fixture(source);ego_fixture(source,'aria_gripper')
            argv=['--input',str(source),'--output-root',str(output),'--fps','10','--size','8','6','--point-stride','1','--object-ids','1']
            self.assertEqual(hoi.main(argv),0)
            scene=output/'fridge_1_obj001'
            metadata=json.loads((scene/'metadata/cameras.json').read_text())
            self.assertEqual([c['name'] for c in metadata['cameras']],['cam_000','cam_001','cam_002','cam_003'])
            for camera in metadata['cameras']:
                name=camera['name']
                self.assertEqual(len(list((scene/'rgb'/name).glob('*.png'))),4)
                self.assertEqual(len(list((scene/'depth_npy'/name).glob('*.npy'))),4)
                self.assertEqual(len(list((scene/'depth_png'/name).glob('*.png'))),4)
                if camera.get('camera_motion')=='moving':
                    self.assertNotIn('pose_matrix_4x4',camera)
                    self.assertFalse(camera['depth_is_metric'])
            for model,frame in [('colmap',0),('colmap_final',3)]:
                _,images,points=read_model(scene/model/'sparse_unmasked/0')
                self.assertEqual(len(images),4)
                self.assertEqual({i.name for i in images.values()},{f'cam_{cam:03d}/frame_{frame:06d}.png' for cam in range(4)})
                for image in images.values():
                    camera=image.name.split('/')[0]
                    if camera in ['cam_002','cam_003']:
                        poses=json.loads((scene/f'metadata/poses/{camera}.json').read_text())
                        row=poses['frames'][frame]
                        transform=np.eye(4);transform[:3,:3]=qvec2rotmat(image.qvec);transform[:3,3]=image.tvec
                        np.testing.assert_allclose(transform,row['world_to_camera_matrix_4x4'],atol=1e-12)
                        self.assertEqual(len(image.point3D_ids),0)
                self.assertTrue(points)
            before={p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()}
            standardize_scene(scene)
            self.assertEqual({p.relative_to(scene):p.read_bytes() for p in scene.rglob('*') if p.is_file()},before)
            for path in (scene/'rgb').glob('*/*.png'):
                # All-white masks are solely a synthetic filtering fixture.
                target=scene/'mask'/path.relative_to(scene/'rgb')
                cv2.imwrite(str(target),np.full((6,8),255,np.uint8))
            apply_scene(scene)
            self.assertEqual(len(read_model(scene/'colmap/sparse/0')[1]),4)
            self.assertEqual(len(read_model(scene/'colmap_final/sparse/0')[1]),4)

    def test_camera_collision_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'legacy';b=root/'common';a.mkdir();b.mkdir()
            (a/'frame.png').write_bytes(b'old');(b/'frame.png').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'Conflicting'):move_camera(a,b)
            self.assertEqual((a/'frame.png').read_bytes(),b'old')
            self.assertEqual((b/'frame.png').read_bytes(),b'different')


if __name__=='__main__':unittest.main()

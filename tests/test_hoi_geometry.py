import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_convert_hoi_dataset import fixture, hoi
from test_hoi_ego import ego_fixture
from refine_hoi_geometry import trim_scene, prepare_scene, register_pose
from mask_colmap_models import read_model


class GeometryTests(unittest.TestCase):
    def test_trim_pose_refinement_fusion_and_endpoint_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);output=root/'out'
            ego_fixture(source);ego_fixture(source,'aria_gripper')
            argv=['--input',str(source),'--output-root',str(output),'--fps','10','--size','8','6','--point-stride','1','--object-ids','1']
            self.assertEqual(hoi.main(argv),0)
            scene=output/'fridge_1_obj001'
            original=(scene/'rgb/cam_000/frame_000001.png').read_bytes()
            target=np.array([[x,y,2.3] for x in np.linspace(-.8,.8,30) for y in np.linspace(-.6,.6,30)])
            np.savetxt(scene/'pointcloud/leica_moving_part.ply',target,header=f'ply\nformat ascii 1.0\nelement vertex {len(target)}\nproperty double x\nproperty double y\nproperty double z\nend_header',comments='')
            prepare_scene(scene,start=1)
            self.assertEqual((scene/'rgb/cam_000/frame_000000.png').read_bytes(),original)
            self.assertEqual(len(list((scene/'rgb/cam_000').glob('*.png'))),3)
            poses=json.loads((scene/'metadata/poses/cam_002.json').read_text())
            self.assertEqual(poses['frames'][0]['original_frame_index'],1)
            self.assertEqual(poses['frames'][0]['frame_index'],0)
            for model,frame in [('colmap',0),('colmap_final',2)]:
                _,images,_=read_model(scene/model/'sparse_unmasked/0')
                self.assertEqual(len(images),4)
                self.assertEqual({i.name for i in images.values()},{f'cam_{i:03d}/frame_{frame:06d}.png' for i in range(4)})
            report=json.loads((scene/'metadata/hoi_leica_alignment.json').read_text())
            self.assertGreater(report['fusion']['leica_points_with_depth_supported_observations'],0)
            self.assertFalse(report['fusion']['final_state_leica_fused'])
            before=(scene/'metadata/cameras.json').read_bytes()
            prepare_scene(scene,start=1)
            self.assertEqual((scene/'metadata/cameras.json').read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'another start'):trim_scene(scene,2)

    def test_registration_improves_a_known_small_offset_and_rejects_no_overlap(self):
        target=np.array([[x,y,2.] for x in np.linspace(-.5,.5,40) for y in np.linspace(-.5,.5,40)])
        source=target+[0,0,.025]
        correction,report=register_pose(source,target)
        self.assertTrue(report['applied'])
        self.assertLess(report['median_distance_after_m'],report['median_distance_before_m']*.2)
        self.assertAlmostEqual(correction[2,3],-.025,delta=.005)
        correction,report=register_pose(source+10,target)
        self.assertFalse(report['applied']);np.testing.assert_allclose(correction,np.eye(4))


if __name__=='__main__':unittest.main()

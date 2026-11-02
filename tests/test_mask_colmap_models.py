"""CPU regression tests: preserve unmasked models and consistent foreground tracks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mask_colmap_models as masking
from read_write_model import Camera, Image, Point3D, write_model


def make_scene(root):
    cameras = {1: Camera(1, 'PINHOLE', 4, 3, np.array([2., 2., 2., 1.]))}
    images = {}
    for iid, name in [(1, 'cam_000'), (2, 'cam_001')]:
        images[iid] = Image(iid, np.array([1., 0., 0., 0.]), np.zeros(3), 1, name+'/frame_000000.png',
                            np.array([[0., 0.], [1., 1.], [3., 2.]]), np.array([10, 20, 30]))
        path = root / 'mask' / name / 'frame_000000.png'
        path.parent.mkdir(parents=True)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 1] = 255  # point 20 has two foreground tracks
        if iid == 1:
            mask[0, 0] = 255  # point 10 has one foreground and one background track
        PILImage.fromarray(mask).save(path)
    points = {pid: Point3D(pid, np.array([0., 0., float(pid)]), np.array([10,20,30], dtype=np.uint8), .1,
                           np.array([1,2]), np.array([index,index])) for index, pid in enumerate([10,20,30])}
    for name in ['colmap', 'colmap_final']:
        folder = root / name / 'sparse/0'
        folder.mkdir(parents=True)
        for image in images.values():
            path=root/name/'images'/image.name
            path.parent.mkdir(parents=True,exist_ok=True)
            PILImage.fromarray(np.full((3,4,3),100,dtype=np.uint8)).save(path)
        for ext in ['.txt', '.bin']:
            write_model(cameras, images, points, str(folder), ext)
    return cameras, images, points


class MaskColmapTests(unittest.TestCase):
    def test_filter_and_rerun_from_unmasked_restores_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_scene(root)
            before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            masking.apply_scene(root, dry_run=True)
            self.assertEqual(before, {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            masking.apply_scene(root)
            for name in ['colmap','colmap_final']:
                for p in (root/name/'sparse_unmasked/0').iterdir():
                    self.assertEqual(p.read_bytes(), before[str(Path(name)/'sparse/0'/p.name)])
                _, images, points = masking.read_model(root/name/'sparse/0')
                self.assertEqual(set(points), {10,20})
                self.assertEqual(points[10].image_ids.tolist(), [1])
                self.assertEqual(images[2].point3D_ids.tolist(), [-1,20,-1])
                self.assertEqual(images[1].point3D_ids.tolist(), [10,20,-1])
                for pid, point in points.items():
                    for iid, index in zip(point.image_ids, point.point2D_idxs):
                        self.assertEqual(images[iid].point3D_ids[index], pid)
                # Binary and text round trips both preserve the masked selection.
                for binary in (root/name/'sparse/0').glob('*.bin'):
                    binary.unlink()
                self.assertEqual(set(masking.read_model(root/name/'sparse/0')[2]), {10,20})
            # Edited masks must restore points from the original, not filter the filtered result.
            for path in (root/'mask').rglob('*.png'):
                PILImage.fromarray(np.full((3,4),255,dtype=np.uint8)).save(path)
            masking.apply_scene(root)
            self.assertEqual(set(masking.read_model(root/'colmap/sparse/0')[2]), {10,20,30})

    def test_missing_or_empty_masks_never_replace_existing_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_scene(root)
            before = (root/'colmap/sparse/0/points3D.bin').read_bytes()
            mask = root/'mask/cam_001/frame_000000.png'
            data=mask.read_bytes();mask.unlink()
            with self.assertRaises(FileNotFoundError):
                masking.apply_scene(root)
            self.assertFalse((root/'colmap/sparse_unmasked').exists())
            self.assertEqual((root/'colmap/sparse/0/points3D.bin').read_bytes(), before)
            mask.write_bytes(data)
            for path in (root/'mask').rglob('*.png'):
                PILImage.fromarray(np.zeros((3,4),dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, 'no 3D points'):
                masking.apply_scene(root)
            self.assertEqual((root/'colmap/sparse/0/points3D.bin').read_bytes(), before)

    def test_final_endpoint_failure_and_bad_dimensions_leave_initial_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cameras, images, points = make_scene(root)
            before = (root/'colmap/sparse/0/points3D.bin').read_bytes()
            final_images = {iid: image._replace(name=image.name.replace('000000','000001')) for iid,image in images.items()}
            write_model(cameras, final_images, points, str(root/'colmap_final/sparse/0'), '.bin')
            with self.assertRaises(FileNotFoundError):
                masking.apply_scene(root)
            self.assertFalse((root/'colmap/sparse_unmasked').exists())
            self.assertEqual((root/'colmap/sparse/0/points3D.bin').read_bytes(),before)
            bad = root/'mask/cam_000/frame_000000.png'
            PILImage.fromarray(np.ones((2,2),dtype=np.uint8)*255).save(bad)
            with self.assertRaisesRegex(ValueError,'camera-sized'):
                masking.apply_scene(root)
            self.assertEqual((root/'colmap/sparse/0/points3D.bin').read_bytes(),before)

    def test_missing_final_rebuilt_from_saved_rgbd_without_sapien(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_scene(root)
            shutil.rmtree(root/'colmap_final')
            (root/'metadata').mkdir()
            specs=[]
            for index,name in enumerate(['cam_000','cam_001']):
                specs.append(dict(name=name,width=4,height=3,near=.01,far=10,fx=2,fy=2,cx=2,cy=1,
                                  position=[0,index,0],target=[1,index,0],up=[0,0,1]))
                (root/'depth_npy'/name).mkdir(parents=True)
                (root/'rgb'/name).mkdir(parents=True)
                for frame in [0,1]:
                    np.save(root/f'depth_npy/{name}/frame_{frame:06d}.npy',np.full((3,4),2,dtype=np.float32))
                    PILImage.fromarray(np.full((3,4,3),100,dtype=np.uint8)).save(root/f'rgb/{name}/frame_{frame:06d}.png')
                PILImage.fromarray(np.full((3,4),255,dtype=np.uint8)).save(root/f'mask/{name}/frame_000001.png')
            (root/'metadata/cameras.json').write_text(json.dumps(dict(cameras=specs)))
            (root/'metadata/job_config_resolved.yaml').write_text('num_frames_resolved: 2\n')
            masking.apply_scene(root,build_missing_endpoints=True)
            cameras,images,points=masking.read_model(root/'colmap_final/sparse/0')
            self.assertEqual(len(points),2)
            self.assertEqual({i.name for i in images.values()},{'cam_000/frame_000001.png','cam_001/frame_000001.png'})
            for pid,point in points.items():
                np.testing.assert_allclose(point.xyz,[2,2+(pid-1),1],atol=1e-12)
            for image in images.values():
                self.assertEqual((root/'colmap_final/images'/image.name).read_bytes(),(root/'rgb'/image.name).read_bytes())
            self.assertTrue((root/'colmap_final/sparse_unmasked/0/points3D.bin').is_file())

    def test_legacy_all_time_models_become_one_image_per_view_in_both_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cameras, base_images, base_points=make_scene(root)
            (root/'metadata').mkdir()
            (root/'metadata/real_conversion.json').write_text(json.dumps(dict(source_frames=[0,2,4],output_frames=[0,1,2])))
            for model in ['colmap','colmap_final']:
                images={}
                for frame in range(3):
                    for iid,image in base_images.items():
                        new_id=frame*2+iid
                        name=image.name.replace('000000',f'{frame:06d}')
                        images[new_id]=image._replace(id=new_id,name=name,point3D_ids=image.point3D_ids.copy())
                        target=root/model/'images'/name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        PILImage.fromarray(np.full((3,4,3),frame,dtype=np.uint8)).save(target)
                        mask=root/'mask'/name
                        PILImage.fromarray(np.full((3,4),255,dtype=np.uint8)).save(mask)
                # All-time tracks intentionally touch all six images.
                points={pid:point._replace(image_ids=np.arange(1,7,dtype=np.int32),
                                           point2D_idxs=np.full(6,index,dtype=np.int32))
                        for index,(pid,point) in enumerate(base_points.items())}
                # One point exists only at the middle time and must be removed.
                images[3]=images[3]._replace(xys=np.vstack([images[3].xys,[2.,1.]]),point3D_ids=np.append(images[3].point3D_ids,999))
                points[999]=Point3D(999,np.ones(3),np.ones(3,dtype=np.uint8),0.,np.array([3]),np.array([3]))
                for ext in ['.txt','.bin']:
                    write_model(cameras,images,points,str(root/model/'sparse/0'),ext)
            masking.apply_scene(root)
            for model,frame,ids in [('colmap',0,{1,2}),('colmap_final',2,{5,6})]:
                expected={f'cam_{i:03d}/frame_{frame:06d}.png' for i in range(2)}
                for variant in ['sparse','sparse_unmasked']:
                    _,images,points=masking.read_model(root/model/variant/'0')
                    self.assertEqual(len(images),2)
                    self.assertEqual(set(images),ids)
                    self.assertEqual({image.name for image in images.values()},expected)
                    self.assertNotIn(999,points)
                    for pid,point in points.items():
                        for iid,index in zip(point.image_ids,point.point2D_idxs):
                            self.assertIn(iid,ids)
                            self.assertEqual(images[iid].point3D_ids[index],pid)
                files={str(p.relative_to(root/model/'images')) for p in (root/model/'images').rglob('*.png')}
                self.assertEqual(files,expected)
                self.assertEqual(json.loads((root/model/'export_summary.json').read_text())['num_images'],2)
            self.assertTrue((root/'metadata/endpoint_model_backup/colmap/sparse/0/images.bin').exists())
            masking.apply_scene(root)  # Saved unmasked models must not restore the discarded times.
            self.assertEqual(len(masking.read_model(root/'colmap/sparse/0')[1]),2)

    def test_fresh_writer_and_geometry_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cameras, images, points = make_scene(root)
            stats = masking.write_pair(root/'fresh', cameras, images, points, root/'mask')
            self.assertEqual(stats['num_points3D_masked'],2)
            _, _, filtered=masking.read_model(root/'fresh/sparse/0')
            _, _, full=masking.read_model(root/'fresh/sparse_unmasked/0')
            self.assertEqual(set(full),set(points))
            for pid, point in filtered.items():
                np.testing.assert_array_equal(point.xyz, points[pid].xyz)
                np.testing.assert_array_equal(point.rgb, points[pid].rgb)
                self.assertEqual(point.error, points[pid].error)


if __name__ == '__main__':
    unittest.main()

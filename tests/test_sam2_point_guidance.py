import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
from sam2_point_guidance import read_guidance, select_mask, predict_guided_mask


class PointGuidanceTests(unittest.TestCase):
    def test_high_score_wrong_object_rejected_and_box_clips_output(self):
        masks=np.zeros((3,8,10),dtype=bool)
        masks[0,:,:]=True  # Includes arm and dog, despite higher model score.
        masks[1,2:6,2:7]=True
        masks[1,5,6]=False
        masks[1,7,9]=True  # Extra island outside the object box.
        masks[2,0:2,:]=True
        points=np.array([[3,3],[6,5]],dtype=float)
        mask,score=select_mask(masks,np.array([.99,.8,.9]),np.array([2,2,7,6]),points,np.array([1,0]))
        self.assertEqual(score,.8)
        self.assertEqual(np.count_nonzero(mask),19)
        self.assertEqual(mask[7,9],0)
        self.assertEqual(mask[3,3],255)

    def test_no_candidate_obeys_points_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'violate explicit'):
            select_mask(np.ones((3,8,10)),np.ones(3),np.array([2,2,7,6]),np.array([[3,3],[4,4]]),np.array([1,0]))

    def test_background_outside_export_box_is_checked_after_clipping(self):
        mask,_=select_mask(np.ones((1,8,10)),np.ones(1),np.array([2,2,7,6]),np.array([[3,3],[1,1]]),np.array([1,0]))
        self.assertEqual(mask[1,1],0)
        self.assertEqual(mask[3,3],255)

    def test_retry_refines_logits_without_relaxing_points(self):
        bad=np.ones((1,8,10),dtype=bool);good=bad.copy();good[0,4,4]=False
        logits=np.ones((1,2,2),dtype=float)
        predictor=Mock()
        predictor.predict.side_effect=[(bad,np.array([.9]),logits),(bad,np.array([.8]),logits),(good,np.array([.85]),logits)]
        mask,score,retries=predict_guided_mask(predictor,np.array([2,2,7,6]),np.array([[3,3],[4,4]]),np.array([1,0]))
        self.assertEqual(mask[4,4],0);self.assertEqual(mask[3,3],255)
        self.assertEqual(retries,2)
        np.testing.assert_array_equal(predictor.predict.call_args.kwargs['mask_input'],logits)
        predictor.predict.side_effect=None
        predictor.predict.return_value=(bad,np.array([.9]),logits)
        with self.assertRaisesRegex(ValueError,'violate explicit'):
            predict_guided_mask(predictor,np.array([2,2,7,6]),np.array([[3,3],[4,4]]),np.array([1,0]))

    def test_guidance_requires_matching_resolution_and_foreground_inside_box(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'points.json'
            spec=dict(image_size=[10,8],box=[2,2,8,7],positive=[[3,3]],negative=[[1,1]])
            data=dict(coordinate_system='image_pixels_xy',cameras={'cam_000':spec})
            path.write_text(json.dumps(data))
            box,points,labels=read_guidance(path,'cam_000',(10,8))
            np.testing.assert_array_equal(labels,[1,0])
            with self.assertRaisesRegex(ValueError,'resolution mismatch'):read_guidance(path,'cam_000',(20,16))
            spec['positive']=[[0,0]];path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'inside guidance box'):read_guidance(path,'cam_000',(10,8))

    def test_quality_refinement_keeps_valid_mask_with_higher_score(self):
        masks=np.zeros((1,8,10),dtype=bool);masks[0,2:6,2:7]=True
        good=masks.copy();good[0,2,2]=False
        predictor=Mock();logits=np.ones((1,2,2))
        predictor.predict.side_effect=[(masks,np.array([.7]),logits),(good,np.array([.9]),logits)]
        mask,score,retries=predict_guided_mask(predictor,np.array([2,2,7,6]),np.array([[3,3],[1,1]]),np.array([1,0]),1)
        self.assertEqual(score,.9);self.assertEqual(mask[2,2],0);self.assertEqual(retries,1)

    def test_crop_must_contain_box_and_stay_in_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.json';spec=dict(image_size=[16,16],box=[4,4,12,12],positive=[[6,6]],negative=[],crop=[3,3,13,13])
            data=dict(coordinate_system='image_pixels_xy',cameras={'cam_000':spec})
            p.write_text(json.dumps(data));read_guidance(p,'cam_000',(16,16))
            spec['crop']=[5,3,13,13];p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'crop must contain'):read_guidance(p,'cam_000',(16,16))


if __name__=='__main__':unittest.main()

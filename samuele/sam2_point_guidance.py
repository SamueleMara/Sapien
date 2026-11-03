"""Explicit per-view box and point prompts for ambiguous fixed-camera scenes."""
import json
from pathlib import Path

import numpy as np


def read_guidance(path, camera, size):
    data = json.loads(Path(path).read_text())
    if data.get('coordinate_system') != 'image_pixels_xy':
        raise ValueError('Point guidance must use image_pixels_xy coordinates')
    spec = data['cameras'][camera]
    width, height = size
    if spec.get('image_size') != [width, height]:
        raise ValueError(f'Guidance resolution mismatch for {camera}; expected {spec.get("image_size")}, got {size}')
    positive = np.asarray(spec['positive'], dtype=float).reshape(-1,2)
    negative = np.asarray(spec.get('negative', []), dtype=float).reshape(-1,2)
    points = np.concatenate([positive, negative])
    box = np.asarray(spec['box'], dtype=float)
    if not len(positive) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= [width,height]):
        raise ValueError(f'Invalid guidance points for {camera}')
    if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f'Invalid guidance box for {camera}')
    if np.any(positive < box[:2]) or np.any(positive >= box[2:]):
        raise ValueError(f'Positive points must be inside guidance box for {camera}')
    if 'crop' in spec:
        crop=np.asarray(spec['crop'],dtype=float)
        if (crop.shape!=(4,) or not np.isfinite(crop).all() or not np.equal(crop,np.floor(crop)).all()
            or not (0<=crop[0]<crop[2]<=width and 0<=crop[1]<crop[3]<=height)
            or np.any(box[:2]<crop[:2]) or np.any(box[2:]>crop[2:])):
            raise ValueError(f'Guidance crop must contain the object box within the image: {camera}')
    labels = np.r_[np.ones(len(positive),dtype=np.int32),np.zeros(len(negative),dtype=np.int32)]
    return box, points.astype(np.float32), labels


def select_mask(masks, scores, box, points, labels):
    masks = np.asarray(masks) > .5
    height,width = masks.shape[1:]
    x0,y0 = np.floor(box[:2]).astype(int)
    x1,y1 = np.ceil(box[2:]).astype(int)
    clipped = np.zeros_like(masks)
    clipped[:,y0:y1,x0:x1] = masks[:,y0:y1,x0:x1]
    xy = np.floor(points).astype(int)
    matches = clipped[:,xy[:,1],xy[:,0]] == labels.astype(bool)
    obeys = matches.all(axis=1)
    if not obeys.any():
        closest = int(np.argmax(matches.sum(axis=1)))
        violations = [dict(point_xy=points[i].tolist(),expected='foreground' if labels[i] else 'background')
                      for i in np.flatnonzero(~matches[closest])]
        raise ValueError('SAM candidates violate explicit foreground/background points; '
                         f'closest candidate violates {violations}; inspect guidance instead of publishing an ambiguous mask')
    index = int(np.argmax(np.where(obeys, scores, -np.inf)))
    mask = clipped[index].astype(np.uint8)*255
    return mask, float(scores[index])


def predict_guided_mask(predictor, box, points, labels, refinement_passes=0):
    """Refine ambiguous SAM predictions while keeping every point constraint."""
    kwargs = dict(box=box,point_coords=points,point_labels=labels)
    masks,scores,logits = predictor.predict(**kwargs,multimask_output=False)
    for attempt in range(4):
        try:
            mask,score = select_mask(masks,scores,box,points,labels)
            if refinement_passes:
                # Find the valid hypothesis selected above, keeping its actual SAM logits.
                index=None
                for i in np.argsort(-np.asarray(scores)):
                    try:
                        select_mask(masks[i:i+1],scores[i:i+1],box,points,labels)
                        index=int(i);break
                    except ValueError: continue
                lowres=logits[index:index+1]
                for _ in range(refinement_passes):
                    refined,quality,lowres=predictor.predict(**kwargs,mask_input=lowres,multimask_output=False)
                    attempt+=1
                    try: candidate,candidate_score=select_mask(refined,quality,box,points,labels)
                    except ValueError: break
                    if candidate_score>score:mask,score=candidate,candidate_score
            return mask,score,attempt
        except ValueError:
            if attempt == 3: raise
        if attempt == 0:
            # Ask for alternate hypotheses only when the single-mask prediction fails.
            masks,scores,logits = predictor.predict(**kwargs,multimask_output=True)
        else:
            xy = np.floor(points).astype(int)
            votes = (np.asarray(masks)[:,xy[:,1],xy[:,0]] > .5) == labels.astype(bool)
            # Refine the closest hypothesis with SAM's low-resolution mask input.
            index = max(range(len(scores)),key=lambda i:(int(votes[i].sum()),float(scores[i])))
            masks,scores,logits = predictor.predict(**kwargs,mask_input=logits[index:index+1],multimask_output=False)


def load_segmenter(args):
    import contextlib
    import sys
    from PIL import Image
    import torch

    root = args.model_root.expanduser().resolve()
    sys.path.insert(0,str(root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = root/'checkpoints/sam2.1_hiera_base_plus.pt'
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the selected Python environment')
    predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(checkpoint),device=args.device))
    robot_cache={}
    print(f'Loading local SAM2 on {args.device}; explicit box/point guidance; text detector bypassed',flush=True)

    def segment(path, prompt):
        with Image.open(path) as source: rgb=np.array(source.convert('RGB'))
        box,points,labels = read_guidance(args.prompt_points,path.parent.name,(rgb.shape[1],rgb.shape[0]))
        profile=json.loads(Path(args.prompt_points).read_text())
        spec=profile['cameras'][path.parent.name]
        if profile.get('robot_state_guidance'):
            from robot_mask_guidance import guidance_path
            file=guidance_path(path.parents[2],getattr(args,'robot_guidance_root',None))
            if file not in robot_cache: robot_cache[file]=json.loads(file.read_text())
            frame=int(path.stem.removeprefix('frame_'))
            extra=np.asarray(robot_cache[file]['cameras'][path.parent.name][frame],dtype=float).reshape(-1,2)
            # Far-away arm prompts cannot affect the clipped export and can distract SAM.
            near=((extra>=box[:2]-8)&(extra<box[2:]+8)).all(axis=1)
            extra=extra[near]
            points=np.concatenate([points,extra]).astype(np.float32)
            labels=np.r_[labels,np.zeros(len(extra),dtype=np.int32)]
        crop=np.asarray(spec.get('crop',[0,0,rgb.shape[1],rgb.shape[0]]),dtype=int)
        offset=crop[:2]
        inside=((points>=crop[:2])&(points<crop[2:])).all(axis=1)
        autocast = torch.autocast('cuda',dtype=torch.bfloat16) if args.device=='cuda' else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            predictor.set_image(rgb[crop[1]:crop[3],crop[0]:crop[2]])
            try:
                passes=int(profile.get('refinement_passes',0))
                if not 0<=passes<=2:raise ValueError('Guidance refinement_passes must be 0..2')
                local,score,retries = predict_guided_mask(predictor,box-np.tile(offset,2),points[inside]-offset,labels[inside],passes)
                mask=np.zeros(rgb.shape[:2],dtype=np.uint8)
                mask[crop[1]:crop[3],crop[0]:crop[2]]=local
                # Negatives outside the crop are background in the full-resolution export.
                mask,_=select_mask(mask[None],np.array([score]),box,points,labels)
            except ValueError as error:
                raise ValueError(f'{path.parent.name}/{path.name}: {error}') from error
        return mask,dict(selection='explicit_camera_box_and_points',sam_score=score,
                         sam_refinement_retries=retries,
                         image_crop_xyxy=crop.tolist(),
                         box_xyxy=box.tolist(),points_xy=points.tolist(),point_labels=labels.tolist(),
                         prompt_note='Text is the output object label; segmentation uses explicit spatial guidance')
    return segment

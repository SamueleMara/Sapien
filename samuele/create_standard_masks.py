#!/usr/bin/env python3
"""GroundingDINO + SAM2 foreground masks for rgb/cam_NNN/frame_NNNNNN.png."""
import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from utils.local_storage import configure_runtime
from point_sampling import point_fraction

DEFAULT_MODEL_ROOT = Path.home() / "Grounded-SAM-2-autodistill"


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def list_frames(dataset, limit=0):
    frames = []
    for camera in sorted((dataset / "rgb").glob("cam_*")):
        if not camera.is_dir() or not re.fullmatch(r"cam_\d+", camera.name):
            continue
        files = sorted(p for p in camera.glob("frame_*.png") if re.fullmatch(r"frame_\d+", p.stem))
        frames.extend(files[:limit] if limit else files)
    if not frames:
        raise ValueError(f"No RGB frames found under {dataset / 'rgb/cam_NNN'}")
    return frames


def check_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
        raise ValueError("Masks must be uint8 0/255 with the original RGB dimensions")
    return mask


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.png.tmp')
    Image.fromarray(mask).save(temporary, format='PNG')
    temporary.replace(path)


def load_segmenter(args):
    if getattr(args, 'prompt_points', None):
        from sam2_point_guidance import load_segmenter as load_guided
        return load_guided(args)
    root = args.model_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "grounding_dino"))
    # All weights already exist locally; use the read-only BERT snapshot directly.
    snapshots = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased/snapshots"
    bert = next((p for p in sorted(snapshots.glob("*")) if (p / "config.json").is_file() and
                 ((p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file())), None)
    if bert is None:
        raise FileNotFoundError(f"No cached bert-base-uncased model: {snapshots}")
    config = root / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_weights = root / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    sam_weights = root / "checkpoints/sam2.1_hiera_base_plus.pt"
    for path in [config, dino_weights, sam_weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the grounded-sam2 environment; CPU requires explicit --device cpu")
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.models import build_model
    from groundingdino.util.inference import load_image, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    configuration = SLConfig.fromfile(str(config))
    configuration.device = args.device
    configuration.text_encoder_type = str(bert)
    print(f"Loading local models on {args.device}; no downloads", flush=True)
    dino = build_model(configuration)
    dino.load_state_dict(clean_state_dict(torch.load(dino_weights, map_location="cpu")["model"]), strict=False)
    dino = dino.eval().to(args.device)
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(sam_weights), device=args.device))
    if getattr(args, "leica_guidance", False):
        from leica_mask_guidance import LeicaGuidance
        guidance = LeicaGuidance()
    else:
        guidance = None

    def segment(path, prompt):
        rgb, tensor = load_image(str(path))
        height, width = rgb.shape[:2]
        combined = np.zeros((height, width), dtype=np.uint8)
        if guidance is not None:
            from leica_mask_guidance import large_qr_occluder
            marker=large_qr_occluder(rgb)
            if marker is not None:
                return combined,dict(detections=0,selection_status="excluded_large_qr_occluder",exclude_from_training=True,
                                     exclusion_reason="A large calibration QR board occludes the acquisition view",qr_polygon=marker)
        with torch.inference_mode():
            boxes, confidence, _ = predict(dino, tensor, prompt, args.box_thr, args.text_thr, device=args.device)
            boxes = boxes.detach().cpu().numpy() * [width, height, width, height]
            xyxy = np.column_stack([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2])
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
            selection = {}
            if guidance is not None:
                index, selection = guidance.select(path, xyxy)
                if selection.get("requires_detector_fallback") and prompt.strip().lower() != "fridge":
                    fallback, fallback_confidence, _ = predict(dino,tensor,"fridge",args.box_thr,args.text_thr,device=args.device)
                    extra=fallback.detach().cpu().numpy()*[width,height,width,height]
                    extra=np.column_stack([extra[:,:2]-extra[:,2:]/2,extra[:,:2]+extra[:,2:]/2])
                    extra[:,[0,2]]=extra[:,[0,2]].clip(0,width);extra[:,[1,3]]=extra[:,[1,3]].clip(0,height)
                    xyxy=np.concatenate([xyxy,extra]);confidence=torch.cat([confidence,fallback_confidence])
                    index,selection=guidance.select(path,xyxy)
                    selection.update(primary_prompt=prompt,detector_fallback_prompt="fridge")
                selection["selected_detection_confidence"] = float(confidence[index].item()) if index is not None else None
                xyxy = xyxy[index:index+1] if index is not None else np.asarray([selection["fallback_box"]]) if "fallback_box" in selection else xyxy[:0]
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else contextlib.nullcontext()
            with autocast:
                if len(xyxy):
                    predictor.set_image(rgb)  # SAM2 expects RGB, not OpenCV BGR.
                for box in xyxy:
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    kwargs = {}
                    positive = selection.get("sam_positive_points", [])
                    negative = selection.get("sam_negative_points", [])
                    positive=[p for p in positive if box[0]<=p[0]<=box[2] and box[1]<=p[1]<=box[3]]
                    negative=[p for p in negative if box[0]<=p[0]<=box[2] and box[1]<=p[1]<=box[3]]
                    if selection.get("selection_status")=="leica_box_fallback":
                        positive=positive[:2]  # Lower static anchors may project onto the foreground acquisition rig.
                    if guidance is not None:
                        selection.update(sam_used_positive_points=positive,sam_used_negative_points=negative)
                    if positive:
                        kwargs.update(point_coords=np.asarray(positive+negative,dtype=np.float32),
                                      point_labels=np.asarray([1]*len(positive)+[0]*len(negative),dtype=np.int32))
                    masks, scores, _ = predictor.predict(box=box, multimask_output=True, **kwargs)
                    if len(scores):
                        ranked=np.array(scores,copy=True)
                        for mi,candidate in enumerate(masks):
                            if positive:ranked[mi]+=np.mean([candidate[int(y),int(x)]>.5 for x,y in positive])
                            if negative:ranked[mi]-=2*np.mean([candidate[int(y),int(x)]>.5 for x,y in negative])
                        selected=(masks[int(np.argmax(ranked))] > .5).astype(np.uint8)*255
                        # SAM point prompts can grow onto a foreground rig outside the appliance box.
                        bounded=np.zeros_like(selected);x0,y0,x1,y1=np.rint(box).astype(int)
                        bounded[max(0,y0):min(height,y1+1),max(0,x0):min(width,x1+1)]=selected[max(0,y0):min(height,y1+1),max(0,x0):min(width,x1+1)]
                        combined |= bounded
        return combined, dict(detections=len(confidence), primary_detections=len(boxes), max_detection_confidence=float(confidence.max().item()) if len(confidence) else None, **selection)

    return segment


def generate(args, segment=None):
    dataset = args.input.expanduser().resolve()
    exclude_prompt = (getattr(args, 'exclude_prompt', None) or '').strip()
    prompt_points = getattr(args, 'prompt_points', None)
    if prompt_points and (exclude_prompt or getattr(args, 'leica_guidance', False)):
        raise ValueError('--prompt-points uses explicit SAM guidance; omit --exclude-prompt and --leica-guidance')
    if exclude_prompt and getattr(args, 'leica_guidance', False):
        raise ValueError('Exclusion prompts require unguided text masking; do not combine with --leica-guidance')
    frames = list_frames(dataset, args.max_frames)
    settings = dict(prompt=args.prompt.strip(), box_threshold=args.box_thr, text_threshold=args.text_thr,
                    sam_model="sam2.1_hiera_base_plus", model_root=str(args.model_root.expanduser().resolve()))
    if exclude_prompt:
        settings.update(exclude_prompt=exclude_prompt, exclusion_rule='target AND NOT excluded; reject complete target removal v2')
    if prompt_points:
        from sam2_point_guidance import read_guidance
        settings.update(prompt_points_sha256=hashlib.sha256(Path(prompt_points).read_bytes()).hexdigest(),
                        spatial_guidance='explicit_per_camera_box_points_v1')
        if json.loads(Path(prompt_points).read_text()).get('robot_state_guidance'):
            from robot_mask_guidance import guidance_path
            robot_file=guidance_path(dataset,getattr(args,'robot_guidance_root',None))
            settings['robot_guidance_sha256']=hashlib.sha256(robot_file.read_bytes()).hexdigest()
        camera_frames = {}
        for path in frames:
            camera_frames.setdefault(path.parent.name, path)
        for camera, path in camera_frames.items():
            with Image.open(path) as source: read_guidance(prompt_points,camera,source.size)
    if getattr(args, "leica_guidance", False):
        settings["leica_guidance"] = "appliance_positive_negative_v5"
        reference_files = [dataset/"metadata/leica_moving_part.json", dataset/"metadata/leica_appliance_guidance.json"]
        reference_files.extend(sorted((dataset/"metadata/poses").glob("cam_*.json")))
        settings["guidance_sha256"] = hashlib.sha256(b"".join(path.read_bytes() for path in reference_files if path.exists())).hexdigest()
    manifest_file = dataset / "metadata/mask_generation.json"
    previous = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    resume = getattr(args, 'resume', False)
    if resume and (args.overwrite or not previous):
        raise ValueError('--resume requires an existing matching generation manifest and cannot be combined with --overwrite')
    if previous and previous.get("settings") != settings and not args.overwrite:
        raise ValueError("Mask settings/prompt changed. Use --overwrite to regenerate masks explicitly")
    print(f"{len(frames)} RGB frames; prompt={settings['prompt']!r}; masks -> {dataset / 'mask'}", flush=True)
    if exclude_prompt:
        print(f'Excluding {exclude_prompt!r}; separate exclusion masks -> {dataset / "mask_excluded"}', flush=True)
    camera_totals=Counter(path.parent.name for path in frames)
    camera_completed=Counter()
    print("Cameras: "+", ".join(f"{name}={count}" for name,count in camera_totals.items()),flush=True)
    if args.dry_run:
        return dict(requested=len(frames))
    for path in [dataset / "mask", dataset / 'mask_excluded', manifest_file]:
        if not path.resolve().is_relative_to(dataset):
            raise ValueError(f"Output escapes the dataset: {path}")
    configure_runtime(dataset)
    for directory in [dataset / "mask", dataset / "metadata"]:
        directory.mkdir(parents=True, exist_ok=True)
    records = {} if args.overwrite else previous.get("frames", {})
    # Refuse unknown masks before loading models or changing any files.
    for path in frames:
        relative = path.relative_to(dataset / "rgb")
        target = dataset / "mask" / relative
        if target.exists() and str(relative) not in records and not args.overwrite and not resume:
            raise ValueError(f"Existing mask has no generation record; use --overwrite: {target}")
    manifest = dict(settings=settings, frames=records, requested=len(frames), completed=0, empty=0, skipped=0)
    for index, path in enumerate(frames):
        relative = path.relative_to(dataset / "rgb")
        target = dataset / "mask" / relative
        excluded_path = dataset / 'mask_excluded' / relative
        if not target.resolve().is_relative_to(dataset):
            raise ValueError(f"Mask output escapes the dataset: {target}")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            shape = (image.height, image.width)
        old = records.get(str(relative), {})
        if target.exists() and str(relative) in records and not args.overwrite:
            if old.get("rgb_sha256") != source_hash:
                raise ValueError(f"RGB changed since masking; use --overwrite: {path}")
            with Image.open(target) as image:
                mask = check_mask(np.asarray(image), shape)
            if exclude_prompt:
                if not excluded_path.resolve().is_relative_to(dataset):
                    raise ValueError(f'Exclusion mask escapes the dataset: {excluded_path}')
                if not excluded_path.is_file() or hashlib.sha256(excluded_path.read_bytes()).hexdigest() != old.get('exclusion_sha256'):
                    raise ValueError(f'Exclusion mask missing or changed; use --overwrite: {excluded_path}')
            details = old
            manifest["skipped"] += 1
        else:
            if segment is None:
                segment = load_segmenter(args)
            mask, details = segment(path, settings["prompt"])
            mask = check_mask(mask, shape)
            if exclude_prompt:
                excluded, exclusion_details = segment(path, exclude_prompt)
                excluded = check_mask(excluded, shape)
                removed = int(np.count_nonzero((mask > 0) & (excluded > 0)))
                if removed and removed == int(np.count_nonzero(mask)):
                    raise ValueError(f'{relative}: exclusion removes the entire target; use spatial --prompt-points guidance instead of another text-only subtraction')
                mask = np.where(excluded > 0, 0, mask).astype(np.uint8)
                if not excluded_path.resolve().is_relative_to(dataset):
                    raise ValueError(f'Exclusion mask output escapes the dataset: {excluded_path}')
                save_mask(excluded_path, excluded)
                details = dict(details, exclusion=exclusion_details, removed_pixels=removed,
                    excluded_pixels=int(np.count_nonzero(excluded)),
                    exclusion_mask=str(excluded_path.relative_to(dataset)),
                    exclusion_sha256=hashlib.sha256(excluded_path.read_bytes()).hexdigest())
            save_mask(target, mask)
        count = int(np.count_nonzero(mask))
        records[str(relative)] = dict(details, rgb_sha256=source_hash, foreground_pixels=count,
                                      foreground_fraction=count / mask.size, status="nonempty" if count else "empty")
        manifest["completed"] += 1
        camera_completed[path.parent.name] += 1
        manifest["empty"] += count == 0
        write_json(manifest_file, manifest)
        if index % 25 == 0 or index == len(frames) - 1:
            camera=path.parent.name
            print(f"{camera}: {camera_completed[camera]}/{camera_totals[camera]} masks; total {index + 1}/{len(frames)}; empty={manifest['empty']}; reused={manifest['skipped']}", flush=True)
    if not args.max_frames and (not manifest["empty"] or getattr(args,"leica_guidance",False)) and all((dataset / model).is_dir() for model in ["colmap", "colmap_final"]):
        from colmap_masking import apply_scene_masks
        fraction = getattr(args, 'point_fraction', None)
        if fraction is None:
            apply_scene_masks(dataset)
        else:
            apply_scene_masks(dataset, fraction=fraction)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Converted scene containing rgb/cam_NNN/*.png")
    parser.add_argument("--prompt", required=True, help='Object description, e.g. "red fridge" or "refrigerator"')
    parser.add_argument("--exclude-prompt", help='Segment separately and subtract from the target, e.g. "robotic arm"; saves masks in mask_excluded/')
    parser.add_argument('--prompt-points',type=Path,help='JSON with per-camera boxes and positive/negative pixel points; uses SAM2 directly without ambiguous text detections')
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--box-thr", type=float, default=.25)
    parser.add_argument("--text-thr", type=float, default=.25)
    parser.add_argument("--max-frames", type=int, default=0, help="Per-camera limit; zero means all frames")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace masks or change the prompt/settings")
    parser.add_argument('--resume',action='store_true',help='Resume a matching manifest: reuse recorded masks and regenerate missing or unrecorded masks from interrupted runs')
    parser.add_argument("--point-fraction", type=point_fraction, help="Real masked PCD/COLMAP point fraction (0, 1]; default saved setting or 1; does not change masks")
    parser.add_argument("--leica-guidance", action="store_true", help="HoI!: select the text detection supported by the annotated Leica part projected through the per-frame camera pose")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.prompt.strip() or args.max_frames < 0 or not 0 <= args.box_thr <= 1 or not 0 <= args.text_thr <= 1:
        parser.error("Prompt must be nonempty, frame limit nonnegative, and thresholds in [0, 1]")
    if args.exclude_prompt is not None and not args.exclude_prompt.strip():
        parser.error('Exclusion prompt must be nonempty')
    result = generate(args)
    if result.get("empty", 0):
        print(f"WARNING: {result['empty']} empty masks; inspect metadata/mask_generation.json and adjust the prompt if needed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

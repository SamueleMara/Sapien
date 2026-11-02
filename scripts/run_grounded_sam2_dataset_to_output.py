#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Grounded-SAM2 installation
# -----------------------------------------------------------------------------
DEFAULT_GROUNDED_SAM_ROOT = Path(
    "/home/samuelemara/Grounded-SAM-2-autodistill"
).resolve()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Settings:
    text_prompt: str = "fridge door"
    box_threshold: float = 0.15
    text_threshold: float = 0.25
    mask_threshold: float = 0.5
    jpeg_quality: int = 95


@dataclass(frozen=True)
class CameraSource:
    source_name: str      # e.g. cam_0
    target_name: str      # e.g. cam_000
    video_dir: Path
    depth_dir: Path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dataset_samuele_fridge_fixedK from its current repository layout "
            "into an output/episode_XXXXXX target-format layout while running "
            "GroundingDINO + SAM2 and saving both combined masks and instance masks."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Root of dataset_samuele_fridge_fixedK. Default: current working directory. "
            "Run the script from inside the dataset folder and normally omit this option."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset-root>/output",
    )
    parser.add_argument(
        "--grounded-sam-root",
        type=Path,
        default=DEFAULT_GROUNDED_SAM_ROOT,
        help=f"Grounded-SAM-2-autodistill repo root (default: {DEFAULT_GROUNDED_SAM_ROOT})",
    )
    parser.add_argument("--prompt", default="fridge door", help='GroundingDINO text prompt')
    parser.add_argument("--box-thr", type=float, default=0.15)
    parser.add_argument("--text-thr", type=float, default=0.25)
    parser.add_argument("--mask-thr", type=float, default=0.5)
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional episode indices to process, e.g. --episodes 0 2. Default: all.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="For testing: process at most this many frames per camera. 0 = all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate selected episode output folders from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered cameras/episodes and planned folders without loading models.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------
def validate_dataset_root(root: Path) -> None:
    required = [
        root / "videos" / "chunk-000",
        root / "depth" / "chunk-000",
        root / "meta",
        root / "data" / "chunk-000",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            "This does not look like dataset_samuele_fridge_fixedK. Missing:\n" + msg
        )


def camera_index(source_name: str) -> int:
    # source_name is expected to be cam_0, cam_1, ...
    try:
        return int(source_name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unsupported camera name: {source_name}") from exc


def target_camera_name(source_name: str) -> str:
    return f"cam_{camera_index(source_name):03d}"


def discover_cameras(root: Path) -> List[CameraSource]:
    video_chunk = root / "videos" / "chunk-000"
    depth_chunk = root / "depth" / "chunk-000"

    cameras: List[CameraSource] = []
    for video_dir in sorted(video_chunk.glob("observation.images.cam_*")):
        if not video_dir.is_dir():
            continue
        source_name = video_dir.name.removeprefix("observation.images.")
        depth_dir = depth_chunk / f"observation.depth.{source_name}"
        if not depth_dir.is_dir():
            raise SystemExit(
                f"RGB camera {source_name!r} exists but matching depth folder is missing: {depth_dir}"
            )
        cameras.append(
            CameraSource(
                source_name=source_name,
                target_name=target_camera_name(source_name),
                video_dir=video_dir,
                depth_dir=depth_dir,
            )
        )

    if not cameras:
        raise SystemExit(f"No observation.images.cam_* folders found under {video_chunk}")

    cameras.sort(key=lambda c: camera_index(c.source_name))
    return cameras


def discover_episode_indices(cameras: Sequence[CameraSource]) -> List[int]:
    per_camera: Dict[str, set[int]] = {}
    for cam in cameras:
        indices: set[int] = set()
        for p in cam.video_dir.glob("episode_*.mp4"):
            try:
                indices.add(int(p.stem.split("_")[-1]))
            except ValueError:
                continue
        per_camera[cam.source_name] = indices

    common = set.intersection(*(s for s in per_camera.values())) if per_camera else set()
    if not common:
        raise SystemExit("No episode_XXXXXX.mp4 files common to all discovered cameras.")

    for cam_name, indices in per_camera.items():
        if indices != common:
            only_here = sorted(indices - common)
            missing_here = sorted(common - indices)
            print(
                f"[WARN] Episode mismatch for {cam_name}: extra={only_here}, missing={missing_here}. "
                "Only episodes available for every camera will be processed."
            )
    return sorted(common)


# -----------------------------------------------------------------------------
# Grounded-SAM2 import/load
# -----------------------------------------------------------------------------
def configure_grounded_sam_imports(root: Path):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Grounded-SAM root does not exist: {root}")

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "grounding_dino"))

    try:
        from grounding_dino.groundingdino.util.inference import (  # type: ignore
            load_image,
            load_model,
            predict,
        )
        from autodistill_grounded_sam_2.helpers import load_SAM  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"Failed to import GroundedDINO/SAM2 from {root}. "
            "Activate the same conda environment used by the original script.\n"
            f"Original error: {exc}"
        ) from exc

    config = root / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint = root / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    if not config.is_file():
        raise SystemExit(f"GroundingDINO config not found: {config}")
    if not checkpoint.is_file():
        raise SystemExit(f"GroundingDINO checkpoint not found: {checkpoint}")

    return load_image, load_model, predict, load_SAM, config, checkpoint


# -----------------------------------------------------------------------------
# Target-format folder creation/copying
# -----------------------------------------------------------------------------
def episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def ensure_target_tree(episode_out: Path, cameras: Sequence[CameraSource]) -> None:
    # Folders present in the attached target-format example.
    for base in [
        episode_out / "rgb",
        episode_out / "depth_png",
        episode_out / "mask",
        episode_out / "mask_instances",
        episode_out / "video_rgb",
        episode_out / "metadata",
        episode_out / "colmap",
        episode_out / "pointcloud",
    ]:
        base.mkdir(parents=True, exist_ok=True)

    for cam in cameras:
        for base in ["rgb", "depth_png", "mask", "mask_instances"]:
            (episode_out / base / cam.target_name).mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def copy_episode_metadata(dataset_root: Path, episode_out: Path, ep_idx: int, overwrite: bool) -> None:
    metadata_out = episode_out / "metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)

    # Copy the source metadata files unchanged.
    source_meta = dataset_root / "meta"
    for src in sorted(source_meta.iterdir()):
        if src.is_file():
            copy_file(src, metadata_out / src.name, overwrite=overwrite)

    # Copy the source parquet for this episode into metadata without modifying it.
    parquet = dataset_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    if parquet.is_file():
        copy_file(parquet, metadata_out / parquet.name, overwrite=overwrite)
    else:
        print(f"[WARN] Missing episode parquet: {parquet}")

    # Add a tiny manifest describing only the reorganization performed by this script.
    manifest_path = metadata_out / "grounded_sam2_output_manifest.json"
    if overwrite or not manifest_path.exists():
        manifest = {
            "source_dataset_root": str(dataset_root),
            "episode_index": ep_idx,
            "rgb_source": "videos/chunk-000/observation.images.cam_X/episode_XXXXXX.mp4",
            "depth_source": "depth/chunk-000/observation.depth.cam_X/episode_XXXXXX/frame_XXXXXX.png",
            "rgb_output": "rgb/cam_XXX/frame_XXXXXX.jpg",
            "depth_output": "depth_png/cam_XXX/frame_XXXXXX.png",
            "combined_mask_output": "mask/cam_XXX/frame_XXXXXX.png",
            "instance_mask_output": "mask_instances/cam_XXX/frame_XXXXXX_instance_YYY.png",
            "note": "Depth PNG files are copied byte-for-byte; RGB frames are decoded from the original MP4 files.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def copy_video(src_video: Path, episode_out: Path, target_cam: str, overwrite: bool) -> Path:
    dst = episode_out / "video_rgb" / f"{target_cam}.mp4"
    copy_file(src_video, dst, overwrite=overwrite)
    return dst


def copy_depth_sequence(
    src_dir: Path,
    dst_dir: Path,
    limit_frames: int,
    overwrite: bool,
) -> List[Path]:
    src_files = sorted(src_dir.glob("frame_*.png"))
    if limit_frames > 0:
        src_files = src_files[:limit_frames]

    if not src_files:
        raise RuntimeError(f"No depth frames found in {src_dir}")

    copied: List[Path] = []
    for src in tqdm(src_files, desc="copy depth", leave=False):
        dst = dst_dir / src.name
        copy_file(src, dst, overwrite=overwrite)
        copied.append(dst)
    return copied


# -----------------------------------------------------------------------------
# RGB extraction
# -----------------------------------------------------------------------------
def extract_rgb_frames(
    src_video: Path,
    rgb_out_dir: Path,
    expected_depth_files: Sequence[Path],
    jpeg_quality: int,
    overwrite: bool,
) -> List[Path]:
    """
    Decode the original MP4 and save frame_XXXXXX.jpg.

    The depth filenames are used as the synchronization reference.  This avoids
    silently creating differently indexed RGB/depth sequences.
    """
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_video}")

    rgb_paths: List[Path] = []
    try:
        for i, depth_path in enumerate(
            tqdm(expected_depth_files, desc="extract RGB", leave=False)
        ):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"Video {src_video} ended at frame {i}, but a matching depth frame "
                    f"exists ({depth_path.name})."
                )

            # Preserve the canonical frame index from the depth sequence.
            out_name = depth_path.with_suffix(".jpg").name
            out_path = rgb_out_dir / out_name
            if overwrite or not out_path.exists():
                ok_write = cv2.imwrite(
                    str(out_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
                )
                if not ok_write:
                    raise RuntimeError(f"cv2.imwrite failed: {out_path}")
            rgb_paths.append(out_path)
    finally:
        cap.release()

    return rgb_paths


# -----------------------------------------------------------------------------
# GroundedDINO + SAM2 masking
# -----------------------------------------------------------------------------
def safe_write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def remove_old_instance_masks(instance_dir: Path, stem: str) -> None:
    for p in instance_dir.glob(f"{stem}_instance_*.png"):
        p.unlink()


def run_masks_for_frame(
    image_path: Path,
    combined_mask_path: Path,
    instance_dir: Path,
    gdino,
    sam2_predictor,
    load_image,
    predict,
    settings: Settings,
    overwrite: bool,
) -> int:
    stem = image_path.stem

    # If the combined mask already exists and overwrite is disabled, leave the
    # frame untouched. This makes interrupted runs restartable.
    if combined_mask_path.is_file() and not overwrite:
        return len(list(instance_dir.glob(f"{stem}_instance_*.png")))

    orig_img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if orig_img_bgr is None:
        raise RuntimeError(f"Failed to read extracted RGB image: {image_path}")

    image_source, image = load_image(str(image_path))
    h, w, _ = image_source.shape

    # Preserve the behavior of the user's existing Grounded-SAM2 batch script.
    sam2_predictor.set_image(orig_img_bgr)

    boxes, _, _ = predict(
        model=gdino,
        image=image,
        caption=settings.text_prompt,
        box_threshold=settings.box_threshold,
        text_threshold=settings.text_threshold,
        device=DEVICE,
    )

    masks_list: List[np.ndarray] = []
    if len(boxes) > 0:
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes = boxes * scale
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        for box in boxes_xyxy:
            masks_per_box, scores, _ = sam2_predictor.predict(
                box=box,
                multimask_output=True,
            )
            if len(scores) == 0:
                continue
            best_idx = int(np.argmax(scores))
            mask_bin = (
                (masks_per_box[best_idx] > settings.mask_threshold).astype(np.uint8) * 255
            )
            masks_list.append(mask_bin)

    # Always save one combined mask so RGB/depth/mask indexing remains complete.
    if masks_list:
        combined = np.maximum.reduce(masks_list).astype(np.uint8)
    else:
        combined = np.zeros((h, w), dtype=np.uint8)
    safe_write_png(combined_mask_path, combined)

    # Keep instance results in a separate parallel hierarchy.
    remove_old_instance_masks(instance_dir, stem)
    for idx, mask in enumerate(masks_list):
        inst_path = instance_dir / f"{stem}_instance_{idx:03d}.png"
        safe_write_png(inst_path, mask)

    return len(masks_list)


def process_camera(
    dataset_root: Path,
    episode_out: Path,
    ep_idx: int,
    cam: CameraSource,
    gdino,
    sam2_predictor,
    load_image,
    predict,
    settings: Settings,
    limit_frames: int,
    overwrite: bool,
) -> None:
    ep = episode_name(ep_idx)
    src_video = cam.video_dir / f"{ep}.mp4"
    src_depth_dir = cam.depth_dir / ep

    if not src_video.is_file():
        raise RuntimeError(f"Missing video: {src_video}")
    if not src_depth_dir.is_dir():
        raise RuntimeError(f"Missing depth directory: {src_depth_dir}")

    rgb_out = episode_out / "rgb" / cam.target_name
    depth_out = episode_out / "depth_png" / cam.target_name
    mask_out = episode_out / "mask" / cam.target_name
    instances_out = episode_out / "mask_instances" / cam.target_name

    print(f"\n[INFO] {ep} | {cam.source_name} -> {cam.target_name}")

    copy_video(src_video, episode_out, cam.target_name, overwrite=overwrite)
    depth_files = copy_depth_sequence(
        src_depth_dir,
        depth_out,
        limit_frames=limit_frames,
        overwrite=overwrite,
    )
    rgb_files = extract_rgb_frames(
        src_video,
        rgb_out,
        expected_depth_files=depth_files,
        jpeg_quality=settings.jpeg_quality,
        overwrite=overwrite,
    )

    if len(rgb_files) != len(depth_files):
        raise RuntimeError(
            f"Internal synchronization error for {ep}/{cam.target_name}: "
            f"RGB={len(rgb_files)}, depth={len(depth_files)}"
        )

    total_instances = 0
    frames_with_instances = 0
    for rgb_path in tqdm(rgb_files, desc=f"SAM2 {cam.target_name}", leave=False):
        combined_path = mask_out / f"{rgb_path.stem}.png"
        n_inst = run_masks_for_frame(
            image_path=rgb_path,
            combined_mask_path=combined_path,
            instance_dir=instances_out,
            gdino=gdino,
            sam2_predictor=sam2_predictor,
            load_image=load_image,
            predict=predict,
            settings=settings,
            overwrite=overwrite,
        )
        total_instances += n_inst
        frames_with_instances += int(n_inst > 0)

    print(
        f"[INFO] {ep}/{cam.target_name}: {len(rgb_files)} RGB, "
        f"{len(depth_files)} depth, {len(rgb_files)} combined masks, "
        f"{total_instances} instance masks across {frames_with_instances} detected frames."
    )


def write_episode_summary(
    episode_out: Path,
    ep_idx: int,
    cameras: Sequence[CameraSource],
    settings: Settings,
) -> None:
    summary = {
        "episode": episode_name(ep_idx),
        "camera_map": {c.source_name: c.target_name for c in cameras},
        "prompt": settings.text_prompt,
        "box_threshold": settings.box_threshold,
        "text_threshold": settings.text_threshold,
        "mask_threshold": settings.mask_threshold,
        "device": DEVICE,
        "layout": {
            "rgb": "rgb/cam_XXX/frame_XXXXXX.jpg",
            "depth": "depth_png/cam_XXX/frame_XXXXXX.png",
            "mask": "mask/cam_XXX/frame_XXXXXX.png",
            "mask_instances": "mask_instances/cam_XXX/frame_XXXXXX_instance_YYY.png",
            "video": "video_rgb/cam_XXX.mp4",
        },
    }
    path = episode_out / "metadata" / "grounded_sam2_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    validate_dataset_root(dataset_root)

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "output"
    )

    settings = Settings(
        text_prompt=args.prompt,
        box_threshold=args.box_thr,
        text_threshold=args.text_thr,
        mask_threshold=args.mask_thr,
    )

    cameras = discover_cameras(dataset_root)
    episodes = discover_episode_indices(cameras)
    if args.episodes is not None:
        requested = set(args.episodes)
        unknown = requested - set(episodes)
        if unknown:
            raise SystemExit(f"Requested episodes do not exist for every camera: {sorted(unknown)}")
        episodes = [e for e in episodes if e in requested]

    if not episodes:
        raise SystemExit("No episodes selected.")

    print(f"[INFO] Dataset root : {dataset_root}")
    print(f"[INFO] Output root  : {output_root}")
    print(f"[INFO] Device       : {DEVICE}")
    print(f"[INFO] Episodes     : {[episode_name(e) for e in episodes]}")
    print("[INFO] Camera map   :")
    for cam in cameras:
        print(f"       {cam.source_name} -> {cam.target_name}")

    for ep_idx in episodes:
        ep_out = output_root / episode_name(ep_idx)
        print(f"[INFO] Planned target: {ep_out}")

    if args.dry_run:
        return

    load_image, load_model, predict, load_SAM, gdino_config, gdino_checkpoint = (
        configure_grounded_sam_imports(args.grounded_sam_root)
    )

    print("[INFO] Loading GroundingDINO...")
    gdino = load_model(
        model_config_path=str(gdino_config),
        model_checkpoint_path=str(gdino_checkpoint),
        device=DEVICE,
    )
    print("[INFO] Loading SAM2...")
    sam2_predictor = load_SAM()

    output_root.mkdir(parents=True, exist_ok=True)

    for ep_idx in tqdm(episodes, desc="Episodes", leave=True):
        ep_out = output_root / episode_name(ep_idx)
        if args.overwrite and ep_out.exists():
            shutil.rmtree(ep_out)

        ensure_target_tree(ep_out, cameras)
        copy_episode_metadata(dataset_root, ep_out, ep_idx, overwrite=args.overwrite)

        for cam in cameras:
            process_camera(
                dataset_root=dataset_root,
                episode_out=ep_out,
                ep_idx=ep_idx,
                cam=cam,
                gdino=gdino,
                sam2_predictor=sam2_predictor,
                load_image=load_image,
                predict=predict,
                settings=settings,
                limit_frames=args.limit_frames,
                overwrite=args.overwrite,
            )

        write_episode_summary(ep_out, ep_idx, cameras, settings)

    print("\n[INFO] Done.")
    print(f"[INFO] Final dataset written under: {output_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import logging
from importlib import metadata
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file does not exist: {p}")

    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            data = json.load(f)
        elif p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config extension '{p.suffix}'. Use .json/.yaml/.yml")

    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} must be a mapping/object")
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_writable_dir(path: str | Path) -> Path:
    out = ensure_dir(path)
    probe = out / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise PermissionError(f"Output directory is not writable: {out}") from exc
    return out


def ensure_output_layout(
    output_dir: str | Path,
    camera_names: list[str],
    save_depth_npy: bool,
    save_depth_png: bool,
    save_video: bool,
) -> dict[str, Path]:
    root = ensure_writable_dir(output_dir)

    dirs: dict[str, Path] = {
        "root": root,
        "metadata": ensure_dir(root / "metadata"),
        "rgb": ensure_dir(root / "rgb"),
        "mask": ensure_dir(root / "mask"),
    }

    if save_depth_npy:
        dirs["depth_npy"] = ensure_dir(root / "depth_npy")
    if save_depth_png:
        dirs["depth_png"] = ensure_dir(root / "depth_png")
    if save_video:
        dirs["video_rgb"] = ensure_dir(root / "video_rgb")

    for cam_name in camera_names:
        ensure_dir(dirs["rgb"] / cam_name)
        ensure_dir(dirs["mask"] / cam_name)
        if "depth_npy" in dirs:
            ensure_dir(dirs["depth_npy"] / cam_name)
        if "depth_png" in dirs:
            ensure_dir(dirs["depth_png"] / cam_name)

    return dirs


def write_json(data: Any, out_path: str | Path, indent: int = 2) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=True)


def write_yaml(data: Any, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_rgb_png(rgb: np.ndarray, out_path: str | Path) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (H,W,3), got {rgb.shape}")
    imageio.imwrite(out_path, rgb)


def save_mask_png(mask: np.ndarray, out_path: str | Path) -> None:
    if mask.ndim != 2:
        raise ValueError(f"Mask image must have shape (H,W), got {mask.shape}")
    imageio.imwrite(out_path, mask.astype(np.uint8))


def save_depth_npy(depth_m: np.ndarray, out_path: str | Path) -> None:
    if depth_m.ndim != 2:
        raise ValueError(f"Depth array must have shape (H,W), got {depth_m.shape}")
    np.save(out_path, depth_m.astype(np.float32))


def save_depth_png_preview(
    depth_m: np.ndarray,
    out_path: str | Path,
    max_depth_m: float | None = None,
) -> None:
    """
    Save depth visualization as 16-bit PNG.
    This image is for preview only; .npy stores authoritative metric depth.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"Depth array must have shape (H,W), got {depth_m.shape}")

    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        preview = np.zeros(depth_m.shape, dtype=np.uint16)
        imageio.imwrite(out_path, preview)
        return

    if max_depth_m is None:
        max_depth_m = float(np.percentile(depth_m[valid], 99.0))
        max_depth_m = max(max_depth_m, 1e-3)

    scaled = np.zeros(depth_m.shape, dtype=np.float64)
    scaled[valid] = np.clip(depth_m[valid] / max_depth_m, 0.0, 1.0)
    preview = (scaled * 65535.0).astype(np.uint16)
    imageio.imwrite(out_path, preview)


def write_frame_values_csv(
    out_path: str | Path,
    frame_values: list[tuple[int, float]],
    joint_name: str,
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", joint_name])
        for frame_idx, value in frame_values:
            writer.writerow([frame_idx, f"{value:.12g}"])


def get_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read version for %s: %s", package, exc)
            versions[package] = "unknown"
    return versions

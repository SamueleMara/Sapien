#!/usr/bin/env python3
"""Generate camera JSON compatible with src/camera_utils.py.

This script writes a camera file with top-level "cameras" entries that can be
used directly with:

  python src/render_urdf_multiview.py render --views-json <generated.json> ...

Example:
  python scripts/generate_cameras_json.py \
    --output configs/cameras.circle.json \
    --num-cams 8 \
    --radius 1.5 \
    --height 0.8 \
    --target 0 0 0.3 \
    --width 640 \
    --height-px 480 \
    --near 0.01 \
    --far 10.0 \
    --fov-y-deg 45
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(vec: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize zero-length vector for {label}")
    return vec / norm


def look_at_camera_to_world(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix using x-forward, y-left, z-up convention."""
    forward = _normalize(target - position, "forward")
    left = _normalize(np.cross(up, forward), "left")
    true_up = _normalize(np.cross(forward, left), "up")

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = forward
    pose[:3, 1] = left
    pose[:3, 2] = true_up
    pose[:3, 3] = position
    return pose


def generate_positions_on_circle(
    num_cams: int,
    radius: float,
    z_height: float,
    center_xy: tuple[float, float],
    start_angle_deg: float,
) -> list[np.ndarray]:
    if num_cams <= 0:
        raise ValueError("num_cams must be > 0")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    center_x, center_y = center_xy
    start_rad = math.radians(start_angle_deg)
    positions: list[np.ndarray] = []

    for idx in range(num_cams):
        angle = start_rad + (2.0 * math.pi * idx / num_cams)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions.append(np.array([x, y, z_height], dtype=np.float64))

    return positions


def _camera_intrinsics(
    width: int,
    height: int,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> tuple[float, float, float, float]:
    fx_val = 525.0 if fx is None else float(fx)
    fy_val = 525.0 if fy is None else float(fy)
    cx_val = (width / 2.0) if cx is None else float(cx)
    cy_val = (height / 2.0) if cy is None else float(cy)
    return fx_val, fy_val, cx_val, cy_val


def generate_cameras(
    num_cams: int,
    radius: float,
    cam_height: float,
    target: tuple[float, float, float],
    up: tuple[float, float, float],
    width: int,
    height_px: int,
    near: float,
    far: float,
    fov_y_deg: float,
    start_angle_deg: float,
    circle_center_xy: tuple[float, float],
    name_prefix: str,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    include_extrinsics: bool,
) -> dict[str, Any]:
    if width <= 0 or height_px <= 0:
        raise ValueError("width and height-px must be > 0")
    if near <= 0:
        raise ValueError("near must be > 0")
    if far <= near:
        raise ValueError("far must be > near")

    target_np = np.asarray(target, dtype=np.float64)
    up_np = np.asarray(up, dtype=np.float64)

    positions = generate_positions_on_circle(
        num_cams=num_cams,
        radius=radius,
        z_height=cam_height,
        center_xy=circle_center_xy,
        start_angle_deg=start_angle_deg,
    )

    intrinsics_mode = any(v is not None for v in (fx, fy, cx, cy))
    if intrinsics_mode:
        fx_val, fy_val, cx_val, cy_val = _camera_intrinsics(
            width=width,
            height=height_px,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    cameras: list[dict[str, Any]] = []
    for idx, pos in enumerate(positions):
        camera: dict[str, Any] = {
            "name": f"{name_prefix}_{idx:03d}",
            "width": int(width),
            "height": int(height_px),
            "near": float(near),
            "far": float(far),
            "position": pos.tolist(),
            "target": target_np.tolist(),
            "up": up_np.tolist(),
        }

        if intrinsics_mode:
            camera["fx"] = float(fx_val)
            camera["fy"] = float(fy_val)
            camera["cx"] = float(cx_val)
            camera["cy"] = float(cy_val)
        else:
            camera["fov_y_deg"] = float(fov_y_deg)

        if include_extrinsics:
            t_world_camera = look_at_camera_to_world(pos, target_np, up_np)
            t_camera_world = np.linalg.inv(t_world_camera)
            camera["extrinsics"] = {
                "T_world_camera": t_world_camera.tolist(),
                "T_camera_world": t_camera_world.tolist(),
            }

        cameras.append(camera)

    return {
        "image_width": int(width),
        "image_height": int(height_px),
        "default_near": float(near),
        "default_far": float(far),
        "cameras": cameras,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate circular camera views JSON compatible with this renderer"
    )

    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--num-cams", type=int, required=True, help="Number of cameras")
    parser.add_argument("--radius", type=float, required=True, help="Camera circle radius")
    parser.add_argument("--height", type=float, required=True, help="Camera z height")

    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("TX", "TY", "TZ"),
        help="Look-at target",
    )
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("UX", "UY", "UZ"),
        help="Up vector",
    )

    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height-px", type=int, default=480, help="Image height")
    parser.add_argument("--near", type=float, default=0.01, help="Near plane")
    parser.add_argument("--far", type=float, default=10.0, help="Far plane")
    parser.add_argument("--fov-y-deg", type=float, default=45.0, help="Vertical FOV in degrees")

    parser.add_argument("--fx", type=float, default=None, help="Optional fx (enables intrinsics mode)")
    parser.add_argument("--fy", type=float, default=None, help="Optional fy (enables intrinsics mode)")
    parser.add_argument("--cx", type=float, default=None, help="Optional cx (enables intrinsics mode)")
    parser.add_argument("--cy", type=float, default=None, help="Optional cy (enables intrinsics mode)")

    parser.add_argument(
        "--start-angle-deg",
        type=float,
        default=0.0,
        help="Angular offset for first camera",
    )
    parser.add_argument(
        "--circle-center-xy",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("CX", "CY"),
        help="Center of camera circle in XY",
    )
    parser.add_argument("--name-prefix", type=str, default="cam", help="Camera name prefix")
    parser.add_argument(
        "--include-extrinsics",
        action="store_true",
        help="Include debug extrinsics matrices under each camera entry",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_cameras(
        num_cams=args.num_cams,
        radius=args.radius,
        cam_height=args.height,
        target=(args.target[0], args.target[1], args.target[2]),
        up=(args.up[0], args.up[1], args.up[2]),
        width=args.width,
        height_px=args.height_px,
        near=args.near,
        far=args.far,
        fov_y_deg=args.fov_y_deg,
        start_angle_deg=args.start_angle_deg,
        circle_center_xy=(args.circle_center_xy[0], args.circle_center_xy[1]),
        name_prefix=args.name_prefix,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        include_extrinsics=args.include_extrinsics,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=args.indent)

    print(f"Saved camera JSON: {output_path}")
    print(f"Cameras generated: {len(payload['cameras'])}")


if __name__ == "__main__":
    main()

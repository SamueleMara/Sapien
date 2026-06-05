#!/usr/bin/env python3
"""
generate_sapien_cameras_json.py

Generate a JSON file for multi-view SAPIEN rendering.

Schema:
{
  "image_width": 640,
  "image_height": 480,
  "cameras": [
    {
      "name": "cam_000",
      "width": 640,
      "height": 480,
      "fx": 525.0,
      "fy": 525.0,
      "cx": 320.0,
      "cy": 240.0,
      "position": [x, y, z],
      "target": [tx, ty, tz],
      "up": [ux, uy, uz]
    }
  ]
}

Features:
- Cameras equally spaced on a circle
- Fixed radius and height
- All cameras point toward a target center
- Optional custom intrinsics
- Sensible defaults if intrinsics are omitted
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def build_intrinsics(
    width: int,
    height: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> Dict[str, float]:
    return {
        "fx": 525.0 if fx is None else float(fx),
        "fy": 525.0 if fy is None else float(fy),
        "cx": width / 2.0 if cx is None else float(cx),
        "cy": height / 2.0 if cy is None else float(cy),
    }


def generate_cameras(
    num_cams: int,
    radius: float,
    cam_height: float,
    target: Sequence[float],
    width: int,
    height_px: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    start_angle_deg: float,
    circle_center_xy: Sequence[float],
    up: Sequence[float],
) -> Dict:
    if num_cams <= 0:
        raise ValueError("num_cams must be > 0")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    intr = build_intrinsics(width, height_px, fx, fy, cx, cy)

    cx0, cy0 = float(circle_center_xy[0]), float(circle_center_xy[1])
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    ux, uy, uz = float(up[0]), float(up[1]), float(up[2])

    start_angle_rad = math.radians(start_angle_deg)

    cameras: List[Dict] = []
    for i in range(num_cams):
        angle = start_angle_rad + 2.0 * math.pi * i / num_cams
        px = cx0 + radius * math.cos(angle)
        py = cy0 + radius * math.sin(angle)
        pz = cam_height

        cameras.append(
            {
                "name": f"cam_{i:03d}",
                "width": width,
                "height": height_px,
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "position": [px, py, pz],
                "target": [tx, ty, tz],
                "up": [ux, uy, uz],
            }
        )

    return {
        "image_width": width,
        "image_height": height_px,
        "cameras": cameras,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAPIEN camera JSON with cameras on a circle."
    )

    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--num-cams", type=int, required=True, help="Number of cameras")
    parser.add_argument("--radius", type=float, required=True, help="Circle radius")
    parser.add_argument("--cam-height", type=float, required=True, help="Camera z height")

    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("TX", "TY", "TZ"),
        help="Target point all cameras look at",
    )

    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")

    parser.add_argument("--fx", type=float, default=None, help="Focal length fx")
    parser.add_argument("--fy", type=float, default=None, help="Focal length fy")
    parser.add_argument("--cx", type=float, default=None, help="Principal point cx")
    parser.add_argument("--cy", type=float, default=None, help="Principal point cy")

    parser.add_argument(
        "--start-angle-deg",
        type=float,
        default=0.0,
        help="Angle of the first camera in degrees",
    )

    parser.add_argument(
        "--circle-center-xy",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("CX", "CY"),
        help="Center of the camera circle in the XY plane",
    )

    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("UX", "UY", "UZ"),
        help="Up vector for all cameras",
    )

    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = generate_cameras(
        num_cams=args.num_cams,
        radius=args.radius,
        cam_height=args.cam_height,
        target=args.target,
        width=args.width,
        height_px=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        start_angle_deg=args.start_angle_deg,
        circle_center_xy=args.circle_center_xy,
        up=args.up,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=args.indent)

    print(f"Saved {len(data['cameras'])} cameras to {output_path}")


if __name__ == "__main__":
    main()
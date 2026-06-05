from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from camera_utils import load_camera_specs
from config_types import CameraSpec, RenderJobConfig, RootPoseSpec
from io_utils import load_json_or_yaml
from sapien_urdf_renderer import SapienURDFRenderer

LOGGER = logging.getLogger("render_urdf_multiview")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_root_pose_from_args(args: argparse.Namespace) -> RootPoseSpec | None:
    if args.root_position is None and args.root_quaternion_wxyz is None:
        return None

    position = args.root_position or [0.0, 0.0, 0.0]
    quat = args.root_quaternion_wxyz or [1.0, 0.0, 0.0, 0.0]
    return RootPoseSpec(
        position=[float(v) for v in position],
        quaternion_wxyz=[float(v) for v in quat],
    )


def _print_joint_rows(rows: list[dict[str, Any]]) -> None:
    print("index\tname\ttype\tdof\tlimits")
    for row in rows:
        print(
            f"{row['index']}\t{row['name']}\t{row['type']}\t{row['dof']}\t{row['limits']}"
        )


def cmd_list_joints(args: argparse.Namespace) -> int:
    renderer = SapienURDFRenderer()
    root_pose = _build_root_pose_from_args(args)
    renderer.load_articulation(args.urdf, root_pose=root_pose)
    rows = renderer.list_active_joints()

    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        _print_joint_rows(rows)
    return 0


def _build_job_config_from_render_args(args: argparse.Namespace) -> RenderJobConfig:
    cameras = load_camera_specs(
        config_path=args.views_json,
        image_width_override=args.image_width,
        image_height_override=args.image_height,
        default_near=args.near,
        default_far=args.far,
    )

    root_pose = _build_root_pose_from_args(args)
    cfg = RenderJobConfig(
        urdf_path=args.urdf,
        joint_name=args.joint_name,
        q_start=float(args.q_start),
        q_end=float(args.q_end),
        num_frames=args.num_frames,
        fps=args.fps,
        duration_sec=args.duration_sec,
        cameras=cameras,
        output_dir=args.output_dir,
        root_pose=root_pose,
        image_width=args.image_width,
        image_height=args.image_height,
        default_near=args.near,
        default_far=args.far,
        save_depth_png=args.save_depth_png,
        save_depth_npy=args.save_depth_npy,
        save_video=args.save_video,
        video_fps=args.video_fps,
        video_codec=args.video_codec,
        background_color=[float(v) for v in args.background_color],
        dry_run=args.dry_run,
        list_joints_only=args.list_joints_only,
        add_ground=args.add_ground,
    )
    cfg.validate()
    return cfg


def _resolve_path_maybe_relative(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_job_config_from_file(config_path: str) -> RenderJobConfig:
    config_file = Path(config_path).expanduser().resolve()
    payload = load_json_or_yaml(config_file)

    cameras_override: list[CameraSpec] | None = None
    if payload.get("camera_config_path") is not None:
        camera_config_path = _resolve_path_maybe_relative(
            str(payload["camera_config_path"]), config_file.parent
        )
        cameras_override = load_camera_specs(
            config_path=camera_config_path,
            image_width_override=payload.get("image_width"),
            image_height_override=payload.get("image_height"),
            default_near=payload.get("default_near"),
            default_far=payload.get("default_far"),
        )

    if payload.get("urdf_path") is not None:
        payload["urdf_path"] = _resolve_path_maybe_relative(str(payload["urdf_path"]), config_file.parent)
    if payload.get("output_dir") is not None:
        payload["output_dir"] = _resolve_path_maybe_relative(str(payload["output_dir"]), config_file.parent)

    cfg = RenderJobConfig.from_dict(payload, cameras_override=cameras_override)
    cfg.validate()
    return cfg


def _list_only_mode(renderer: SapienURDFRenderer, cfg: RenderJobConfig) -> int:
    renderer.load_articulation(cfg.urdf_path, root_pose=cfg.root_pose)
    rows = renderer.list_active_joints()
    _print_joint_rows(rows)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg = _build_job_config_from_render_args(args)
    renderer = SapienURDFRenderer(
        background_color=tuple(float(v) for v in cfg.background_color),
        add_ground=cfg.add_ground,
    )

    if cfg.list_joints_only:
        return _list_only_mode(renderer, cfg)

    renderer.render_sequence(cfg)
    LOGGER.info("Rendering complete. Outputs written to: %s", cfg.resolved_output_dir())
    return 0


def cmd_render_config(args: argparse.Namespace) -> int:
    cfg = _load_job_config_from_file(args.config)

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.save_video:
        cfg.save_video = True

    cfg.validate()
    renderer = SapienURDFRenderer(
        background_color=tuple(float(v) for v in cfg.background_color),
        add_ground=cfg.add_ground,
    )

    if cfg.list_joints_only:
        return _list_only_mode(renderer, cfg)

    renderer.render_sequence(cfg)
    LOGGER.info("Rendering complete. Outputs written to: %s", cfg.resolved_output_dir())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render multi-view RGB/depth/mask sequences from URDF articulations using SAPIEN 2.2.2"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_joints = subparsers.add_parser("list-joints", help="List active joints in a URDF")
    list_joints.add_argument("--urdf", required=True, help="Path to URDF")
    list_joints.add_argument(
        "--root-position",
        type=float,
        nargs=3,
        default=None,
        help="Optional root position xyz",
    )
    list_joints.add_argument(
        "--root-quaternion-wxyz",
        type=float,
        nargs=4,
        default=None,
        help="Optional root orientation quaternion (w x y z)",
    )
    list_joints.add_argument("--as-json", action="store_true", help="Print as JSON")
    list_joints.set_defaults(func=cmd_list_joints)

    render = subparsers.add_parser("render", help="Render a sequence from direct CLI arguments")
    render.add_argument("--urdf", required=True, help="Path to URDF")
    render.add_argument("--joint-name", required=True, help="Active joint name")
    render.add_argument("--q-start", required=True, type=float, help="Initial joint value")
    render.add_argument("--q-end", required=True, type=float, help="Final joint value")
    render.add_argument("--num-frames", type=int, default=None, help="Number of frames (inclusive endpoints)")
    render.add_argument("--fps", type=float, default=None, help="FPS (used with --duration-sec)")
    render.add_argument("--duration-sec", type=float, default=None, help="Duration in seconds")
    render.add_argument("--views-json", required=True, help="Camera definitions file (.json/.yaml/.yml)")
    render.add_argument("--output-dir", required=True, help="Output directory")
    render.add_argument("--image-width", type=int, default=None, help="Optional width override for all cameras")
    render.add_argument("--image-height", type=int, default=None, help="Optional height override for all cameras")
    render.add_argument("--near", type=float, default=None, help="Optional near-plane override")
    render.add_argument("--far", type=float, default=None, help="Optional far-plane override")
    render.add_argument("--save-video", action="store_true", help="Save per-camera RGB MP4")
    render.add_argument("--video-fps", type=float, default=None, help="Video FPS override")
    render.add_argument("--video-codec", default="libx264", help="Video codec for imageio writer")
    render.add_argument("--save-depth-png", action=argparse.BooleanOptionalAction, default=True)
    render.add_argument("--save-depth-npy", action=argparse.BooleanOptionalAction, default=True)
    render.add_argument(
        "--background-color",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Background color RGB in [0,1]",
    )
    render.add_argument(
        "--root-position",
        type=float,
        nargs=3,
        default=None,
        help="Optional root position xyz",
    )
    render.add_argument(
        "--root-quaternion-wxyz",
        type=float,
        nargs=4,
        default=None,
        help="Optional root orientation quaternion (w x y z)",
    )
    render.add_argument("--dry-run", action="store_true", help="Write metadata only and skip image rendering")
    render.add_argument("--list-joints-only", action="store_true", help="Print joints and exit")
    render.add_argument("--add-ground", action="store_true", help="Add a ground plane (default: disabled)")
    render.set_defaults(func=cmd_render)

    render_cfg = subparsers.add_parser("render-config", help="Render sequence from YAML/JSON config")
    render_cfg.add_argument("--config", required=True, help="Path to job config (.yaml/.yml/.json)")
    render_cfg.add_argument("--output-dir", default=None, help="Optional output dir override")
    render_cfg.add_argument("--save-video", action="store_true", help="Force-enable video output")
    render_cfg.set_defaults(func=cmd_render_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

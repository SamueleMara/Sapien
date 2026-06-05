#!/usr/bin/env python3
"""
Run the full URDF preparation pipeline over all URDF files.

Pipeline steps:
1. Resolve camera config (use provided one or generate one).
2. Discover all *.urdf files in the URDF directory (recursive).
3. For each URDF:
   - remap stale absolute mesh paths to current runtime assets (if needed),
   - list joints,
   - pick the first 1-DoF joint,
   - choose q_start/q_end from limits when available,
   - run the render pipeline into a per-URDF output folder.
4. Save a summary JSON with success/failure per URDF.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


OBJECT_VIEW_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "none": {},
    # Requested showcase profile:
    # - blade horizontal
    # - oven/storage front-facing for cam_000 in the default Z-orbit camera ring
    "blade_horizontal_stove_storage_front": {
        "blade_103706": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [0.7071067811865476, 0.7071067811865475, 0.0, 0.0],  # +90deg around X
        },
        "oven_101917": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [0.7071067811865476, 0.0, 0.0, -0.7071067811865475],  # -90deg yaw
        },
        "storage_45135": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [0.7071067811865476, 0.0, 0.0, -0.7071067811865475],  # -90deg yaw
        },
    },
}

# Effective orbit-axis switch (implemented via root-pose rotation) for all models
# except blade_103706. Cameras still orbit in world-Z; this remaps model frame.
NON_BLADE_ORBIT_AXIS_ROOT_POSE: dict[str, dict[str, Any]] = {
    # Identity: model-Z aligns with world-Z (default camera ring behavior).
    "z": {
        "position": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    # Rotate +90deg around X so model-Y aligns with world-Z.
    "y": {
        "position": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [0.7071067811865476, 0.7071067811865475, 0.0, 0.0],
    },
    # Rotate -90deg around Y so model-X aligns with world-Z.
    "x": {
        "position": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [0.7071067811865476, 0.0, -0.7071067811865475, 0.0],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run URDF preparation/render pipeline")

    parser.add_argument(
        "--urdf-dir",
        default=None,
        help=(
            "Directory containing URDF files (*.urdf). Search is recursive. "
            "Default: auto-detect (/workspace/assets/definitive_urdf/definitive_urdf, "
            "/workspace/assets/urdf_collection/urdf_collection, /workspace/assets, "
            "repo/assets/definitive_urdf/definitive_urdf, "
            "repo/assets/urdf_collection/urdf_collection, repo/assets, /data/assets)."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Deprecated alias of --urdf-dir (kept for backward compatibility)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output folder for all URDF results. Default: <repo>/output/preparation_all",
    )

    parser.add_argument(
        "--camera-config-json",
        default=None,
        help=(
            "Path to existing camera config file (.json/.yaml/.yml). "
            "If it exists, it is used directly (no generation)."
        ),
    )
    parser.add_argument(
        "--views-json",
        default=None,
        help=(
            "Alias/output path for camera config (.json/.yaml/.yml). "
            "Default prefers <repo>/configs/cameras.PARIS.json when available, "
            "otherwise <repo>/configs/cameras.pipeline.json."
        ),
    )

    parser.add_argument("--num-cams", type=int, default=8)
    parser.add_argument("--radius", type=float, default=1.4)
    parser.add_argument("--height", type=float, default=0.7)
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.3])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height-px", type=int, default=480)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=10.0)
    parser.add_argument("--fov-y-deg", type=float, default=45.0)

    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--object-view-profile",
        default="blade_horizontal_stove_storage_front",
        choices=sorted(OBJECT_VIEW_PROFILES.keys()),
        help=(
            "Optional per-object root-pose overrides for preferred visual orientation. "
            "Default: 'blade_horizontal_stove_storage_front' for PartNet definitive URDF rendering "
            "(blade horizontal, oven/storage front-facing)."
        ),
    )
    parser.add_argument(
        "--non-blade-orbit-axis",
        default="y",
        choices=["keep", "x", "y", "z"],
        help=(
            "Change effective camera orbit axis for all models except blade_103706. "
            "Implemented via root-pose rotation. "
            "Default: 'y' for PartNet definitive URDF rendering. "
            "'keep' disables this override."
        ),
    )
    parser.add_argument("--skip-camera-generation", action="store_true")
    parser.add_argument(
        "--regenerate-camera-config",
        action="store_true",
        help=(
            "Force regeneration of --views-json even if that file already exists. "
            "By default, existing camera configs are reused."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-renderer-preflight",
        action="store_true",
        help="Skip one-shot SAPIEN renderer initialization check before processing URDFs.",
    )
    parser.add_argument(
        "--force-link-colors",
        action="store_true",
        help=(
            "Inject deterministic colors into visual elements that do not already define "
            "a material color/texture. Useful for colorless OBJ-based URDF datasets."
        ),
    )
    parser.add_argument(
        "--force-color-seed",
        default="0",
        help="Seed string used to generate deterministic link colors with --force-link-colors.",
    )

    return parser.parse_args()


def resolve_urdf_dir(arg_urdf_dir: str | None, repo_root: Path) -> Path:
    if arg_urdf_dir:
        return Path(arg_urdf_dir).expanduser().resolve()

    candidates = [
        Path("/workspace/assets/definitive_urdf/definitive_urdf"),
        Path("/workspace/assets/urdf_collection/urdf_collection"),
        Path("/workspace/assets"),
        repo_root / "assets" / "definitive_urdf" / "definitive_urdf",
        repo_root / "assets" / "urdf_collection" / "urdf_collection",
        repo_root / "assets",
        Path("/data/assets"),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()

    return (repo_root / "assets").resolve()


def resolve_output_root(arg_output_root: str | None, repo_root: Path) -> Path:
    if arg_output_root:
        return Path(arg_output_root).expanduser().resolve()
    return (repo_root / "output" / "preparation_all").resolve()


def resolve_views_json(arg_views_json: str | None, repo_root: Path) -> Path:
    if arg_views_json:
        return Path(arg_views_json).expanduser().resolve()
    # Prefer the high-radius preset when present to avoid overly zoomed captures.
    preferred = (repo_root / "configs" / "cameras.PARIS.json").resolve()
    if preferred.exists():
        return preferred
    return (repo_root / "configs" / "cameras.pipeline.json").resolve()


def describe_camera_config(camera_config_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(camera_config_path),
        "exists": camera_config_path.exists(),
    }
    if not camera_config_path.exists():
        return info

    try:
        text = camera_config_path.read_text(encoding="utf-8")
        if camera_config_path.suffix.lower() in {".yaml", ".yml"}:
            parsed = yaml.safe_load(text)
            payload = parsed if isinstance(parsed, dict) else {}
        else:
            payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        info["parse_error"] = str(exc)
        return info

    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        info["num_cameras"] = 0
        return info

    info["num_cameras"] = len(cameras)
    if not cameras:
        return info

    first = cameras[0] if isinstance(cameras[0], dict) else {}
    position = first.get("position")
    target = first.get("target")
    if (
        isinstance(position, list)
        and len(position) == 3
        and all(isinstance(v, (int, float)) for v in position)
    ):
        px, py, pz = (float(position[0]), float(position[1]), float(position[2]))
        info["cam0_position"] = [px, py, pz]
        info["cam0_distance_origin"] = math.sqrt(px * px + py * py + pz * pz)
    if (
        isinstance(position, list)
        and isinstance(target, list)
        and len(position) == 3
        and len(target) == 3
        and all(isinstance(v, (int, float)) for v in [*position, *target])
    ):
        dx = float(position[0]) - float(target[0])
        dy = float(position[1]) - float(target[1])
        dz = float(position[2]) - float(target[2])
        info["cam0_distance_to_target"] = math.sqrt(dx * dx + dy * dy + dz * dz)

    return info


def run_command(cmd: list[str], dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return None

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details: list[str] = []
        if stderr:
            details.append(f"stderr: {stderr}")
        if stdout:
            details.append(f"stdout: {stdout}")
        detail_text = " | ".join(details) if details else "no stdout/stderr captured"
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)} | {detail_text}")

    return proc


def preflight_sapien_renderer(dry_run: bool = False) -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sapien.core as sapien; "
            "engine = sapien.Engine(); "
            "renderer = sapien.SapienRenderer(offscreen_only=True); "
            "engine.set_renderer(renderer); "
            "print('SAPIEN renderer preflight OK')"
        ),
    ]
    run_command(cmd, dry_run=dry_run)


def discover_urdfs(urdf_dir: Path) -> list[Path]:
    urdfs: list[Path] = []
    for path in urdf_dir.rglob("*.urdf"):
        if not path.is_file():
            continue
        parts = path.parts
        if "__MACOSX" in parts:
            continue
        if path.name.startswith("._"):
            continue
        urdfs.append(path)
    return sorted(urdfs)


def parse_joints_from_cli_output(stdout: str) -> list[dict[str, Any]]:
    data = json.loads(stdout)
    if not isinstance(data, list):
        raise ValueError("Unexpected joint list format; expected a JSON list")
    return data


def choose_joint_and_range(joints: list[dict[str, Any]]) -> tuple[str, float, float]:
    candidates = [j for j in joints if int(j.get("dof", 0)) == 1]
    if not candidates:
        raise RuntimeError("No 1-DoF active joints found")

    joint = candidates[0]
    name = str(joint["name"])

    q_start = -0.5
    q_end = 0.5

    limits = joint.get("limits")
    if isinstance(limits, list) and limits and isinstance(limits[0], list) and len(limits[0]) == 2:
        lo = float(limits[0][0])
        hi = float(limits[0][1])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            span = hi - lo
            margin = 0.05 * span
            q_start = lo + margin
            q_end = hi - margin
            if q_end <= q_start:
                q_start = lo
                q_end = hi

    return name, q_start, q_end


def safe_stem(path: Path) -> str:
    stem = path.stem
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)


def safe_model_name(urdf_path: Path, urdf_root: Path) -> str:
    """
    Stable output name from URDF path relative to root, to avoid collisions in recursive scans.
    Example: sub/USB_100109.urdf -> sub__USB_100109
    """
    try:
        rel = urdf_path.relative_to(urdf_root)
    except ValueError:
        return safe_stem(urdf_path)

    parts = list(rel.parts)
    if parts and parts[-1].lower().endswith(".urdf"):
        parts[-1] = Path(parts[-1]).stem
    raw = "__".join(parts) if parts else urdf_path.stem
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in raw)


def build_mesh_search_roots(repo_root: Path, urdf_dir: Path) -> list[Path]:
    roots = [
        urdf_dir,
        urdf_dir.parent,
        repo_root,
        repo_root / "assets",
        repo_root / "assets" / "sapien_unified",
        repo_root / "assets" / "sapien_unified" / "sapien_unified",
        repo_root / "assets" / "urdf_collection",
        repo_root / "assets" / "urdf_collection" / "urdf_collection",
    ]
    dedup: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        r = root.resolve()
        if r in seen:
            continue
        seen.add(r)
        if r.exists():
            dedup.append(r)
    return dedup


def _mesh_reference_exists(mesh_ref: str, urdf_path: Path) -> bool:
    normalized = mesh_ref.strip().replace("\\", "/")

    if normalized.startswith("file://"):
        return Path(normalized[len("file://") :]).expanduser().exists()

    # package:// references are resolver-specific (handled by renderer); treat as unresolved here.
    if normalized.startswith("package://"):
        return False

    mesh_path = Path(normalized).expanduser()
    if mesh_path.is_absolute():
        return mesh_path.exists()

    return (urdf_path.parent / mesh_path).resolve().exists()


def _resolve_missing_mesh_path(
    mesh_filename: str,
    urdf_path: Path,
    urdf_dir: Path,
    extra_search_roots: list[Path] | None = None,
) -> Path | None:
    normalized = mesh_filename.strip().replace("\\", "/")

    if normalized.startswith("file://"):
        normalized = normalized[len("file://") :]

    mesh_path = Path(normalized).expanduser()
    if mesh_path.is_absolute() and mesh_path.exists():
        return mesh_path.resolve()

    if not mesh_path.is_absolute():
        rel_candidate = (urdf_path.parent / mesh_path).resolve()
        if rel_candidate.exists():
            return rel_candidate

    search_roots: list[Path] = [
        urdf_path.parent.resolve(),
        urdf_dir.resolve(),
        urdf_dir.parent.resolve(),
    ]
    if extra_search_roots:
        search_roots.extend(root.resolve() for root in extra_search_roots)

    seen_roots: set[Path] = set()
    dedup_roots: list[Path] = []
    for root in search_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        dedup_roots.append(root)

    for marker in ("../meshes/", "./meshes/", "meshes/", "/assets/meshes/", "/meshes/"):
        idx = normalized.find(marker)
        if idx == -1:
            continue

        suffix = normalized[idx + len(marker) :].lstrip("/")
        if not suffix:
            continue

        for root in dedup_roots:
            candidate = (root / "meshes" / suffix).resolve()
            if candidate.exists():
                return candidate

    return None


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _iter_children_with_local_tag(parent: ET.Element, local_tag: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local_tag(str(child.tag)) == local_tag]


def _first_child_with_local_tag(parent: ET.Element, local_tag: str) -> ET.Element | None:
    for child in list(parent):
        if _local_tag(str(child.tag)) == local_tag:
            return child
    return None


def _material_has_visual_style(material_elem: ET.Element) -> bool:
    for child in list(material_elem):
        if _local_tag(str(child.tag)) in {"color", "texture"}:
            return True
    return False


def _collect_global_material_styles(root: ET.Element) -> dict[str, bool]:
    styles: dict[str, bool] = {}
    for child in list(root):
        if _local_tag(str(child.tag)) != "material":
            continue
        name = (child.attrib.get("name") or "").strip()
        if not name:
            continue
        styles[name] = _material_has_visual_style(child)
    return styles


def _format_rgba(r: float, g: float, b: float, a: float = 1.0) -> str:
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def _deterministic_link_rgb(link_name: str, visual_index: int, seed: str) -> tuple[float, float, float]:
    key = f"{seed}|{link_name}|{visual_index}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    hue = int.from_bytes(digest[0:2], "big") / 65535.0
    sat = 0.60 + 0.30 * (digest[2] / 255.0)
    val = 0.70 + 0.25 * (digest[3] / 255.0)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return r, g, b


def _inject_missing_visual_colors(root: ET.Element, seed: str) -> int:
    global_material_styles = _collect_global_material_styles(root)
    injected = 0

    for link in (elem for elem in root.iter() if _local_tag(str(elem.tag)) == "link"):
        link_name = (link.attrib.get("name") or "link").strip() or "link"
        visuals = _iter_children_with_local_tag(link, "visual")
        for visual_idx, visual in enumerate(visuals):
            material = _first_child_with_local_tag(visual, "material")
            if material is not None:
                if _material_has_visual_style(material):
                    continue
                material_name = (material.attrib.get("name") or "").strip()
                if material_name and global_material_styles.get(material_name, False):
                    continue
            else:
                material = ET.SubElement(visual, "material", {"name": f"auto_color_{link_name}_{visual_idx}"})

            if _material_has_visual_style(material):
                continue

            r, g, b = _deterministic_link_rgb(link_name=link_name, visual_index=visual_idx, seed=seed)
            ET.SubElement(material, "color", {"rgba": _format_rgba(r, g, b, 1.0)})
            injected += 1

    return injected


def prepare_runtime_urdf(
    urdf_path: Path,
    urdf_dir: Path,
    temp_dir: Path,
    force_link_colors: bool = False,
    force_color_seed: str = "0",
    extra_search_roots: list[Path] | None = None,
) -> tuple[Path, bool, int]:
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError:
        return urdf_path, False, 0

    root = tree.getroot()
    changed = False
    forced_colors = 0

    for mesh_elem in (elem for elem in root.iter() if _local_tag(str(elem.tag)) == "mesh"):
        for attr in ("filename", "url"):
            mesh_ref = mesh_elem.attrib.get(attr)
            if not mesh_ref:
                continue

            if _mesh_reference_exists(mesh_ref, urdf_path):
                continue

            replacement = _resolve_missing_mesh_path(
                mesh_ref,
                urdf_path=urdf_path,
                urdf_dir=urdf_dir,
                extra_search_roots=extra_search_roots,
            )
            if replacement is None:
                continue

            mesh_elem.set(attr, str(replacement))
            changed = True

    if force_link_colors:
        forced_colors = _inject_missing_visual_colors(root, seed=force_color_seed)
        if forced_colors > 0:
            changed = True

    if not changed:
        return urdf_path, False, 0

    out_path = temp_dir / urdf_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, True, forced_colors


def main() -> int:
    args = parse_args()

    if args.urdf_dir and args.assets_dir:
        p1 = Path(args.urdf_dir).expanduser().resolve()
        p2 = Path(args.assets_dir).expanduser().resolve()
        if p1 != p2:
            raise ValueError(
                "Both --urdf-dir and --assets-dir were provided with different values. "
                "Use only --urdf-dir (or pass the same path to both)."
            )

    selected_urdf_arg = args.urdf_dir if args.urdf_dir is not None else args.assets_dir

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]

    urdf_dir = resolve_urdf_dir(selected_urdf_arg, repo_root)
    output_root = resolve_output_root(args.output_root, repo_root)

    # --views-json is generation output target.
    views_json = resolve_views_json(args.views_json, repo_root)

    # --camera-config-json is explicit input camera file.
    camera_config_input: Path | None = None
    if args.camera_config_json:
        camera_config_input = Path(args.camera_config_json).expanduser().resolve()

    renderer_cli = repo_root / "src" / "render_urdf_multiview.py"
    camera_gen = repo_root / "scripts" / "generate_cameras_json.py"

    print(f"Resolved repo_root: {repo_root}")
    print(f"Resolved urdf_dir: {urdf_dir}")
    print(f"Resolved output_root: {output_root}")
    print(f"Resolved views_json: {views_json}")
    if args.force_link_colors:
        print(f"Force-link-colors enabled (seed={args.force_color_seed!r})")
    if args.object_view_profile != "none":
        print(f"Object-view profile enabled: {args.object_view_profile}")
    if args.non_blade_orbit_axis != "keep":
        print(f"Non-blade orbit-axis override enabled: {args.non_blade_orbit_axis}")

    if not urdf_dir.exists():
        raise FileNotFoundError(f"URDF directory does not exist: {urdf_dir}")
    if not renderer_cli.exists():
        raise FileNotFoundError(f"Renderer CLI not found: {renderer_cli}")
    if not camera_gen.exists():
        raise FileNotFoundError(f"Camera generator not found: {camera_gen}")

    # Decide camera config path and whether generation is needed.
    should_generate_camera = False
    camera_config_path: Path

    if camera_config_input is not None:
        camera_config_path = camera_config_input
        if not args.dry_run and not camera_config_path.exists():
            raise FileNotFoundError(
                f"--camera-config-json was provided, but file was not found: {camera_config_path}"
            )
        print(f"Using existing camera config: {camera_config_path}")
    else:
        camera_config_path = views_json
        if args.regenerate_camera_config:
            should_generate_camera = True
        elif args.skip_camera_generation:
            if not args.dry_run and not camera_config_path.exists():
                raise FileNotFoundError(
                    f"--skip-camera-generation was used, but camera config was not found: {camera_config_path}"
                )
        elif not camera_config_path.exists():
            should_generate_camera = True

    if should_generate_camera:
        camera_config_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(camera_gen),
            "--output",
            str(camera_config_path),
            "--num-cams",
            str(args.num_cams),
            "--radius",
            str(args.radius),
            "--height",
            str(args.height),
            "--target",
            str(args.target[0]),
            str(args.target[1]),
            str(args.target[2]),
            "--width",
            str(args.width),
            "--height-px",
            str(args.height_px),
            "--near",
            str(args.near),
            "--far",
            str(args.far),
            "--fov-y-deg",
            str(args.fov_y_deg),
        ]
        run_command(cmd, dry_run=args.dry_run)
        print(f"Generated camera config: {camera_config_path}")
    else:
        print(f"Using camera config without regeneration: {camera_config_path}")

    camera_cfg_info = describe_camera_config(camera_config_path)
    cam_count = camera_cfg_info.get("num_cameras")
    cam0_dist = camera_cfg_info.get("cam0_distance_to_target")
    cam0_dist_origin = camera_cfg_info.get("cam0_distance_origin")
    if cam_count is not None:
        print(
            f"Camera config stats: num_cameras={cam_count}, "
            f"cam0_distance_to_target={cam0_dist}, cam0_distance_origin={cam0_dist_origin}"
        )

    urdf_files = discover_urdfs(urdf_dir)
    if not urdf_files:
        raise RuntimeError(f"No URDF files found in: {urdf_dir}")

    if not args.dry_run and not args.skip_renderer_preflight:
        print("Running SAPIEN renderer preflight...")
        preflight_sapien_renderer(dry_run=False)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "urdf_dir": str(urdf_dir),
        "views_json": str(camera_config_path),
        "camera_config_stats": camera_cfg_info,
        "object_view_profile": args.object_view_profile,
        "non_blade_orbit_axis": args.non_blade_orbit_axis,
        "output_root": str(output_root),
        "num_urdfs": len(urdf_files),
        "results": [],
    }

    mesh_search_roots = build_mesh_search_roots(repo_root=repo_root, urdf_dir=urdf_dir)
    view_profile = OBJECT_VIEW_PROFILES.get(args.object_view_profile, {})

    for urdf in urdf_files:
        model_name = safe_model_name(urdf_path=urdf, urdf_root=urdf_dir)
        model_output = output_root / model_name

        item: dict[str, Any] = {
            "urdf": str(urdf),
            "model_name": model_name,
            "output_dir": str(model_output),
            "status": "pending",
        }
        root_pose_override = view_profile.get(model_name)
        root_pose_source = "object_view_profile" if root_pose_override is not None else None
        if args.non_blade_orbit_axis != "keep" and model_name != "blade_103706":
            axis_pose = NON_BLADE_ORBIT_AXIS_ROOT_POSE[args.non_blade_orbit_axis]
            root_pose_override = axis_pose
            root_pose_source = f"non_blade_orbit_axis:{args.non_blade_orbit_axis}"
        if root_pose_override is not None:
            item["root_pose_override"] = root_pose_override
            item["root_pose_override_source"] = root_pose_source

        try:
            with tempfile.TemporaryDirectory(prefix="run_pipeline_urdf_") as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                runtime_urdf, remapped, forced_colors = prepare_runtime_urdf(
                    urdf,
                    urdf_dir,
                    tmp_dir,
                    force_link_colors=args.force_link_colors,
                    force_color_seed=args.force_color_seed,
                    extra_search_roots=mesh_search_roots,
                )
                item["urdf_runtime"] = str(runtime_urdf)
                item["mesh_paths_remapped"] = remapped
                item["forced_visual_colors"] = forced_colors

                list_cmd = [
                    sys.executable,
                    str(renderer_cli),
                    "list-joints",
                    "--urdf",
                    str(runtime_urdf),
                    "--as-json",
                ]
                list_proc = run_command(list_cmd, dry_run=args.dry_run)

                if args.dry_run:
                    item["joint_name"] = "<auto-first-1dof>"
                    item["q_start"] = "<auto>"
                    item["q_end"] = "<auto>"
                    item["status"] = "ok"
                else:
                    assert list_proc is not None
                    joints = parse_joints_from_cli_output(list_proc.stdout)

                    try:
                        joint_name, q_start, q_end = choose_joint_and_range(joints)
                    except RuntimeError as exc:
                        if "No 1-DoF active joints found" in str(exc):
                            item["status"] = "skipped_no_1dof"
                            item["error"] = str(exc)
                            print(f"[SKIPPED no_1dof] {urdf.name}: {exc}")
                            summary["results"].append(item)
                            continue
                        raise

                    item["joint_name"] = joint_name
                    item["q_start"] = q_start
                    item["q_end"] = q_end

                    render_cmd = [
                        sys.executable,
                        str(renderer_cli),
                        "render",
                        "--urdf",
                        str(runtime_urdf),
                        "--joint-name",
                        joint_name,
                        "--q-start",
                        str(q_start),
                        "--q-end",
                        str(q_end),
                        "--num-frames",
                        str(args.num_frames),
                        "--views-json",
                        str(camera_config_path),
                        "--output-dir",
                        str(model_output),
                    ]
                    if root_pose_override is not None:
                        position = root_pose_override.get("position", [0.0, 0.0, 0.0])
                        quat = root_pose_override.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
                        render_cmd.extend(
                            [
                                "--root-position",
                                str(float(position[0])),
                                str(float(position[1])),
                                str(float(position[2])),
                                "--root-quaternion-wxyz",
                                str(float(quat[0])),
                                str(float(quat[1])),
                                str(float(quat[2])),
                                str(float(quat[3])),
                            ]
                        )
                    if args.save_video:
                        render_cmd.append("--save-video")

                    run_command(render_cmd, dry_run=False)
                    item["status"] = "ok"
                    print(f"[OK] {urdf.name} -> {model_output}")

        except Exception as exc:  # noqa: BLE001
            item["status"] = "failed"
            item["error"] = str(exc)
            print(f"[FAILED] {urdf.name}: {exc}")

        summary["results"].append(item)

    if not args.dry_run:
        summary_path = output_root / "pipeline_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary written to: {summary_path}")

    ok_count = sum(1 for r in summary["results"] if r.get("status") == "ok")
    skipped_count = sum(1 for r in summary["results"] if r.get("status") == "skipped_no_1dof")
    fail_count = len(summary["results"]) - ok_count - skipped_count
    print(
        f"\nDone. total={len(summary['results'])}, ok={ok_count}, skipped_no_1dof={skipped_count}, failed={fail_count}"
    )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

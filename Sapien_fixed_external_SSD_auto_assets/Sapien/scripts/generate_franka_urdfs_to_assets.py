#!/usr/bin/env python3
"""
Generate URDF files from Franka xacro robot definitions and rewrite mesh paths.

What it does:
1. Copies meshes from:
   <assets>/franka_description/meshes
   to:
   <assets>/meshes
2. Generates URDF for every *.urdf.xacro under:
   <assets>/franka_description/robots/*
3. Writes URDFs directly in <assets> and rewrites mesh filenames to point to:
   <assets>/meshes
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Franka URDFs into assets and rewrite mesh paths")
    parser.add_argument(
        "--assets-dir",
        default="/home/samuelemara/Joint3DGS/Sapien/assets",
        help="Target assets directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    parser.add_argument(
        "--force-link-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ensure each visual has an effective material color/texture. "
            "Missing visual materials are filled with deterministic colors."
        ),
    )
    parser.add_argument(
        "--force-color-seed",
        default="franka_urdf_gen_v1",
        help="Seed string for deterministic color assignment with --force-link-colors.",
    )
    return parser.parse_args()


def discover_xacro_files(robots_dir: Path) -> list[Path]:
    xacros: list[Path] = []
    for robot_dir in sorted(p for p in robots_dir.iterdir() if p.is_dir()):
        if robot_dir.name == "common":
            continue
        xacros.extend(sorted(robot_dir.rglob("*.urdf.xacro")))
    return xacros


def output_name_for_xacro(xacro_path: Path) -> str:
    name = xacro_path.name
    if name.endswith(".urdf.xacro"):
        return name[: -len(".xacro")]
    return f"{xacro_path.stem}.urdf"


def render_targets_for_xacro(xacro_path: Path) -> list[tuple[str, list[str]]]:
    """
    Return (output_file_name, extra_xacro_args) targets for a given xacro.
    """
    base_name = xacro_path.name
    default_output = output_name_for_xacro(xacro_path)

    if base_name == "fr3_duo_arm.urdf.xacro":
        return [
            ("fr3_duo_arm_left.urdf", ["arm_prefix:=left"]),
            ("fr3_duo_arm_right.urdf", ["arm_prefix:=right"]),
        ]

    return [(default_output, [])]


def make_temp_ament_prefix(franka_description_dir: Path) -> tuple[Path, dict[str, str]]:
    temp_prefix = Path(tempfile.mkdtemp(prefix="franka_ament_prefix_"))
    pkg_index = temp_prefix / "share" / "ament_index" / "resource_index" / "packages"
    pkg_index.mkdir(parents=True, exist_ok=True)
    (pkg_index / "franka_description").write_text("", encoding="utf-8")

    share_dir = temp_prefix / "share"
    share_dir.mkdir(parents=True, exist_ok=True)
    (share_dir / "franka_description").symlink_to(franka_description_dir)

    env = os.environ.copy()
    existing = env.get("AMENT_PREFIX_PATH", "")
    env["AMENT_PREFIX_PATH"] = f"{temp_prefix}:{existing}" if existing else str(temp_prefix)
    return temp_prefix, env


def run_xacro(xacro_file: Path, env: dict[str, str], extra_args: list[str]) -> str:
    cmd = ["xacro", str(xacro_file), *extra_args]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return proc.stdout


def rewrite_mesh_paths(urdf_text: str, new_mesh_root: Path) -> str:
    old_prefix = "package://franka_description/meshes/"
    new_prefix = f"{new_mesh_root.as_posix()}/"
    return urdf_text.replace(old_prefix, new_prefix)


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


def _deterministic_link_rgb(link_name: str, visual_index: int, seed: str) -> tuple[float, float, float]:
    key = f"{seed}|{link_name}|{visual_index}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    hue = int.from_bytes(digest[0:2], "big") / 65535.0
    sat = 0.60 + 0.30 * (digest[2] / 255.0)
    val = 0.70 + 0.25 * (digest[3] / 255.0)
    return colorsys.hsv_to_rgb(hue, sat, val)


def _format_rgba(r: float, g: float, b: float, a: float = 1.0) -> str:
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def force_missing_visual_colors(urdf_text: str, seed: str) -> tuple[str, int]:
    """
    Ensure every visual has an effective style. This is needed for colorless meshes
    (for example OBJ files without mtl/vertex colors) to avoid all-gray renders.
    """
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text, 0

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
                material = ET.SubElement(
                    visual,
                    "material",
                    {"name": f"auto_color_{link_name}_{visual_idx}"},
                )

            if _material_has_visual_style(material):
                continue

            r, g, b = _deterministic_link_rgb(link_name, visual_idx, seed)
            ET.SubElement(material, "color", {"rgba": _format_rgba(r, g, b, 1.0)})
            injected += 1

    if injected == 0:
        return urdf_text, 0

    xml_bytes = ET.tostring(root, encoding="utf-8")
    return '<?xml version="1.0" ?>\n' + xml_bytes.decode("utf-8"), injected


def postprocess_existing_urdfs_for_colors(
    assets_dir: Path,
    generated_paths: set[Path],
    force_link_colors: bool,
    seed: str,
) -> int:
    if not force_link_colors:
        return 0

    injected_total = 0
    for urdf_path in sorted(assets_dir.glob("*.urdf")):
        resolved = urdf_path.resolve()
        if resolved in generated_paths:
            continue

        try:
            original = urdf_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue

        patched, injected = force_missing_visual_colors(original, seed=seed)
        if injected <= 0:
            continue

        urdf_path.write_text(patched, encoding="utf-8")
        injected_total += injected
        print(f"Post-processed: {urdf_path} (forced_visual_colors={injected})")

    return injected_total


def main() -> int:
    args = parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()
    franka_description_dir = assets_dir / "franka_description"
    robots_dir = franka_description_dir / "robots"
    source_meshes = franka_description_dir / "meshes"
    target_meshes = assets_dir / "meshes"

    if not franka_description_dir.exists():
        raise FileNotFoundError(f"Missing franka_description directory: {franka_description_dir}")
    if not robots_dir.exists():
        raise FileNotFoundError(f"Missing robots directory: {robots_dir}")
    if not source_meshes.exists():
        raise FileNotFoundError(f"Missing source meshes directory: {source_meshes}")

    xacro_files = discover_xacro_files(robots_dir)
    if not xacro_files:
        raise RuntimeError(f"No *.urdf.xacro files found under {robots_dir}")

    print(f"Discovered {len(xacro_files)} URDF xacro files")

    if args.dry_run:
        print(f"[dry-run] Would copy meshes: {source_meshes} -> {target_meshes}")
    else:
        target_meshes.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_meshes, target_meshes, dirs_exist_ok=True)
        print(f"Copied meshes to: {target_meshes}")

    temp_prefix, env = make_temp_ament_prefix(franka_description_dir)
    generated: list[Path] = []
    failures: list[str] = []
    forced_color_total = 0

    try:
        for xacro_file in xacro_files:
            for out_name, extra_args in render_targets_for_xacro(xacro_file):
                out_path = assets_dir / out_name
                if args.dry_run:
                    print(f"[dry-run] Would generate: {out_path} from {xacro_file.name} {extra_args}")
                    continue

                try:
                    urdf = run_xacro(xacro_file, env=env, extra_args=extra_args)
                    urdf = rewrite_mesh_paths(urdf, target_meshes)
                    forced_count = 0
                    if args.force_link_colors:
                        urdf, forced_count = force_missing_visual_colors(urdf, seed=args.force_color_seed)
                        forced_color_total += forced_count
                    out_path.write_text(urdf, encoding="utf-8")
                    generated.append(out_path)
                    if args.force_link_colors:
                        print(f"Generated: {out_path} (forced_visual_colors={forced_count})")
                    else:
                        print(f"Generated: {out_path}")
                except subprocess.CalledProcessError as exc:
                    stderr = (exc.stderr or "").strip()
                    failures.append(f"{xacro_file} {extra_args}: {stderr}")
                    print(f"FAILED: {xacro_file} {extra_args}")
                    if stderr:
                        print(stderr)

        if not args.dry_run and args.force_link_colors:
            generated_set = {path.resolve() for path in generated}
            forced_color_total += postprocess_existing_urdfs_for_colors(
                assets_dir=assets_dir,
                generated_paths=generated_set,
                force_link_colors=True,
                seed=args.force_color_seed,
            )

    finally:
        shutil.rmtree(temp_prefix, ignore_errors=True)

    if failures:
        print("\nCompleted with failures:")
        for fail in failures:
            print(f"- {fail}")
        return 1

    if not args.dry_run:
        print("\nDone. Generated URDF files:")
        for path in generated:
            print(f"- {path}")
        if args.force_link_colors:
            print(f"Total injected visual colors: {forced_color_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

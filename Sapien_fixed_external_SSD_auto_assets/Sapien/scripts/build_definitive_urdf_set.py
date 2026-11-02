#!/usr/bin/env python3
"""
Build a definitive URDF set by combining:
- visuals/materials from a color URDF set
- joint definitions from a kinematics-correct URDF set

Designed for the two folders in this repo:
- assets/urdf_collection/urdf_collection (good colors, wrong joint names/types)
- assets/sapien_unified/sapien_unified (correct joints, gray visuals)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge URDF sets into a definitive folder (colors + correct joints)"
    )
    parser.add_argument(
        "--color-urdf-dir",
        default=None,
        help="URDF directory providing visuals/materials (default: assets/urdf_collection/urdf_collection)",
    )
    parser.add_argument(
        "--joint-urdf-dir",
        default=None,
        help="URDF directory providing joint blocks (default: assets/sapien_unified/sapien_unified)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output definitive dataset directory (default: assets/definitive_urdf/definitive_urdf)",
    )
    parser.add_argument(
        "--copy-meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy meshes/ from color URDF directory into output directory.",
    )
    parser.add_argument(
        "--copy-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy metadata JSONs from joint URDF dir (metadata/*.json) and color URDF dir "
            "(*_metadata.json) into output metadata/."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output directory before generating files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files.")
    return parser.parse_args()


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iter_direct_children(parent: ET.Element, local_tag: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local_tag(str(child.tag)) == local_tag]


def discover_urdf_map(urdf_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(urdf_dir.glob("*.urdf")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        mapping[path.stem] = path.resolve()
    return mapping


def _joint_parent_child_key(joint_elem: ET.Element) -> tuple[str, str]:
    parent_link = ""
    child_link = ""
    for child in list(joint_elem):
        tag = _local_tag(str(child.tag))
        if tag == "parent":
            parent_link = (child.attrib.get("link") or "").strip()
        elif tag == "child":
            child_link = (child.attrib.get("link") or "").strip()
    return parent_link, child_link


def replace_joints(
    color_root: ET.Element,
    joint_root: ET.Element,
    stem: str,
) -> tuple[ET.Element, list[dict[str, Any]]]:
    color_joints = _iter_direct_children(color_root, "joint")
    joint_joints = _iter_direct_children(joint_root, "joint")

    if not color_joints:
        raise RuntimeError(f"{stem}: color URDF has no <joint> elements")
    if not joint_joints:
        raise RuntimeError(f"{stem}: joint URDF has no <joint> elements")

    joint_by_key: dict[tuple[str, str], ET.Element] = {}
    for jj in joint_joints:
        key = _joint_parent_child_key(jj)
        if key[0] and key[1]:
            joint_by_key[key] = jj

    replacements: list[dict[str, Any]] = []
    merged_joints: list[ET.Element] = []

    for idx, cj in enumerate(color_joints):
        c_key = _joint_parent_child_key(cj)
        selected: ET.Element | None = None
        strategy = ""

        if c_key in joint_by_key:
            selected = copy.deepcopy(joint_by_key[c_key])
            strategy = "parent_child_match"
        elif len(joint_joints) == 1:
            selected = copy.deepcopy(joint_joints[0])
            strategy = "single_joint_fallback"
        elif idx < len(joint_joints):
            selected = copy.deepcopy(joint_joints[idx])
            strategy = "index_fallback"
        else:
            selected = copy.deepcopy(cj)
            strategy = "kept_color_joint"

        replacements.append(
            {
                "old_joint_name": (cj.attrib.get("name") or "").strip(),
                "old_joint_type": (cj.attrib.get("type") or "").strip(),
                "new_joint_name": (selected.attrib.get("name") or "").strip(),
                "new_joint_type": (selected.attrib.get("type") or "").strip(),
                "parent_child": [c_key[0], c_key[1]],
                "strategy": strategy,
            }
        )
        merged_joints.append(selected)

    children = list(color_root)
    first_joint_index = next(
        (i for i, child in enumerate(children) if _local_tag(str(child.tag)) == "joint"),
        len(children),
    )

    for child in children:
        if _local_tag(str(child.tag)) == "joint":
            color_root.remove(child)

    for offset, joint in enumerate(merged_joints):
        color_root.insert(first_joint_index + offset, joint)

    return color_root, replacements


def copy_meshes(color_dir: Path, output_dir: Path, dry_run: bool) -> None:
    src = color_dir / "meshes"
    if not src.exists():
        raise FileNotFoundError(f"Missing meshes directory in color dataset: {src}")
    dst = output_dir / "meshes"
    if dry_run:
        print(f"[dry-run] Would copy meshes: {src} -> {dst}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Copied meshes: {src} -> {dst}")


def copy_metadata(color_dir: Path, joint_dir: Path, output_dir: Path, dry_run: bool) -> None:
    dst = output_dir / "metadata"
    color_meta = sorted(color_dir.glob("*_metadata.json"))
    joint_meta_dir = joint_dir / "metadata"
    joint_meta = sorted(joint_meta_dir.glob("*.json")) if joint_meta_dir.exists() else []

    if dry_run:
        print(f"[dry-run] Would copy {len(color_meta)} color metadata files to {dst}")
        print(f"[dry-run] Would copy {len(joint_meta)} joint metadata files to {dst}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    for path in color_meta:
        shutil.copy2(path, dst / path.name)
    for path in joint_meta:
        shutil.copy2(path, dst / path.name)

    print(f"Copied metadata files to: {dst} (color={len(color_meta)}, joint={len(joint_meta)})")


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]

    color_dir = (
        Path(args.color_urdf_dir).expanduser().resolve()
        if args.color_urdf_dir
        else (repo_root / "assets" / "urdf_collection" / "urdf_collection").resolve()
    )
    joint_dir = (
        Path(args.joint_urdf_dir).expanduser().resolve()
        if args.joint_urdf_dir
        else (repo_root / "assets" / "sapien_unified" / "sapien_unified").resolve()
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (repo_root / "assets" / "definitive_urdf" / "definitive_urdf").resolve()
    )

    print(f"color_urdf_dir: {color_dir}")
    print(f"joint_urdf_dir: {joint_dir}")
    print(f"output_dir: {output_dir}")

    if not color_dir.exists():
        raise FileNotFoundError(f"Color URDF directory not found: {color_dir}")
    if not joint_dir.exists():
        raise FileNotFoundError(f"Joint URDF directory not found: {joint_dir}")

    color_map = discover_urdf_map(color_dir)
    joint_map = discover_urdf_map(joint_dir)

    stems = sorted(set(color_map.keys()) & set(joint_map.keys()))
    if not stems:
        raise RuntimeError("No common URDF stems found between color and joint folders")

    only_color = sorted(set(color_map.keys()) - set(joint_map.keys()))
    only_joint = sorted(set(joint_map.keys()) - set(color_map.keys()))
    if only_color:
        print(f"[warn] Present only in color set (skipped): {only_color}")
    if only_joint:
        print(f"[warn] Present only in joint set (skipped): {only_joint}")

    if args.clean_output and output_dir.exists() and not args.dry_run:
        shutil.rmtree(output_dir)
        print(f"Removed output directory: {output_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_meshes:
        copy_meshes(color_dir=color_dir, output_dir=output_dir, dry_run=args.dry_run)
    if args.copy_metadata:
        copy_metadata(color_dir=color_dir, joint_dir=joint_dir, output_dir=output_dir, dry_run=args.dry_run)

    summary: dict[str, Any] = {
        "color_urdf_dir": str(color_dir),
        "joint_urdf_dir": str(joint_dir),
        "output_dir": str(output_dir),
        "num_models": len(stems),
        "results": [],
    }

    for stem in stems:
        color_path = color_map[stem]
        joint_path = joint_map[stem]
        out_path = output_dir / f"{stem}.urdf"

        color_root = ET.parse(color_path).getroot()
        joint_root = ET.parse(joint_path).getroot()
        merged_root, replacements = replace_joints(color_root=color_root, joint_root=joint_root, stem=stem)

        if not args.dry_run:
            tree = ET.ElementTree(merged_root)
            tree.write(out_path, encoding="utf-8", xml_declaration=True)

        result = {
            "model": stem,
            "color_urdf": str(color_path),
            "joint_urdf": str(joint_path),
            "output_urdf": str(out_path),
            "joint_replacements": replacements,
        }
        summary["results"].append(result)
        new_joint_names = [r["new_joint_name"] for r in replacements]
        print(f"Merged {stem}: joints={new_joint_names}")

    summary_path = output_dir / "merge_summary.json"
    if args.dry_run:
        print(f"[dry-run] Would write summary: {summary_path}")
    else:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {summary_path}")

    print(f"Done. merged_models={len(stems)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

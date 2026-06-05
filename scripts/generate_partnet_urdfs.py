#!/usr/bin/env python3
"""
Generate definitive 2-link URDFs using PartNet-Mobility-consistent kinematics and meshes.

Core behavior:
1) Select a model set.
   - --paris: discover models from PARIS folder (<category>/<id>) and read joint_id from trans.json.
   - --subset-file: optional filter (for --paris) or explicit model list (without --paris).
2) For each model, load PartNet mobility.urdf and select the articulation joint.
3) Split original PartNet mesh links into:
   - moving group: selected joint child-link subtree
   - fixed group: all other links that have visuals
4) Build a 2-link URDF:
   - base_link visuals from fixed group meshes
   - moving_link visuals from moving group meshes
   - single joint <stem>_joint with PartNet type/origin/axis/limits
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    stem: str
    category: str
    object_id: str
    joint_id: int | None


@dataclass
class VisualSpec:
    mesh_ref: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    mesh_scale: str | None


@dataclass
class ResolvedVisual:
    source_link: str
    mesh_path: Path
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    mesh_scale: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PartNet-consistent URDFs (2-link: fixed+moving)"
    )
    parser.add_argument(
        "--partnet-root",
        default="assets/partnet-mobility-v0/dataset",
        help="PartNet-Mobility root (<id>/mobility.urdf, textured_objs, ...)",
    )
    parser.add_argument(
        "--source-root",
        default="assets/PARIS/dataset/data/sapien",
        help="PARIS source root (<category>/<id>/textured_objs/trans.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/definitive_urdf/definitive_urdf",
        help="Output URDF directory",
    )
    parser.add_argument(
        "--paris",
        action="store_true",
        help="Use only models discovered from PARIS source root.",
    )
    parser.add_argument(
        "--subset-file",
        default=None,
        help=(
            "Optional .txt/.json subset list. "
            "With --paris: acts as filter. "
            "Without --paris: defines explicit models."
        ),
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help=(
            "Optional output manifest path for resolved model list. "
            "Default: <output-dir>/partnet_urdf_manifest.json"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help=(
            "Optional generation summary output path. "
            "Default: <output-dir>/partnet_urdf_generation_summary.json"
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing *.urdf files in output dir before writing new ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    return parser.parse_args()


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_child_with_tag(parent: ET.Element, local_tag: str) -> ET.Element | None:
    for child in list(parent):
        if _local_tag(str(child.tag)) == local_tag:
            return child
    return None


def parse_vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if text is None:
        return default
    fields = [f for f in text.strip().split() if f]
    if len(fields) < 3:
        return default
    try:
        return (float(fields[0]), float(fields[1]), float(fields[2]))
    except ValueError:
        return default


def fmt_float(value: float) -> str:
    x = 0.0 if abs(value) < 1e-12 else value
    text = f"{x:.12g}"
    return "0" if text == "-0" else text


def fmt_vec3(values: tuple[float, float, float]) -> str:
    return " ".join(fmt_float(v) for v in values)


def identity_tf() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
            )
    return out


def pose_from_xyz_rpy(
    xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cx = math.cos(roll)
    sx = math.sin(roll)
    cy = math.cos(pitch)
    sy = math.sin(pitch)
    cz = math.cos(yaw)
    sz = math.sin(yaw)

    # URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    x, y, z = xyz
    return [
        [r00, r01, r02, x],
        [r10, r11, r12, y],
        [r20, r21, r22, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def inverse_pose(tf: list[list[float]]) -> list[list[float]]:
    r00, r01, r02, tx = tf[0]
    r10, r11, r12, ty = tf[1]
    r20, r21, r22, tz = tf[2]

    # Inverse rigid transform: [R^T, -R^T t]
    itx = -(r00 * tx + r10 * ty + r20 * tz)
    ity = -(r01 * tx + r11 * ty + r21 * tz)
    itz = -(r02 * tx + r12 * ty + r22 * tz)

    return [
        [r00, r10, r20, itx],
        [r01, r11, r21, ity],
        [r02, r12, r22, itz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def xyz_rpy_from_pose(tf: list[list[float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x, y, z = tf[0][3], tf[1][3], tf[2][3]
    r00, r01, r02 = tf[0][0], tf[0][1], tf[0][2]
    r10, r11, r12 = tf[1][0], tf[1][1], tf[1][2]
    r20, r21, r22 = tf[2][0], tf[2][1], tf[2][2]

    # For R = Rz(yaw) * Ry(pitch) * Rx(roll)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        # Gimbal lock fallback: keep roll=0 and solve yaw from remaining terms.
        roll = 0.0
        yaw = math.atan2(-r01, r11)

    return (x, y, z), (roll, pitch, yaw)


def compute_link_transforms_at_zero(
    all_links: set[str], joints: list[dict[str, Any]]
) -> dict[str, list[list[float]]]:
    parent_joint_by_child: dict[str, tuple[str, dict[str, Any]]] = {}
    for joint in joints:
        parent_link = str(joint.get("parent_link") or "").strip()
        child_link = str(joint.get("child_link") or "").strip()
        if not parent_link or not child_link:
            continue
        parent_joint_by_child[child_link] = (parent_link, joint)

    memo: dict[str, list[list[float]]] = {}
    visiting: set[str] = set()

    def resolve(link_name: str) -> list[list[float]]:
        if link_name in memo:
            return memo[link_name]
        if link_name in visiting:
            raise RuntimeError(f"Cycle detected in kinematic graph around link '{link_name}'")
        visiting.add(link_name)
        parent_info = parent_joint_by_child.get(link_name)
        if parent_info is None:
            tf = identity_tf()
        else:
            parent_link, joint = parent_info
            parent_tf = resolve(parent_link)
            joint_tf = pose_from_xyz_rpy(
                tuple(joint.get("origin_xyz_tuple") or (0.0, 0.0, 0.0)),
                tuple(joint.get("origin_rpy_tuple") or (0.0, 0.0, 0.0)),
            )
            tf = matmul4(parent_tf, joint_tf)
        visiting.remove(link_name)
        memo[link_name] = tf
        return tf

    for link_name in sorted(all_links):
        resolve(link_name)
    return memo


def parse_trans_joint_id(paris_textured_dir: Path) -> int:
    trans_path = paris_textured_dir / "trans.json"
    payload = json.loads(trans_path.read_text(encoding="utf-8"))
    value = payload.get("input", {}).get("joint_id")
    if value is None:
        raise RuntimeError(f"Missing input.joint_id in {trans_path}")
    return int(value)


def parse_mobility_urdf(
    mobility_urdf_path: Path,
) -> tuple[dict[str, list[VisualSpec]], list[dict[str, Any]], dict[str, list[str]]]:
    root = ET.parse(mobility_urdf_path).getroot()

    link_visuals: dict[str, list[VisualSpec]] = {}
    for link in root.iter():
        if _local_tag(str(link.tag)) != "link":
            continue
        link_name = (link.attrib.get("name") or "").strip()
        if not link_name:
            continue

        visuals: list[VisualSpec] = []
        for visual in list(link):
            if _local_tag(str(visual.tag)) != "visual":
                continue
            origin_elem = _first_child_with_tag(visual, "origin")
            origin_xyz = parse_vec3(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
            origin_rpy = parse_vec3(origin_elem.attrib.get("rpy") if origin_elem is not None else None)
            geometry = _first_child_with_tag(visual, "geometry")
            if geometry is None:
                continue
            mesh = _first_child_with_tag(geometry, "mesh")
            if mesh is None:
                continue
            mesh_ref = (mesh.attrib.get("filename") or mesh.attrib.get("url") or "").strip()
            if mesh_ref:
                visuals.append(
                    VisualSpec(
                        mesh_ref=mesh_ref,
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        mesh_scale=(mesh.attrib.get("scale") or None),
                    )
                )
        link_visuals[link_name] = visuals

    joints: list[dict[str, Any]] = []
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.iter():
        if _local_tag(str(joint.tag)) != "joint":
            continue

        parent_elem = _first_child_with_tag(joint, "parent")
        child_elem = _first_child_with_tag(joint, "child")
        origin_elem = _first_child_with_tag(joint, "origin")
        axis_elem = _first_child_with_tag(joint, "axis")
        limit_elem = _first_child_with_tag(joint, "limit")

        parent_link = (parent_elem.attrib.get("link") if parent_elem is not None else None) or None
        child_link = (child_elem.attrib.get("link") if child_elem is not None else None) or None
        origin_xyz_tuple = parse_vec3(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
        origin_rpy_tuple = parse_vec3(origin_elem.attrib.get("rpy") if origin_elem is not None else None)

        item = {
            "name": (joint.attrib.get("name") or "").strip() or None,
            "type": (joint.attrib.get("type") or "").strip() or None,
            "parent_link": parent_link,
            "child_link": child_link,
            "origin_xyz": (origin_elem.attrib.get("xyz") if origin_elem is not None else None),
            "origin_rpy": (origin_elem.attrib.get("rpy") if origin_elem is not None else None),
            "origin_xyz_tuple": origin_xyz_tuple,
            "origin_rpy_tuple": origin_rpy_tuple,
            "axis_xyz": (axis_elem.attrib.get("xyz") if axis_elem is not None else None),
            "limit_lower": (limit_elem.attrib.get("lower") if limit_elem is not None else None),
            "limit_upper": (limit_elem.attrib.get("upper") if limit_elem is not None else None),
        }
        joints.append(item)

        if parent_link and child_link:
            children_by_parent.setdefault(parent_link, []).append(child_link)

    return link_visuals, joints, children_by_parent


def _joint_numeric_index(joint_name: str | None) -> int | None:
    if not joint_name:
        return None
    m = re.fullmatch(r"joint_(\d+)", joint_name)
    if not m:
        return None
    return int(m.group(1))


def is_active_joint(joint: dict[str, Any]) -> bool:
    return (
        joint.get("type") in {"revolute", "prismatic", "continuous"}
        and bool(joint.get("parent_link"))
        and bool(joint.get("child_link"))
    )


def select_joint(joints: list[dict[str, Any]], requested_joint_id: int | None) -> dict[str, Any]:
    if requested_joint_id is not None:
        wanted_name = f"joint_{requested_joint_id}"
        for joint in joints:
            if joint.get("name") == wanted_name and is_active_joint(joint):
                return joint

        if 0 <= requested_joint_id < len(joints):
            by_index = joints[requested_joint_id]
            if is_active_joint(by_index):
                return by_index

        candidates = [j for j in joints if is_active_joint(j)]
        candidates.sort(
            key=lambda j: (
                10**9 if _joint_numeric_index(j.get("name")) is None else _joint_numeric_index(j.get("name")),
                str(j.get("name") or ""),
            )
        )
        if 0 <= requested_joint_id < len(candidates):
            return candidates[requested_joint_id]
        raise RuntimeError(f"Requested joint_id={requested_joint_id} not found as active articulation joint")

    candidates = [j for j in joints if is_active_joint(j)]
    if not candidates:
        raise RuntimeError("No active kinematic joint candidates found")

    candidates.sort(
        key=lambda j: (
            10**9 if _joint_numeric_index(j.get("name")) is None else _joint_numeric_index(j.get("name")),
            str(j.get("name") or ""),
        )
    )
    return candidates[0]


def collect_descendant_links(root_link: str, children_by_parent: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [root_link]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children_by_parent.get(current, []))
    return seen


def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_mesh_path(partnet_obj_dir: Path, mesh_ref: str) -> Path | None:
    text = mesh_ref.strip()
    if not text:
        return None

    if text.startswith("package://"):
        text = text[len("package://") :]
    if text.startswith("file://"):
        text = text[len("file://") :]

    p = Path(text)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((partnet_obj_dir / p).resolve())
        candidates.append((partnet_obj_dir / "textured_objs" / p.name).resolve())
        candidates.append((partnet_obj_dir / p.name).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def add_visual_mesh(
    link_elem: ET.Element,
    mesh_rel_path: str,
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
    mesh_scale: str | None,
) -> None:
    visual = ET.SubElement(link_elem, "visual")
    ET.SubElement(visual, "origin", {"xyz": fmt_vec3(origin_xyz), "rpy": fmt_vec3(origin_rpy)})
    geometry = ET.SubElement(visual, "geometry")
    mesh_attrib = {"filename": mesh_rel_path}
    if mesh_scale:
        mesh_attrib["scale"] = mesh_scale
    ET.SubElement(geometry, "mesh", mesh_attrib)


def _normalize_subset_token(token: str) -> str:
    text = token.strip().replace("\\", "/")
    if not text:
        return ""
    if "/" in text:
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[-2].lower()}_{parts[-1]}"
    if "_" in text:
        category, object_id = text.rsplit("_", 1)
        return f"{category.lower()}_{object_id}"
    return text


def load_subset_filter(path: Path) -> set[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("models", [])
        out: set[str] = set()
        for item in items:
            if isinstance(item, str):
                key = _normalize_subset_token(item)
                if key:
                    out.add(key)
                continue
            if isinstance(item, dict):
                stem = item.get("stem")
                category = item.get("category")
                object_id = item.get("object_id") or item.get("id")
                if isinstance(stem, str):
                    key = _normalize_subset_token(stem)
                elif isinstance(category, str) and isinstance(object_id, str):
                    key = f"{category.lower()}_{object_id}"
                else:
                    key = ""
                if key:
                    out.add(key)
        return out

    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[,\s]+", line) if p]
        if not parts:
            continue
        key = _normalize_subset_token(parts[0])
        if key:
            out.add(key)
    return out


def discover_paris_models(source_root: Path) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        category = category_dir.name
        for object_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            object_id = object_dir.name
            textured_dir = object_dir / "textured_objs"
            trans_path = textured_dir / "trans.json"
            if not textured_dir.is_dir() or not trans_path.exists():
                continue
            joint_id = parse_trans_joint_id(textured_dir)
            stem = f"{category}_{object_id}"
            models.append(ModelSpec(stem=stem, category=category, object_id=object_id, joint_id=joint_id))
    return models


def _parse_subset_entry(entry: str) -> tuple[str | None, str, int | None]:
    parts = [p for p in re.split(r"[,\s]+", entry.strip()) if p]
    if not parts:
        raise ValueError("Empty subset entry")
    model_token = parts[0].replace("\\", "/")
    joint_id = int(parts[1]) if len(parts) > 1 else None

    if "/" in model_token:
        xs = [p for p in model_token.split("/") if p]
        if len(xs) < 2:
            raise ValueError(f"Invalid subset model token: {model_token}")
        return xs[-2], xs[-1], joint_id
    if "_" in model_token:
        category, object_id = model_token.rsplit("_", 1)
        return category, object_id, joint_id
    return None, model_token, joint_id


def _resolve_category_for_object(source_root: Path, object_id: str, hint: str | None) -> str:
    if hint:
        direct = source_root / hint / object_id
        if direct.exists():
            return hint
        for cat_dir in source_root.iterdir():
            if cat_dir.is_dir() and cat_dir.name.lower() == hint.lower():
                if (cat_dir / object_id).exists():
                    return cat_dir.name

    matches = [cat_dir.name for cat_dir in source_root.iterdir() if cat_dir.is_dir() and (cat_dir / object_id).exists()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"Could not resolve category for object_id={object_id}")
    raise RuntimeError(f"Ambiguous category for object_id={object_id}: {matches}")


def load_explicit_models_from_subset(subset_file: Path, source_root: Path) -> list[ModelSpec]:
    if subset_file.suffix.lower() == ".json":
        payload = json.loads(subset_file.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("models", [])
        raw_entries: list[str] = []
        for item in items:
            if isinstance(item, str):
                raw_entries.append(item)
            elif isinstance(item, dict):
                stem = item.get("stem")
                category = item.get("category")
                object_id = item.get("object_id") or item.get("id")
                joint_id = item.get("joint_id")
                if isinstance(stem, str):
                    raw = stem if joint_id is None else f"{stem} {joint_id}"
                    raw_entries.append(raw)
                elif isinstance(object_id, str):
                    if isinstance(category, str):
                        token = f"{category}/{object_id}"
                    else:
                        token = object_id
                    raw = token if joint_id is None else f"{token} {joint_id}"
                    raw_entries.append(raw)
        lines = raw_entries
    else:
        lines = subset_file.read_text(encoding="utf-8").splitlines()

    models: list[ModelSpec] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        category_hint, object_id, joint_id = _parse_subset_entry(line)
        category = _resolve_category_for_object(source_root=source_root, object_id=object_id, hint=category_hint)
        trans_joint = None
        trans_path = source_root / category / object_id / "textured_objs" / "trans.json"
        if trans_path.exists():
            trans_joint = parse_trans_joint_id(source_root / category / object_id / "textured_objs")
        resolved_joint_id = joint_id if joint_id is not None else trans_joint
        stem = f"{category}_{object_id}"
        models.append(ModelSpec(stem=stem, category=category, object_id=object_id, joint_id=resolved_joint_id))

    return models


def build_two_link_urdf(
    *,
    stem: str,
    out_path: Path,
    selected_joint: dict[str, Any],
    fixed_visuals: list[ResolvedVisual],
    moving_visuals: list[ResolvedVisual],
    dry_run: bool,
) -> dict[str, Any]:
    robot = ET.Element("robot", {"name": stem})
    base_link = ET.SubElement(robot, "link", {"name": "base_link"})
    moving_link = ET.SubElement(robot, "link", {"name": "moving_link"})

    for visual_spec in fixed_visuals:
        rel = os.path.relpath(str(visual_spec.mesh_path), str(out_path.parent)).replace("\\", "/")
        add_visual_mesh(
            base_link,
            rel,
            origin_xyz=visual_spec.origin_xyz,
            origin_rpy=visual_spec.origin_rpy,
            mesh_scale=visual_spec.mesh_scale,
        )
    for visual_spec in moving_visuals:
        rel = os.path.relpath(str(visual_spec.mesh_path), str(out_path.parent)).replace("\\", "/")
        add_visual_mesh(
            moving_link,
            rel,
            origin_xyz=visual_spec.origin_xyz,
            origin_rpy=visual_spec.origin_rpy,
            mesh_scale=visual_spec.mesh_scale,
        )

    joint_type = str(selected_joint.get("type") or "revolute")
    joint = ET.SubElement(robot, "joint", {"name": f"{stem}_joint", "type": joint_type})
    ET.SubElement(joint, "parent", {"link": "base_link"})
    ET.SubElement(joint, "child", {"link": "moving_link"})

    origin_xyz_tuple = tuple(selected_joint.get("origin_xyz_tuple") or (0.0, 0.0, 0.0))
    origin_rpy_tuple = tuple(selected_joint.get("origin_rpy_tuple") or (0.0, 0.0, 0.0))
    ET.SubElement(joint, "origin", {"xyz": fmt_vec3(origin_xyz_tuple), "rpy": fmt_vec3(origin_rpy_tuple)})

    axis_xyz = str(selected_joint.get("axis_xyz") or "").strip()
    if axis_xyz and joint_type not in {"fixed", "floating", "planar"}:
        ET.SubElement(joint, "axis", {"xyz": axis_xyz})

    lower = selected_joint.get("limit_lower")
    upper = selected_joint.get("limit_upper")
    if joint_type in {"revolute", "prismatic"} and (lower is not None or upper is not None):
        attrib = {"effort": "100", "velocity": "10"}
        if lower is not None:
            attrib["lower"] = str(lower)
        if upper is not None:
            attrib["upper"] = str(upper)
        ET.SubElement(joint, "limit", attrib)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(robot)
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ")  # type: ignore[attr-defined]
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    return {
        "output_urdf": str(out_path),
        "joint_name": f"{stem}_joint",
        "joint_type": joint_type,
        "fixed_mesh_count": len(fixed_visuals),
        "moving_mesh_count": len(moving_visuals),
    }


def main() -> int:
    args = parse_args()

    partnet_root = Path(args.partnet_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not partnet_root.exists():
        raise FileNotFoundError(f"PartNet root not found: {partnet_root}")
    if not source_root.exists():
        raise FileNotFoundError(f"PARIS source root not found: {source_root}")

    subset_file = Path(args.subset_file).expanduser().resolve() if args.subset_file else None
    if subset_file is not None and not subset_file.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_file}")

    models: list[ModelSpec]
    if args.paris:
        models = discover_paris_models(source_root=source_root)
        if subset_file is not None:
            keys = load_subset_filter(subset_file)
            models = [m for m in models if f"{m.category.lower()}_{m.object_id}" in keys or m.object_id in keys]
    else:
        if subset_file is None:
            raise RuntimeError("Without --paris, you must provide --subset-file")
        models = load_explicit_models_from_subset(subset_file=subset_file, source_root=source_root)

    if not models:
        raise RuntimeError("No models selected")

    models = sorted(models, key=lambda m: m.stem.lower())

    if args.clean_output and output_dir.exists() and not args.dry_run:
        for urdf_path in output_dir.glob("*.urdf"):
            urdf_path.unlink()

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model in models:
        record: dict[str, Any] = {
            "model": model.stem,
            "category": model.category,
            "object_id": model.object_id,
            "requested_joint_id": model.joint_id,
            "status": "pending",
        }
        try:
            partnet_obj_dir = partnet_root / model.object_id
            mobility_path = partnet_obj_dir / "mobility.urdf"
            if not mobility_path.exists():
                raise FileNotFoundError(f"Missing mobility.urdf: {mobility_path}")

            link_visuals, joints, children_by_parent = parse_mobility_urdf(mobility_path)
            selected_joint = select_joint(joints=joints, requested_joint_id=model.joint_id)
            moving_root = selected_joint.get("child_link")
            if not moving_root:
                raise RuntimeError("Selected joint has no child_link")

            selected_parent = selected_joint.get("parent_link")
            selected_child = selected_joint.get("child_link")
            if not selected_parent or not selected_child:
                raise RuntimeError("Selected joint parent/child is incomplete")

            moving_links = collect_descendant_links(str(moving_root), children_by_parent)
            links_with_visuals = [ln for ln, visuals in link_visuals.items() if visuals]
            fixed_links = [ln for ln in links_with_visuals if ln not in moving_links]

            all_links: set[str] = set(link_visuals.keys())
            for joint in joints:
                parent_link = joint.get("parent_link")
                child_link = joint.get("child_link")
                if parent_link:
                    all_links.add(str(parent_link))
                if child_link:
                    all_links.add(str(child_link))

            link_world_tf = compute_link_transforms_at_zero(all_links=all_links, joints=joints)
            if str(selected_parent) not in link_world_tf:
                raise RuntimeError(f"Missing transform for selected parent link: {selected_parent}")
            if str(selected_child) not in link_world_tf:
                raise RuntimeError(f"Missing transform for selected child link: {selected_child}")

            parent_inv = inverse_pose(link_world_tf[str(selected_parent)])
            child_inv = inverse_pose(link_world_tf[str(selected_child)])

            unresolved: list[str] = []
            moving_visuals: list[ResolvedVisual] = []
            fixed_visuals: list[ResolvedVisual] = []

            for link_name in sorted(moving_links):
                link_tf = link_world_tf.get(link_name)
                if link_tf is None:
                    unresolved.append(f"moving:{link_name}:<missing_link_tf>")
                    continue
                rel_tf = matmul4(child_inv, link_tf)
                for visual in link_visuals.get(link_name, []):
                    mesh_path = resolve_mesh_path(partnet_obj_dir, visual.mesh_ref)
                    if mesh_path is None:
                        unresolved.append(f"moving:{link_name}:{visual.mesh_ref}")
                        continue
                    vis_tf = pose_from_xyz_rpy(visual.origin_xyz, visual.origin_rpy)
                    collapsed_tf = matmul4(rel_tf, vis_tf)
                    out_xyz, out_rpy = xyz_rpy_from_pose(collapsed_tf)
                    moving_visuals.append(
                        ResolvedVisual(
                            source_link=link_name,
                            mesh_path=mesh_path,
                            origin_xyz=out_xyz,
                            origin_rpy=out_rpy,
                            mesh_scale=visual.mesh_scale,
                        )
                    )

            for link_name in fixed_links:
                link_tf = link_world_tf.get(link_name)
                if link_tf is None:
                    unresolved.append(f"fixed:{link_name}:<missing_link_tf>")
                    continue
                rel_tf = matmul4(parent_inv, link_tf)
                for visual in link_visuals.get(link_name, []):
                    mesh_path = resolve_mesh_path(partnet_obj_dir, visual.mesh_ref)
                    if mesh_path is None:
                        unresolved.append(f"fixed:{link_name}:{visual.mesh_ref}")
                        continue
                    vis_tf = pose_from_xyz_rpy(visual.origin_xyz, visual.origin_rpy)
                    collapsed_tf = matmul4(rel_tf, vis_tf)
                    out_xyz, out_rpy = xyz_rpy_from_pose(collapsed_tf)
                    fixed_visuals.append(
                        ResolvedVisual(
                            source_link=link_name,
                            mesh_path=mesh_path,
                            origin_xyz=out_xyz,
                            origin_rpy=out_rpy,
                            mesh_scale=visual.mesh_scale,
                        )
                    )

            if not moving_visuals or not fixed_visuals:
                raise RuntimeError(
                    f"Mesh split invalid (fixed={len(fixed_visuals)}, moving={len(moving_visuals)}). "
                    f"Unresolved refs: {len(unresolved)}"
                )

            out_path = output_dir / f"{model.stem}.urdf"
            build_info = build_two_link_urdf(
                stem=model.stem,
                out_path=out_path,
                selected_joint=selected_joint,
                fixed_visuals=fixed_visuals,
                moving_visuals=moving_visuals,
                dry_run=bool(args.dry_run),
            )

            record.update(
                {
                    "status": "ok",
                    "selected_joint_name": selected_joint.get("name"),
                    "selected_joint_type": selected_joint.get("type"),
                    "selected_joint_child_link": selected_joint.get("child_link"),
                    "moving_links": sorted(moving_links),
                    "fixed_links": fixed_links,
                    "unresolved_mesh_refs": unresolved,
                    **build_info,
                }
            )
            print(
                f"[OK] {model.stem}: joint={record['selected_joint_name']} "
                f"fixed_meshes={record['fixed_mesh_count']} moving_meshes={record['moving_mesh_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"[FAILED] {model.stem}: {exc}")

        results.append(record)

    summary = {
        "partnet_root": str(partnet_root),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "paris_mode": bool(args.paris),
        "subset_file": str(subset_file) if subset_file is not None else None,
        "num_models": len(results),
        "counts": {
            "ok": sum(1 for r in results if r.get("status") == "ok"),
            "error": sum(1 for r in results if r.get("status") == "error"),
        },
        "results": results,
    }

    manifest_out = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else (output_dir / "partnet_urdf_manifest.json").resolve()
    )
    summary_out = (
        Path(args.summary_out).expanduser().resolve()
        if args.summary_out
        else (output_dir / "partnet_urdf_generation_summary.json").resolve()
    )

    manifest = [
        {
            "stem": m.stem,
            "category": m.category,
            "object_id": m.object_id,
            "joint_id": m.joint_id,
        }
        for m in models
    ]

    if args.dry_run:
        print(f"[dry-run] Would write manifest: {manifest_out}")
        print(f"[dry-run] Would write summary: {summary_out}")
    else:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_out}")
        print(f"Wrote summary: {summary_out}")

    return 0 if summary["counts"]["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

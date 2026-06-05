#!/usr/bin/env python3
"""
Validate/fix colors of definitive URDF links against original PartNet-Mobility assets.

Approach per model:
1) Parse URDF and read link visual RGB for base/moving links.
2) Locate matching PARIS source object folder under:
     <source_root>/<category>/<object_id>/textured_objs
3) Read selected articulation joint from PARIS trans.json (joint_id).
4) Load PartNet mobility.urdf and split original meshes into moving/fixed sets
   using the selected joint child-link subtree.
5) Parse original OBJ+MTL material assignments and estimate expected link colors.
6) Compare URDF colors with expected colors and optionally write corrected URDFs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ObjData:
    vertices: np.ndarray
    faces: list[tuple[list[int], str | None]]


@dataclass
class MaterialInfo:
    kd: np.ndarray | None
    map_kd: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check definitive URDF link colors against PartNet-Mobility original parts"
    )
    parser.add_argument(
        "--urdf-dir",
        default="assets/definitive_urdf/definitive_urdf",
        help="Directory containing definitive *.urdf files",
    )
    parser.add_argument(
        "--source-root",
        default="assets/PARIS/dataset/data/sapien",
        help="PARIS source root (<category>/<id>/textured_objs); used for trans.json and id mapping",
    )
    parser.add_argument(
        "--partnet-root",
        default="assets/partnet-mobility-v0/dataset",
        help="PartNet-Mobility dataset root (<object_id>/mobility.urdf, textured_objs, ...)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Max Euclidean RGB distance to pass each link check",
    )
    parser.add_argument(
        "--vertex-eps",
        type=float,
        default=1e-6,
        help="Vertex movement threshold used to classify static vs moving vertices",
    )
    parser.add_argument(
        "--fallback-quant-eps",
        type=float,
        default=1e-5,
        help="Quantization epsilon for fallback vertex matching when start/end vertex counts differ",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit 1 if any model fails or cannot be evaluated",
    )
    parser.add_argument(
        "--apply-fix",
        action="store_true",
        help=(
            "Write corrected URDF colors from PartNet-derived expected base/moving colors. "
            "By default writes copies into --fixed-urdf-dir."
        ),
    )
    parser.add_argument(
        "--fixed-urdf-dir",
        default=None,
        help=(
            "Output directory for corrected URDF files when --apply-fix is enabled. "
            "Default: <urdf-dir-name>_color_fixed beside --urdf-dir."
        ),
    )
    parser.add_argument(
        "--fix-in-place",
        action="store_true",
        help="Overwrite source URDF files in --urdf-dir (use with care).",
    )
    parser.add_argument(
        "--fix-only-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When fixing, process only failed models (default: true).",
    )
    parser.add_argument(
        "--fix-strategy",
        choices=["uniform_link_color", "partnet_visual_meshes"],
        default="uniform_link_color",
        help=(
            "How to correct URDF visuals. "
            "'uniform_link_color' sets one RGB color per link (legacy behavior). "
            "'partnet_visual_meshes' rewrites link visuals with original PartNet mesh groups "
            "so per-part MTL/texture colors are preserved."
        ),
    )
    parser.add_argument(
        "--part-color-method",
        choices=["dominant_material", "area_weighted"],
        default="dominant_material",
        help=(
            "How to convert OBJ+MTL face materials to one RGB per link. "
            "'dominant_material' picks the area-dominant material per part (default). "
            "'area_weighted' computes an area-weighted average across materials."
        ),
    )
    parser.add_argument(
        "--material-color-priority",
        choices=["kd_first", "texture_first"],
        default="kd_first",
        help=(
            "How to resolve a material color from MTL. "
            "'kd_first' uses Kd first and falls back to map_Kd texture mean; "
            "'texture_first' does the opposite."
        ),
    )
    return parser.parse_args()


def parse_urdf_link_colors(urdf_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()

    global_materials: dict[str, np.ndarray] = {}
    for child in list(root):
        if child.tag.endswith("material"):
            name = (child.attrib.get("name") or "").strip()
            if not name:
                continue
            rgba = extract_material_rgba(child)
            if rgba is not None:
                global_materials[name] = rgba[:3]

    colors: dict[str, np.ndarray] = {}
    for link in root.iter():
        if not link.tag.endswith("link"):
            continue
        link_name = (link.attrib.get("name") or "").strip()
        if not link_name:
            continue

        link_color: np.ndarray | None = None
        for visual in list(link):
            if not visual.tag.endswith("visual"):
                continue
            material_elem = first_child_with_tag(visual, "material")
            if material_elem is None:
                continue

            rgba = extract_material_rgba(material_elem)
            if rgba is not None:
                link_color = rgba[:3]
                break

            mat_name = (material_elem.attrib.get("name") or "").strip()
            if mat_name and mat_name in global_materials:
                link_color = global_materials[mat_name]
                break

        if link_color is not None:
            colors[link_name] = link_color

    return colors


def parse_urdf_link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    for link in root.iter():
        if not link.tag.endswith("link"):
            continue
        link_name = (link.attrib.get("name") or "").strip()
        if link_name:
            names.append(link_name)
    return names


def extract_material_rgba(material_elem: ET.Element) -> np.ndarray | None:
    color_elem = first_child_with_tag(material_elem, "color")
    if color_elem is None:
        return None
    rgba_text = color_elem.attrib.get("rgba")
    if not rgba_text:
        return None
    parts = rgba_text.split()
    if len(parts) not in (3, 4):
        return None
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        return None
    if len(vals) == 3:
        vals.append(1.0)
    vals = [max(0.0, min(1.0, v)) for v in vals[:4]]
    return np.asarray(vals, dtype=np.float64)


def first_child_with_tag(parent: ET.Element, local_tag: str) -> ET.Element | None:
    for child in list(parent):
        if child.tag.endswith(local_tag):
            return child
    return None


def parse_obj(path: Path) -> ObjData:
    vertices: list[list[float]] = []
    faces: list[tuple[list[int], str | None]] = []
    current_material: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("usemtl "):
                name = line[len("usemtl ") :].strip()
                current_material = name if name else None
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                idxs: list[int] = []
                for tok in tokens:
                    # token format: v, v/vt, v//vn, v/vt/vn
                    head = tok.split("/")[0]
                    if not head:
                        continue
                    vi = int(head)
                    if vi > 0:
                        idx = vi - 1
                    else:
                        idx = len(vertices) + vi
                    idxs.append(idx)
                if len(idxs) >= 3:
                    faces.append((idxs, current_material))

    if not vertices:
        raise RuntimeError(f"No vertices parsed from OBJ: {path}")

    arr = np.asarray(vertices, dtype=np.float64)
    return ObjData(vertices=arr, faces=faces)


def parse_mtl(path: Path) -> dict[str, MaterialInfo]:
    materials: dict[str, MaterialInfo] = {}
    current: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("newmtl "):
                current = line[len("newmtl ") :].strip() or None
                if current is not None and current not in materials:
                    materials[current] = MaterialInfo(kd=None, map_kd=None)
                continue

            if current is None:
                continue

            if line.startswith("Kd "):
                parts = line.split()
                if len(parts) >= 4:
                    kd = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    materials[current].kd = np.clip(kd, 0.0, 1.0)
                continue

            if line.startswith("map_Kd "):
                tex_rel = line[len("map_Kd ") :].strip()
                materials[current].map_kd = tex_rel or None

    return materials


def quantize_vertex(v: np.ndarray, eps: float) -> tuple[int, int, int]:
    return (
        int(round(float(v[0]) / eps)),
        int(round(float(v[1]) / eps)),
        int(round(float(v[2]) / eps)),
    )


def classify_moved_vertices(
    start_vertices: np.ndarray,
    end_vertices: np.ndarray,
    move_eps: float,
    fallback_quant_eps: float,
) -> tuple[np.ndarray, str]:
    if start_vertices.shape == end_vertices.shape:
        d = np.linalg.norm(start_vertices - end_vertices, axis=1)
        moved = d > move_eps
        return moved, "index_aligned"

    # Fallback: static vertices likely keep exact coordinates and can be found in end set.
    end_set = {quantize_vertex(v, fallback_quant_eps) for v in end_vertices}
    moved = np.ones((start_vertices.shape[0],), dtype=bool)
    for i in range(start_vertices.shape[0]):
        if quantize_vertex(start_vertices[i], fallback_quant_eps) in end_set:
            moved[i] = False
    return moved, "quantized_overlap"


def face_area(vertices: np.ndarray, indices: list[int]) -> float:
    if len(indices) < 3:
        return 0.0
    v0 = vertices[indices[0]]
    area = 0.0
    for i in range(1, len(indices) - 1):
        v1 = vertices[indices[i]]
        v2 = vertices[indices[i + 1]]
        tri = np.cross(v1 - v0, v2 - v0)
        area += 0.5 * float(np.linalg.norm(tri))
    return area


def image_mean_rgb(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray | None:
    if path in cache:
        return cache[path]
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        mean = arr.reshape(-1, 3).mean(axis=0)
    except Exception:
        return None
    cache[path] = mean
    return mean


def material_rgb(
    mat_name: str | None,
    materials: dict[str, MaterialInfo],
    mtl_dir: Path,
    tex_cache: dict[Path, np.ndarray],
    priority: str,
) -> np.ndarray | None:
    if mat_name is None:
        return None
    info = materials.get(mat_name)
    if info is None:
        return None

    def from_kd() -> np.ndarray | None:
        if info.kd is None:
            return None
        # Common MTL convention: textured materials keep Kd near white.
        # In kd_first mode we still prefer map_Kd for this specific case.
        if priority == "kd_first" and info.map_kd and is_nearly_white(info.kd):
            return None
        return info.kd

    def from_texture() -> np.ndarray | None:
        if not info.map_kd:
            return None
        tex_path = (mtl_dir / info.map_kd).resolve()
        return image_mean_rgb(tex_path, tex_cache)

    if priority == "texture_first":
        order = (from_texture, from_kd)
    else:
        order = (from_kd, from_texture)

    for getter in order:
        rgb = getter()
        if rgb is not None:
            return rgb

    return None


def weighted_part_color(
    obj_data: ObjData,
    moved_vertices: np.ndarray,
    materials: dict[str, MaterialInfo],
    mtl_dir: Path,
    tex_cache: dict[Path, np.ndarray],
    part_color_method: str,
    material_color_priority: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    static_mat_area: dict[str | None, float] = {}
    dynamic_mat_area: dict[str | None, float] = {}

    mixed_faces = 0
    static_faces = 0
    dynamic_faces = 0

    for indices, mat_name in obj_data.faces:
        if not indices:
            continue
        moved_count = int(sum(1 for idx in indices if moved_vertices[idx]))
        if moved_count == 0:
            bucket = "static"
            static_faces += 1
        elif moved_count == len(indices):
            bucket = "dynamic"
            dynamic_faces += 1
        else:
            mixed_faces += 1
            # Assign mixed faces by majority moved vertices.
            bucket = "dynamic" if moved_count * 2 >= len(indices) else "static"

        w = face_area(obj_data.vertices, indices)
        if w <= 0.0:
            w = 1.0

        if bucket == "static":
            static_mat_area[mat_name] = static_mat_area.get(mat_name, 0.0) + w
        else:
            dynamic_mat_area[mat_name] = dynamic_mat_area.get(mat_name, 0.0) + w

    def area_weighted_rgb(area_by_mat: dict[str | None, float]) -> tuple[np.ndarray | None, float]:
        if not area_by_mat:
            return None, 0.0
        sum_rgb = np.zeros((3,), dtype=np.float64)
        sum_w = 0.0
        for mat_name, w in area_by_mat.items():
            if w <= 0.0:
                continue
            rgb = material_rgb(
                mat_name=mat_name,
                materials=materials,
                mtl_dir=mtl_dir,
                tex_cache=tex_cache,
                priority=material_color_priority,
            )
            if rgb is None:
                continue
            sum_rgb += rgb * w
            sum_w += w
        if sum_w <= 0.0:
            return None, 0.0
        return (sum_rgb / sum_w), sum_w

    def dominant_material_rgb(
        area_by_mat: dict[str | None, float],
    ) -> tuple[np.ndarray | None, str | None]:
        if not area_by_mat:
            return None, None
        ranked = sorted(area_by_mat.items(), key=lambda kv: kv[1], reverse=True)
        fallback: tuple[np.ndarray, str] | None = None
        for mat_name, _ in ranked:
            rgb = material_rgb(
                mat_name=mat_name,
                materials=materials,
                mtl_dir=mtl_dir,
                tex_cache=tex_cache,
                priority=material_color_priority,
            )
            if rgb is not None:
                if fallback is None and mat_name is not None:
                    fallback = (rgb, mat_name)
                mx = float(np.max(rgb))
                mn = float(np.min(rgb))
                mean = float(np.mean(rgb))
                nearly_white = mean >= 0.92 and (mx - mn) <= 0.12
                if not nearly_white:
                    return rgb, mat_name
        if fallback is not None:
            return fallback[0], fallback[1]
        return None, None

    if part_color_method == "dominant_material":
        static_rgb, static_mat = dominant_material_rgb(static_mat_area)
        dynamic_rgb, dynamic_mat = dominant_material_rgb(dynamic_mat_area)
        static_w = float(sum(static_mat_area.values()))
        dynamic_w = float(sum(dynamic_mat_area.values()))
    else:
        static_rgb, static_w = area_weighted_rgb(static_mat_area)
        dynamic_rgb, dynamic_w = area_weighted_rgb(dynamic_mat_area)
        static_mat = None
        dynamic_mat = None

    def top_materials(area_by_mat: dict[str | None, float], limit: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        total = float(sum(area_by_mat.values()))
        for mat_name, area in sorted(area_by_mat.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
            out.append(
                {
                    "material": mat_name,
                    "area": float(area),
                    "area_ratio": (float(area) / total) if total > 0 else 0.0,
                }
            )
        return out

    debug = {
        "static_faces": static_faces,
        "dynamic_faces": dynamic_faces,
        "mixed_faces": mixed_faces,
        "static_weight": static_w,
        "dynamic_weight": dynamic_w,
        "part_color_method": part_color_method,
        "material_color_priority": material_color_priority,
        "static_material_selected": static_mat,
        "dynamic_material_selected": dynamic_mat,
        "static_top_materials": top_materials(static_mat_area),
        "dynamic_top_materials": top_materials(dynamic_mat_area),
    }
    return static_rgb, dynamic_rgb, debug


def split_stem(stem: str) -> tuple[str, str]:
    if "_" not in stem:
        raise ValueError(f"Cannot split model stem '{stem}' into category/id")
    category, object_id = stem.rsplit("_", 1)
    if not object_id:
        raise ValueError(f"Invalid model stem '{stem}'")
    return category, object_id


def find_source_textured_dir(source_root: Path, stem: str) -> Path:
    category_hint, object_id = split_stem(stem)

    hint_candidates = [
        source_root / category_hint / object_id / "textured_objs",
        source_root / category_hint.lower() / object_id / "textured_objs",
        source_root / category_hint.upper() / object_id / "textured_objs",
    ]
    for c in hint_candidates:
        if c.exists() and c.is_dir():
            return c

    matches: list[Path] = []
    for cat_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        candidate = cat_dir / object_id / "textured_objs"
        if candidate.exists() and candidate.is_dir():
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(f"No source folder found for '{stem}' under {source_root}")

    # Prefer exact case-insensitive category match if present.
    for m in matches:
        cat = m.parent.parent.name
        if cat.lower() == category_hint.lower():
            return m

    return matches[0]


def is_nearly_white(rgb: np.ndarray) -> bool:
    mx = float(np.max(rgb))
    mn = float(np.min(rgb))
    mean = float(np.mean(rgb))
    return mean >= 0.92 and (mx - mn) <= 0.12


def parse_trans_joint_id(paris_textured_dir: Path) -> int:
    trans_path = paris_textured_dir / "trans.json"
    if not trans_path.exists():
        raise FileNotFoundError(f"Missing trans.json: {trans_path}")
    payload = json.loads(trans_path.read_text(encoding="utf-8"))
    joint_id = payload.get("input", {}).get("joint_id")
    if joint_id is None:
        raise RuntimeError(f"trans.json missing input.joint_id: {trans_path}")
    try:
        return int(joint_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Invalid joint_id in {trans_path}: {joint_id!r}") from exc


def parse_mobility_urdf(
    mobility_urdf_path: Path,
) -> tuple[dict[str, list[str]], list[dict[str, str | None]], dict[str, list[str]]]:
    root = ET.parse(mobility_urdf_path).getroot()

    link_meshes: dict[str, list[str]] = {}
    for link in root.iter():
        if not link.tag.endswith("link"):
            continue
        link_name = (link.attrib.get("name") or "").strip()
        if not link_name:
            continue
        mesh_refs: list[str] = []
        for visual in list(link):
            if not visual.tag.endswith("visual"):
                continue
            geometry = first_child_with_tag(visual, "geometry")
            if geometry is None:
                continue
            mesh = first_child_with_tag(geometry, "mesh")
            if mesh is None:
                continue
            filename = (mesh.attrib.get("filename") or "").strip()
            if filename:
                mesh_refs.append(filename)
        link_meshes[link_name] = mesh_refs

    joints: list[dict[str, str | None]] = []
    children_map: dict[str, list[str]] = {}
    for joint in root.iter():
        if not joint.tag.endswith("joint"):
            continue
        parent_elem = first_child_with_tag(joint, "parent")
        child_elem = first_child_with_tag(joint, "child")
        parent_link = (parent_elem.attrib.get("link") if parent_elem is not None else None) or None
        child_link = (child_elem.attrib.get("link") if child_elem is not None else None) or None
        entry = {
            "name": (joint.attrib.get("name") or "").strip() or None,
            "type": (joint.attrib.get("type") or "").strip() or None,
            "parent_link": parent_link,
            "child_link": child_link,
            "origin_xyz": None,
            "origin_rpy": None,
            "axis_xyz": None,
            "limit_lower": None,
            "limit_upper": None,
        }

        origin_elem = first_child_with_tag(joint, "origin")
        if origin_elem is not None:
            entry["origin_xyz"] = (origin_elem.attrib.get("xyz") or "").strip() or None
            entry["origin_rpy"] = (origin_elem.attrib.get("rpy") or "").strip() or None

        axis_elem = first_child_with_tag(joint, "axis")
        if axis_elem is not None:
            entry["axis_xyz"] = (axis_elem.attrib.get("xyz") or "").strip() or None

        limit_elem = first_child_with_tag(joint, "limit")
        if limit_elem is not None:
            entry["limit_lower"] = (limit_elem.attrib.get("lower") or "").strip() or None
            entry["limit_upper"] = (limit_elem.attrib.get("upper") or "").strip() or None

        joints.append(entry)
        if parent_link and child_link:
            children_map.setdefault(parent_link, []).append(child_link)

    return link_meshes, joints, children_map


def pick_mobility_joint(joints: list[dict[str, str | None]], joint_id: int) -> dict[str, str | None]:
    by_name = {str(j.get("name")): j for j in joints if j.get("name")}
    wanted_name = f"joint_{joint_id}"
    if wanted_name in by_name:
        return by_name[wanted_name]
    if 0 <= joint_id < len(joints):
        return joints[joint_id]
    raise RuntimeError(f"Could not find joint_id={joint_id} in mobility.urdf")


def collect_descendant_links(root_link: str, children_map: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    stack = [root_link]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(children_map.get(cur, []))
    return out


def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def resolve_partnet_mesh_path(partnet_obj_dir: Path, mesh_ref: str) -> Path | None:
    mesh_ref = mesh_ref.strip()
    if not mesh_ref:
        return None

    # Common URDF reference forms in PartNet.
    if mesh_ref.startswith("package://"):
        mesh_ref = mesh_ref[len("package://") :]
    if mesh_ref.startswith("file://"):
        mesh_ref = mesh_ref[len("file://") :]

    p = Path(mesh_ref)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((partnet_obj_dir / p).resolve())
        candidates.append((partnet_obj_dir / "textured_objs" / p.name).resolve())
        candidates.append((partnet_obj_dir / p.name).resolve())

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def parse_mtllib_refs(obj_path: Path) -> list[str]:
    refs: list[str] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                # OBJ allows multiple filenames on one mtllib line.
                refs.extend([x for x in line[len("mtllib ") :].split() if x])
    return refs


def load_obj_materials(
    obj_path: Path,
    mtl_cache: dict[Path, tuple[dict[str, MaterialInfo], Path]],
) -> tuple[dict[str, MaterialInfo], Path]:
    if obj_path in mtl_cache:
        return mtl_cache[obj_path]

    candidates: list[Path] = []
    for ref in parse_mtllib_refs(obj_path):
        candidates.append((obj_path.parent / ref).resolve())
    candidates.append(obj_path.with_suffix(".mtl").resolve())

    for c in candidates:
        if c.exists() and c.is_file():
            materials = parse_mtl(c)
            mtl_cache[obj_path] = (materials, c.parent)
            return materials, c.parent

    # No MTL found: return empty map and obj dir (texture lookups will fail safely).
    materials: dict[str, MaterialInfo] = {}
    mtl_cache[obj_path] = (materials, obj_path.parent)
    return materials, obj_path.parent


def mesh_material_area_map(obj_data: ObjData) -> dict[str | None, float]:
    out: dict[str | None, float] = {}
    for indices, mat_name in obj_data.faces:
        if not indices:
            continue
        w = face_area(obj_data.vertices, indices)
        if w <= 0.0:
            w = 1.0
        out[mat_name] = out.get(mat_name, 0.0) + w
    return out


def group_color_from_meshes(
    mesh_paths: list[Path],
    tex_cache: dict[Path, np.ndarray],
    obj_cache: dict[Path, ObjData],
    mtl_cache: dict[Path, tuple[dict[str, MaterialInfo], Path]],
    part_color_method: str,
    material_color_priority: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total_area = 0.0

    for mesh_path in mesh_paths:
        obj_data = obj_cache.get(mesh_path)
        if obj_data is None:
            obj_data = parse_obj(mesh_path)
            obj_cache[mesh_path] = obj_data

        mat_areas = mesh_material_area_map(obj_data)
        materials, mtl_dir = load_obj_materials(mesh_path, mtl_cache)

        for mat_name, area in mat_areas.items():
            if area <= 0.0:
                continue
            rgb = material_rgb(
                mat_name=mat_name,
                materials=materials,
                mtl_dir=mtl_dir,
                tex_cache=tex_cache,
                priority=material_color_priority,
            )
            if rgb is None:
                continue
            total_area += area
            candidates.append(
                {
                    "mesh": str(mesh_path),
                    "material": mat_name,
                    "area": float(area),
                    "rgb": rgb,
                }
            )

    if not candidates or total_area <= 0.0:
        return None, {"num_candidates": 0, "top_contributors": []}

    selected_rgb: np.ndarray | None = None
    selected_mat: str | None = None
    selected_mesh: str | None = None

    if part_color_method == "dominant_material":
        ranked = sorted(candidates, key=lambda x: float(x["area"]), reverse=True)
        fallback = ranked[0]
        for item in ranked:
            rgb = item["rgb"]
            if not is_nearly_white(rgb):
                selected_rgb = rgb
                selected_mat = item["material"]
                selected_mesh = item["mesh"]
                break
        if selected_rgb is None:
            selected_rgb = fallback["rgb"]
            selected_mat = fallback["material"]
            selected_mesh = fallback["mesh"]
    else:
        sum_rgb = np.zeros((3,), dtype=np.float64)
        sum_w = 0.0
        for item in candidates:
            w = float(item["area"])
            sum_rgb += item["rgb"] * w
            sum_w += w
        if sum_w > 0.0:
            selected_rgb = sum_rgb / sum_w

    top_contributors: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: float(x["area"]), reverse=True)[:8]:
        top_contributors.append(
            {
                "mesh": item["mesh"],
                "material": item["material"],
                "area": float(item["area"]),
                "area_ratio": float(item["area"]) / total_area,
                "rgb": item["rgb"].tolist(),
            }
        )

    debug = {
        "num_candidates": len(candidates),
        "total_area": total_area,
        "part_color_method": part_color_method,
        "material_color_priority": material_color_priority,
        "selected_mesh": selected_mesh,
        "selected_material": selected_mat,
        "top_contributors": top_contributors,
    }
    return selected_rgb, debug


def pick_urdf_links(link_colors: dict[str, np.ndarray]) -> tuple[str, str]:
    # expected names in definitive set
    if "base_link" in link_colors and "moving_link" in link_colors:
        return "base_link", "moving_link"

    names = sorted(link_colors.keys())
    if len(names) < 2:
        raise RuntimeError(f"Expected at least 2 colored links, found {len(names)}")

    moving_candidates = [n for n in names if "moving" in n.lower()]
    if moving_candidates:
        moving = moving_candidates[0]
        base = next((n for n in names if n != moving), names[0])
        return base, moving

    return names[0], names[1]


def pick_urdf_links_from_names(link_names: list[str]) -> tuple[str, str]:
    names = sorted(set(n for n in link_names if n))
    if "base_link" in names and "moving_link" in names:
        return "base_link", "moving_link"

    if len(names) < 2:
        raise RuntimeError(f"Expected at least 2 links, found {len(names)}")

    moving_candidates = [n for n in names if "moving" in n.lower()]
    if moving_candidates:
        moving = moving_candidates[0]
        base = next((n for n in names if n != moving), names[0])
        return base, moving

    return names[0], names[1]


def color_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def fmt_rgb(rgb: np.ndarray | None) -> str:
    if rgb is None:
        return "None"
    return "[" + ", ".join(f"{float(v):.4f}" for v in rgb.tolist()) + "]"


def evaluate_model(
    urdf_path: Path,
    source_root: Path,
    partnet_root: Path,
    tol: float,
    tex_cache: dict[Path, np.ndarray],
    obj_cache: dict[Path, ObjData],
    mtl_cache: dict[Path, tuple[dict[str, MaterialInfo], Path]],
    part_color_method: str,
    material_color_priority: str,
) -> dict[str, Any]:
    stem = urdf_path.stem
    out: dict[str, Any] = {
        "model": stem,
        "urdf": str(urdf_path),
        "status": "unknown",
    }

    link_colors = parse_urdf_link_colors(urdf_path)
    urdf_base: np.ndarray | None = None
    urdf_moving: np.ndarray | None = None

    if link_colors:
        try:
            base_link, moving_link = pick_urdf_links(link_colors)
        except Exception as exc:  # noqa: BLE001
            out["status"] = "error"
            out["error"] = str(exc)
            out["link_colors"] = {k: link_colors[k].tolist() for k in sorted(link_colors)}
            return out
        urdf_base = link_colors[base_link]
        urdf_moving = link_colors[moving_link]
    else:
        try:
            link_names = parse_urdf_link_names(urdf_path)
            base_link, moving_link = pick_urdf_links_from_names(link_names)
        except Exception as exc:  # noqa: BLE001
            out["status"] = "error"
            out["error"] = f"No link colors found in URDF, and could not resolve link names: {exc}"
            return out

    out["links"] = {
        "base_link_name": base_link,
        "moving_link_name": moving_link,
        "urdf_base_rgb": urdf_base.tolist() if urdf_base is not None else None,
        "urdf_moving_rgb": urdf_moving.tolist() if urdf_moving is not None else None,
        "urdf_has_explicit_link_colors": bool(link_colors),
    }

    try:
        source_dir = find_source_textured_dir(source_root, stem)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(exc)
        return out

    try:
        _category, object_id = split_stem(stem)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(exc)
        out["source_dir"] = str(source_dir)
        return out

    partnet_obj_dir = (partnet_root / object_id).resolve()
    mobility_urdf = partnet_obj_dir / "mobility.urdf"
    if not mobility_urdf.exists():
        out["status"] = "error"
        out["error"] = f"Missing PartNet mobility.urdf: {mobility_urdf}"
        out["source_dir"] = str(source_dir)
        out["partnet_obj_dir"] = str(partnet_obj_dir)
        return out

    try:
        selected_joint_id = parse_trans_joint_id(source_dir)
        link_meshes, joints, children_map = parse_mobility_urdf(mobility_urdf)
        selected_joint = pick_mobility_joint(joints, selected_joint_id)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(exc)
        out["source_dir"] = str(source_dir)
        out["partnet_obj_dir"] = str(partnet_obj_dir)
        return out

    moving_root_link = selected_joint.get("child_link")
    if not moving_root_link:
        out["status"] = "error"
        out["error"] = f"Selected joint has no child link: {selected_joint}"
        out["source_dir"] = str(source_dir)
        out["partnet_obj_dir"] = str(partnet_obj_dir)
        return out

    moving_links = collect_descendant_links(str(moving_root_link), children_map)
    links_with_mesh = [ln for ln, refs in link_meshes.items() if refs]
    fixed_links = [ln for ln in links_with_mesh if ln not in moving_links]

    moving_mesh_refs: list[str] = []
    for ln in sorted(moving_links):
        moving_mesh_refs.extend(link_meshes.get(ln, []))
    fixed_mesh_refs: list[str] = []
    for ln in fixed_links:
        fixed_mesh_refs.extend(link_meshes.get(ln, []))

    moving_mesh_refs = dedupe_keep_order(moving_mesh_refs)
    fixed_mesh_refs = dedupe_keep_order(fixed_mesh_refs)

    moving_mesh_paths: list[Path] = []
    fixed_mesh_paths: list[Path] = []
    unresolved_refs: list[str] = []

    for ref in moving_mesh_refs:
        p = resolve_partnet_mesh_path(partnet_obj_dir, ref)
        if p is None:
            unresolved_refs.append(f"moving:{ref}")
            continue
        moving_mesh_paths.append(p)
    for ref in fixed_mesh_refs:
        p = resolve_partnet_mesh_path(partnet_obj_dir, ref)
        if p is None:
            unresolved_refs.append(f"fixed:{ref}")
            continue
        fixed_mesh_paths.append(p)

    base_rgb, fixed_debug = group_color_from_meshes(
        mesh_paths=fixed_mesh_paths,
        tex_cache=tex_cache,
        obj_cache=obj_cache,
        mtl_cache=mtl_cache,
        part_color_method=part_color_method,
        material_color_priority=material_color_priority,
    )
    moving_rgb, moving_debug = group_color_from_meshes(
        mesh_paths=moving_mesh_paths,
        tex_cache=tex_cache,
        obj_cache=obj_cache,
        mtl_cache=mtl_cache,
        part_color_method=part_color_method,
        material_color_priority=material_color_priority,
    )

    out["source_dir"] = str(source_dir)
    out["partnet_obj_dir"] = str(partnet_obj_dir)
    out["selected_joint_id"] = selected_joint_id
    out["selected_joint"] = selected_joint
    out["moving_root_link"] = moving_root_link
    out["moving_links"] = sorted(moving_links)
    out["fixed_links"] = fixed_links
    out["mesh_refs"] = {
        "moving": moving_mesh_refs,
        "fixed": fixed_mesh_refs,
    }
    out["mesh_paths"] = {
        "moving": [str(p) for p in moving_mesh_paths],
        "fixed": [str(p) for p in fixed_mesh_paths],
    }
    out["unresolved_mesh_refs"] = unresolved_refs
    out["face_stats"] = {
        "fixed": fixed_debug,
        "moving": moving_debug,
    }
    out["expected"] = {
        "base_rgb": base_rgb.tolist() if base_rgb is not None else None,
        "moving_rgb": moving_rgb.tolist() if moving_rgb is not None else None,
    }

    if base_rgb is None or moving_rgb is None:
        out["status"] = "inconclusive"
        out["error"] = "Could not derive both expected base/moving colors from PartNet source"
        return out

    if urdf_base is None or urdf_moving is None:
        out["status"] = "inconclusive"
        out["error"] = "No explicit link colors in URDF (expected colors were derived successfully)"
        return out

    d_base = color_distance(urdf_base, base_rgb)
    d_moving = color_distance(urdf_moving, moving_rgb)
    d_base_swapped = color_distance(urdf_base, moving_rgb)
    d_moving_swapped = color_distance(urdf_moving, base_rgb)

    direct_sum = d_base + d_moving
    swapped_sum = d_base_swapped + d_moving_swapped

    out["distance"] = {
        "base": d_base,
        "moving": d_moving,
        "base_vs_dynamic_swapped": d_base_swapped,
        "moving_vs_static_swapped": d_moving_swapped,
        "direct_sum": direct_sum,
        "swapped_sum": swapped_sum,
    }

    pass_direct = d_base <= tol and d_moving <= tol

    if pass_direct:
        out["status"] = "pass"
    else:
        out["status"] = "fail"
        if swapped_sum + 1e-9 < direct_sum:
            out["hint"] = "Link colors appear closer if base/moving are swapped"

    return out


def print_report(results: list[dict[str, Any]], tol: float) -> None:
    print(f"Tolerance: {tol:.4f}")
    print("model\tstatus\tbase_dist\tmoving_dist\tnotes")
    for item in results:
        model = item.get("model", "?")
        status = item.get("status", "?")
        dist = item.get("distance") or {}
        db = dist.get("base")
        dm = dist.get("moving")
        db_str = f"{db:.4f}" if isinstance(db, (int, float)) else "-"
        dm_str = f"{dm:.4f}" if isinstance(dm, (int, float)) else "-"
        note = ""
        if item.get("hint"):
            note = str(item["hint"])
        elif item.get("error"):
            note = str(item["error"])
        print(f"{model}\t{status}\t{db_str}\t{dm_str}\t{note}")


def _iter_children_with_tag(parent: ET.Element, local_tag: str) -> list[ET.Element]:
    return [child for child in list(parent) if child.tag.endswith(local_tag)]


def _remove_children_with_tag(parent: ET.Element, local_tag: str) -> int:
    removed = 0
    for child in list(parent):
        if child.tag.endswith(local_tag):
            parent.remove(child)
            removed += 1
    return removed


def _format_rgba(rgb: np.ndarray, alpha: float = 1.0) -> str:
    r, g, b = [max(0.0, min(1.0, float(v))) for v in rgb.tolist()]
    a = max(0.0, min(1.0, float(alpha)))
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def _apply_link_color(link_elem: ET.Element, rgb: np.ndarray, material_name: str) -> int:
    visuals = _iter_children_with_tag(link_elem, "visual")
    if not visuals:
        return 0

    rgba_text = _format_rgba(rgb, alpha=1.0)
    updated = 0
    for idx, visual in enumerate(visuals):
        material = first_child_with_tag(visual, "material")
        if material is None:
            mat_name = material_name if len(visuals) == 1 else f"{material_name}_{idx}"
            material = ET.SubElement(visual, "material", {"name": mat_name})
        elif not (material.attrib.get("name") or "").strip():
            mat_name = material_name if len(visuals) == 1 else f"{material_name}_{idx}"
            material.set("name", mat_name)

        color = first_child_with_tag(material, "color")
        if color is None:
            ET.SubElement(material, "color", {"rgba": rgba_text})
        else:
            color.set("rgba", rgba_text)
        updated += 1
    return updated


def _rewrite_link_visual_meshes(
    link_elem: ET.Element,
    mesh_paths: list[Path],
    urdf_parent_dir: Path,
) -> dict[str, Any]:
    removed = _remove_children_with_tag(link_elem, "visual")
    added = 0
    refs: list[str] = []

    for mesh_path in mesh_paths:
        mesh_abs = mesh_path.expanduser().resolve()
        mesh_rel = os.path.relpath(str(mesh_abs), str(urdf_parent_dir.resolve())).replace("\\", "/")

        visual = ET.SubElement(link_elem, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": mesh_rel})
        refs.append(mesh_rel)
        added += 1

    if added == 0:
        raise RuntimeError(f"No meshes to create visuals for link '{link_elem.attrib.get('name', '?')}'")

    return {
        "visuals_removed": removed,
        "visuals_added": added,
        "mesh_refs": refs,
    }


def _ensure_direct_child(parent: ET.Element, local_tag: str) -> ET.Element:
    child = first_child_with_tag(parent, local_tag)
    if child is None:
        child = ET.SubElement(parent, local_tag)
    return child


def _apply_partnet_joint_kinematics(
    joint_elem: ET.Element,
    *,
    base_link_name: str,
    moving_link_name: str,
    selected_joint: dict[str, Any] | None,
) -> dict[str, Any]:
    if not selected_joint:
        return {"applied": False, "reason": "missing_selected_joint"}

    joint_type = (selected_joint.get("type") or "").strip()
    if joint_type:
        joint_elem.set("type", joint_type)

    parent_elem = _ensure_direct_child(joint_elem, "parent")
    parent_elem.set("link", base_link_name)

    child_elem = _ensure_direct_child(joint_elem, "child")
    child_elem.set("link", moving_link_name)

    origin_xyz = (selected_joint.get("origin_xyz") or "").strip()
    origin_rpy = (selected_joint.get("origin_rpy") or "").strip()
    if origin_xyz or origin_rpy:
        origin_elem = _ensure_direct_child(joint_elem, "origin")
        if origin_xyz:
            origin_elem.set("xyz", origin_xyz)
        if origin_rpy:
            origin_elem.set("rpy", origin_rpy)
        elif "rpy" not in origin_elem.attrib:
            origin_elem.set("rpy", "0 0 0")

    axis_xyz = (selected_joint.get("axis_xyz") or "").strip()
    if axis_xyz and joint_type not in {"fixed", "floating", "planar"}:
        axis_elem = _ensure_direct_child(joint_elem, "axis")
        axis_elem.set("xyz", axis_xyz)

    lower = (selected_joint.get("limit_lower") or "").strip()
    upper = (selected_joint.get("limit_upper") or "").strip()
    if (lower or upper) and joint_type in {"revolute", "prismatic"}:
        limit_elem = _ensure_direct_child(joint_elem, "limit")
        if lower:
            limit_elem.set("lower", lower)
        if upper:
            limit_elem.set("upper", upper)
        if "effort" not in limit_elem.attrib:
            limit_elem.set("effort", "100")
        if "velocity" not in limit_elem.attrib:
            limit_elem.set("velocity", "10")
    elif joint_type == "continuous":
        # Continuous joints in PartNet usually have no finite lower/upper.
        _remove_children_with_tag(joint_elem, "limit")

    return {
        "applied": True,
        "joint_type": joint_type or None,
        "origin_xyz": origin_xyz or None,
        "origin_rpy": origin_rpy or None,
        "axis_xyz": axis_xyz or None,
        "limit_lower": lower or None,
        "limit_upper": upper or None,
    }


def write_corrected_urdf_colors(
    input_urdf: Path,
    output_urdf: Path,
    base_link_name: str,
    moving_link_name: str,
    base_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    *,
    fix_strategy: str = "uniform_link_color",
    base_mesh_paths: list[Path] | None = None,
    moving_mesh_paths: list[Path] | None = None,
    selected_joint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    base_link = None
    moving_link = None
    primary_joint = None
    for link in root.iter():
        if not link.tag.endswith("link"):
            continue
        name = (link.attrib.get("name") or "").strip()
        if name == base_link_name:
            base_link = link
        elif name == moving_link_name:
            moving_link = link

    for elem in root.iter():
        if elem.tag.endswith("joint"):
            primary_joint = elem
            break

    if base_link is None or moving_link is None:
        raise RuntimeError(
            f"Could not find target links in URDF: base='{base_link_name}', moving='{moving_link_name}'"
        )

    if fix_strategy == "partnet_visual_meshes":
        if base_mesh_paths is None or moving_mesh_paths is None:
            raise RuntimeError("partnet_visual_meshes strategy requires base/moving mesh path lists")
        base_info = _rewrite_link_visual_meshes(
            link_elem=base_link,
            mesh_paths=base_mesh_paths,
            urdf_parent_dir=output_urdf.parent,
        )
        moving_info = _rewrite_link_visual_meshes(
            link_elem=moving_link,
            mesh_paths=moving_mesh_paths,
            urdf_parent_dir=output_urdf.parent,
        )
        joint_info = None
        if primary_joint is not None:
            joint_info = _apply_partnet_joint_kinematics(
                joint_elem=primary_joint,
                base_link_name=base_link_name,
                moving_link_name=moving_link_name,
                selected_joint=selected_joint,
            )
        base_updated = base_info["visuals_added"]
        moving_updated = moving_info["visuals_added"]
    else:
        base_updated = _apply_link_color(base_link, base_rgb, material_name=f"fixed_color_{base_link_name}")
        moving_updated = _apply_link_color(moving_link, moving_rgb, material_name=f"fixed_color_{moving_link_name}")
        joint_info = None

    if base_updated == 0:
        raise RuntimeError(f"No <visual> entries found for base link '{base_link_name}'")
    if moving_updated == 0:
        raise RuntimeError(f"No <visual> entries found for moving link '{moving_link_name}'")

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    out = {
        "base_visuals_updated": base_updated,
        "moving_visuals_updated": moving_updated,
    }
    if fix_strategy == "partnet_visual_meshes":
        out["strategy"] = "partnet_visual_meshes"
        out["base_mesh_count"] = len(base_mesh_paths or [])
        out["moving_mesh_count"] = len(moving_mesh_paths or [])
        out["joint_kinematics"] = joint_info
    else:
        out["strategy"] = "uniform_link_color"
    return out



def main() -> int:
    args = parse_args()

    urdf_dir = Path(args.urdf_dir).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    partnet_root = Path(args.partnet_root).expanduser().resolve()

    if not urdf_dir.exists():
        raise FileNotFoundError(f"URDF dir not found: {urdf_dir}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not partnet_root.exists():
        raise FileNotFoundError(f"PartNet root not found: {partnet_root}")
    if args.fix_in_place and args.fixed_urdf_dir is not None:
        raise ValueError("Use either --fix-in-place or --fixed-urdf-dir, not both.")

    urdfs = sorted(urdf_dir.glob("*.urdf"))
    if not urdfs:
        raise RuntimeError(f"No URDF files found in: {urdf_dir}")

    tex_cache: dict[Path, np.ndarray] = {}
    obj_cache: dict[Path, ObjData] = {}
    mtl_cache: dict[Path, tuple[dict[str, MaterialInfo], Path]] = {}
    results: list[dict[str, Any]] = []

    for urdf in urdfs:
        try:
            result = evaluate_model(
                urdf_path=urdf,
                source_root=source_root,
                partnet_root=partnet_root,
                tol=float(args.tolerance),
                tex_cache=tex_cache,
                obj_cache=obj_cache,
                mtl_cache=mtl_cache,
                part_color_method=str(args.part_color_method),
                material_color_priority=str(args.material_color_priority),
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "model": urdf.stem,
                "urdf": str(urdf),
                "status": "error",
                "error": f"Unhandled exception: {exc}",
            }
        results.append(result)

    summary = {
        "urdf_dir": str(urdf_dir),
        "source_root": str(source_root),
        "partnet_root": str(partnet_root),
        "tolerance": float(args.tolerance),
        "part_color_method": str(args.part_color_method),
        "material_color_priority": str(args.material_color_priority),
        "num_models": len(results),
        "counts": {
            "pass": sum(1 for r in results if r.get("status") == "pass"),
            "fail": sum(1 for r in results if r.get("status") == "fail"),
            "inconclusive": sum(1 for r in results if r.get("status") == "inconclusive"),
            "error": sum(1 for r in results if r.get("status") == "error"),
            "unknown": sum(1 for r in results if r.get("status") == "unknown"),
        },
        "results": results,
    }

    fixed_out_dir: Path | None = None
    if args.apply_fix and not args.fix_in_place:
        if args.fixed_urdf_dir:
            fixed_out_dir = Path(args.fixed_urdf_dir).expanduser().resolve()
        else:
            fixed_out_dir = (urdf_dir.parent / f"{urdf_dir.name}_color_fixed").resolve()
        fixed_out_dir.mkdir(parents=True, exist_ok=True)

    fix_stats = {
        "requested": bool(args.apply_fix),
        "fix_in_place": bool(args.fix_in_place),
        "fix_only_failed": bool(args.fix_only_failed),
        "fix_strategy": str(args.fix_strategy),
        "fixed_dir": str(fixed_out_dir) if fixed_out_dir is not None else None,
        "written": 0,
        "skipped": 0,
        "errors": 0,
    }

    if args.apply_fix:
        result_by_model = {str(r.get("model")): r for r in results}
        for urdf in urdfs:
            item = result_by_model.get(urdf.stem)
            if item is None:
                fix_stats["skipped"] += 1
                continue

            status = str(item.get("status"))
            allowed_for_fix = status == "fail"
            if str(args.fix_strategy) == "partnet_visual_meshes" and status == "inconclusive":
                allowed_for_fix = True
            if args.fix_only_failed and not allowed_for_fix:
                item["fix_applied"] = False
                item["fix_skip_reason"] = f"status={status}"
                fix_stats["skipped"] += 1
                continue

            expected = item.get("expected") or {}
            links = item.get("links") or {}
            base_rgb_list = expected.get("base_rgb")
            moving_rgb_list = expected.get("moving_rgb")
            base_link_name = links.get("base_link_name")
            moving_link_name = links.get("moving_link_name")

            if (
                base_rgb_list is None
                or moving_rgb_list is None
                or not base_link_name
                or not moving_link_name
            ):
                item["fix_applied"] = False
                item["fix_error"] = "Missing expected colors or link names"
                fix_stats["errors"] += 1
                continue

            base_rgb = np.asarray(base_rgb_list, dtype=np.float64)
            moving_rgb = np.asarray(moving_rgb_list, dtype=np.float64)

            out_urdf = urdf if args.fix_in_place else (fixed_out_dir / urdf.name)  # type: ignore[arg-type]

            mesh_paths = item.get("mesh_paths") or {}
            base_mesh_paths_list = mesh_paths.get("fixed") or []
            moving_mesh_paths_list = mesh_paths.get("moving") or []
            base_mesh_paths = [Path(str(p)).expanduser().resolve() for p in base_mesh_paths_list]
            moving_mesh_paths = [Path(str(p)).expanduser().resolve() for p in moving_mesh_paths_list]
            selected_joint = item.get("selected_joint")

            try:
                patch_info = write_corrected_urdf_colors(
                    input_urdf=urdf,
                    output_urdf=out_urdf,
                    base_link_name=str(base_link_name),
                    moving_link_name=str(moving_link_name),
                    base_rgb=base_rgb,
                    moving_rgb=moving_rgb,
                    fix_strategy=str(args.fix_strategy),
                    base_mesh_paths=base_mesh_paths,
                    moving_mesh_paths=moving_mesh_paths,
                    selected_joint=selected_joint if isinstance(selected_joint, dict) else None,
                )
                item["fix_applied"] = True
                item["fixed_urdf"] = str(out_urdf)
                item["fixed_colors"] = {
                    "base_rgb": base_rgb.tolist(),
                    "moving_rgb": moving_rgb.tolist(),
                }
                item["fix_details"] = patch_info
                fix_stats["written"] += 1
            except Exception as exc:  # noqa: BLE001
                item["fix_applied"] = False
                item["fix_error"] = str(exc)
                fix_stats["errors"] += 1

    summary["fix"] = fix_stats

    print_report(results, tol=float(args.tolerance))
    counts = summary["counts"]
    print(
        "\nSummary: "
        f"pass={counts['pass']} fail={counts['fail']} "
        f"inconclusive={counts['inconclusive']} error={counts['error']}"
    )
    if args.apply_fix:
        print(
            "Fixes: "
            f"written={fix_stats['written']} skipped={fix_stats['skipped']} errors={fix_stats['errors']}"
        )
        if fix_stats["fixed_dir"] is not None:
            print(f"Fixed URDF output dir: {fix_stats['fixed_dir']}")
        elif args.fix_in_place:
            print("Fixed URDFs were written in-place.")

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON report written to: {out_path}")

    if args.fail_on_error and (counts["fail"] > 0 or counts["inconclusive"] > 0 or counts["error"] > 0):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

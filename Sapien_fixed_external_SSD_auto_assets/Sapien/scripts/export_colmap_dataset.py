#!/usr/bin/env python3
"""
Export a COLMAP-style dataset and a part-labeled point cloud for rendered outputs.

The exporter starts from an existing per-model output folder written by
src/render_urdf_multiview.py. It replays the same render schedule to recover
actor-level segmentation, then writes:

- colmap/images/<camera>/<frame>.png
- colmap/sparse/0/cameras.{txt,bin}
- colmap/sparse/0/images.{txt,bin}
- colmap/sparse/0/points3D.{txt,bin}
- pointcloud/points3D_parts.ply with a uchar "part" vertex property

Part labels follow the definitive two-link URDF convention:
0 = static/base link, 1 = moving/child link.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal host envs

    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[no-redef]
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for import_dir in (SCRIPT_DIR, SRC_DIR):
    import_dir_str = str(import_dir)
    if import_dir_str not in sys.path:
        sys.path.insert(0, import_dir_str)

from camera_utils import camera_pose_matrix  # noqa: E402
from config_types import CameraSpec, RenderJobConfig  # noqa: E402
from read_write_model import Camera, Image, Point3D, qvec2rotmat, rotmat2qvec, write_model  # noqa: E402


SAPIEN_CAMERA_TO_COLMAP_CAMERA = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class PartLinkInfo:
    link_order: list[str]
    selected_joint: str | None
    selected_joint_parent: str | None
    selected_joint_child: str | None
    static_links: list[str]
    moving_links: list[str]


@dataclass(slots=True)
class JointMetadata:
    name: str
    type: str
    point_of_application: list[float]
    axis_vector: list[float]
    coordinate_frame: str
    urdf_origin_xyz: list[float]
    urdf_origin_rpy: list[float]
    urdf_axis: list[float]


@dataclass(slots=True)
class ExportStats:
    model_output: str
    urdf_path: str
    colmap_dir: str
    sparse_dir: str
    images_dir: str | None
    pointcloud_ply: str
    num_cameras: int
    num_images: int
    num_points3D: int
    point_part_counts: dict[str, int]
    model_format: str
    image_mode: str
    sample_stride: int
    max_points_per_image: int
    frame_step: int
    max_frames: int | None
    joint_metadata: dict[str, Any]
    part_link_info: dict[str, Any]
    actor_id_to_link_name: dict[str, str]
    actor_id_to_part: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add COLMAP cameras/images/points3D and a part-labeled PLY point cloud "
            "to rendered SAPIEN output folders."
        )
    )
    parser.add_argument(
        "--model-output",
        action="append",
        default=[],
        help=(
            "Per-model output directory to export. Can be passed multiple times. "
            "The directory must contain metadata/job_config_resolved.yaml."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Batch mode: root containing per-model output directories. Each child "
            "with metadata/job_config_resolved.yaml is exported."
        ),
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="URDF override for single-model export. Useful when metadata points to a temp URDF.",
    )
    parser.add_argument(
        "--urdf-dir",
        default=None,
        help=(
            "Batch URDF directory. For each model output, <model_name>.urdf is used "
            "when present before falling back to the metadata URDF path."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="colmap",
        help="COLMAP dataset subdirectory name inside each model output.",
    )
    parser.add_argument(
        "--pointcloud-dir-name",
        default="pointcloud",
        help="Point cloud subdirectory name inside each model output.",
    )
    parser.add_argument(
        "--ply-name",
        default="points3D_parts.ply",
        help="PLY filename written under --pointcloud-dir-name.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=8,
        help="Pixel stride used when sampling rendered depth maps for sparse points.",
    )
    parser.add_argument(
        "--max-points-per-image",
        type=int,
        default=1000,
        help="Maximum sampled 3D points per image. Use 0 for no per-image cap.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to export (0-based). Default 0 exports from frame_000000.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Export every Nth rendered frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1,
        help=(
            "Maximum exported frames per model after frame-step sampling. "
            "Default: 1, so only frame_000000 is exported. Use 0 for all frames."
        ),
    )
    parser.add_argument(
        "--no-final-frame", action="store_true",
        help="Disable the additional independent COLMAP/PLY export at the last rendered frame.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "copy", "none"],
        default="copy",
        help=(
            "How to populate colmap/images (copy works on exFAT). "
            "Missing source RGB files are written "
            "from the replayed render when mode is symlink or copy."
        ),
    )
    parser.add_argument(
        "--model-format",
        choices=["txt", "bin", "both"],
        default="both",
        help="COLMAP sparse model format to write.",
    )
    return parser.parse_args()


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _first_child_with_tag(parent: ET.Element, local_tag: str) -> ET.Element | None:
    for child in list(parent):
        if _local_tag(str(child.tag)) == local_tag:
            return child
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def _remap_common_container_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()

    path_str = str(path)
    for prefix in ("/workspace/",):
        if path_str.startswith(prefix):
            candidate = (REPO_ROOT / path_str[len(prefix) :]).resolve()
            if candidate.exists():
                return candidate

    return path.expanduser().resolve()


def _load_render_config(model_output: Path, urdf_path: Path | None) -> RenderJobConfig:
    cfg_path = model_output / "metadata" / "job_config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing render metadata: {cfg_path}")

    payload = _read_yaml(cfg_path)
    cfg = RenderJobConfig.from_dict(payload)
    cfg.output_dir = str(model_output.resolve())

    resolved_urdf = urdf_path
    if resolved_urdf is None:
        resolved_urdf = _remap_common_container_path(Path(cfg.urdf_path).expanduser())
    if not resolved_urdf.exists():
        raise FileNotFoundError(
            f"URDF for COLMAP export does not exist: {resolved_urdf}. "
            "Pass --urdf for single-model export or --urdf-dir for batch mode."
        )
    cfg.urdf_path = str(resolved_urdf.resolve())
    return cfg


def _discover_model_outputs(args: argparse.Namespace) -> list[Path]:
    outputs: list[Path] = []

    for item in args.model_output:
        outputs.append(Path(item).expanduser().resolve())

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Output root does not exist: {root}")
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "metadata" / "job_config_resolved.yaml").exists():
                outputs.append(child.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for output in outputs:
        if output in seen:
            continue
        seen.add(output)
        deduped.append(output)

    if not deduped:
        raise ValueError("Provide --model-output or --output-root")
    return deduped


def _build_urdf_map(urdf_dir: str | None) -> dict[str, Path]:
    if urdf_dir is None:
        return {}

    root = Path(urdf_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"URDF directory does not exist: {root}")

    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.urdf")):
        if not path.is_file():
            continue
        out.setdefault(path.stem, path.resolve())
    return out


def _resolve_urdf_for_model(
    model_output: Path,
    args: argparse.Namespace,
    urdf_map: dict[str, Path],
) -> Path | None:
    if args.urdf is not None:
        return Path(args.urdf).expanduser().resolve()
    return urdf_map.get(model_output.name)


def parse_part_link_info(urdf_path: Path, joint_name: str) -> PartLinkInfo:
    root = ET.parse(urdf_path).getroot()

    link_order: list[str] = []
    for elem in list(root):
        if _local_tag(str(elem.tag)) != "link":
            continue
        name = (elem.attrib.get("name") or "").strip()
        if name:
            link_order.append(name)

    joint_parent_child: dict[str, tuple[str | None, str | None]] = {}
    children_by_parent: dict[str, list[str]] = {}
    for elem in list(root):
        if _local_tag(str(elem.tag)) != "joint":
            continue
        name = (elem.attrib.get("name") or "").strip()
        parent_elem = _first_child_with_tag(elem, "parent")
        child_elem = _first_child_with_tag(elem, "child")
        parent_link = (
            (parent_elem.attrib.get("link") or "").strip()
            if parent_elem is not None
            else None
        )
        child_link = (
            (child_elem.attrib.get("link") or "").strip()
            if child_elem is not None
            else None
        )
        if name:
            joint_parent_child[name] = (parent_link or None, child_link or None)
        if parent_link and child_link:
            children_by_parent.setdefault(parent_link, []).append(child_link)

    selected_joint = joint_name if joint_name in joint_parent_child else None
    if selected_joint is None and len(joint_parent_child) == 1:
        selected_joint = next(iter(joint_parent_child.keys()))

    selected_parent: str | None = None
    selected_child: str | None = None
    if selected_joint is not None:
        selected_parent, selected_child = joint_parent_child[selected_joint]

    if selected_parent is None and link_order:
        selected_parent = link_order[0]
    if selected_child is None and len(link_order) >= 2:
        selected_child = link_order[1]

    moving_links: set[str] = set()

    def visit_moving_subtree(link_name: str) -> None:
        if link_name in moving_links:
            return
        moving_links.add(link_name)
        for child in children_by_parent.get(link_name, []):
            visit_moving_subtree(child)

    if selected_child:
        visit_moving_subtree(selected_child)

    if not moving_links and len(link_order) >= 2:
        moving_links.add(link_order[1])

    static_links = [link for link in link_order if link not in moving_links]
    if not static_links and selected_parent:
        static_links = [selected_parent]

    return PartLinkInfo(
        link_order=link_order,
        selected_joint=selected_joint,
        selected_joint_parent=selected_parent,
        selected_joint_child=selected_child,
        static_links=static_links,
        moving_links=[link for link in link_order if link in moving_links],
    )


def _parse_vector(text: str | None, default: list[float]) -> list[float]:
    if text is None:
        return list(default)
    parts = text.strip().split()
    if len(parts) != 3:
        return list(default)
    try:
        return [float(v) for v in parts]
    except ValueError:
        return list(default)


def _rpy_to_rotmat(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _root_pose_matrix(cfg: RenderJobConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.root_pose is None:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    quat = np.asarray(cfg.root_pose.quaternion_wxyz, dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 1e-12:
        raise ValueError("root_pose quaternion norm must be > 0")
    quat = quat / quat_norm
    translation = np.asarray(cfg.root_pose.position, dtype=np.float64)
    return qvec2rotmat(quat), translation


def parse_joint_metadata(
    urdf_path: Path, joint_name: str, cfg: RenderJobConfig
) -> JointMetadata:
    root = ET.parse(urdf_path).getroot()

    joint_elem: ET.Element | None = None
    joints: list[ET.Element] = []
    for elem in list(root):
        if _local_tag(str(elem.tag)) != "joint":
            continue
        joints.append(elem)
        if (elem.attrib.get("name") or "").strip() == joint_name:
            joint_elem = elem

    if joint_elem is None and len(joints) == 1:
        joint_elem = joints[0]
    if joint_elem is None:
        raise ValueError(f"Could not find joint '{joint_name}' in URDF: {urdf_path}")

    name = (joint_elem.attrib.get("name") or joint_name).strip()
    joint_type = (joint_elem.attrib.get("type") or "unknown").strip() or "unknown"

    origin_elem = _first_child_with_tag(joint_elem, "origin")
    axis_elem = _first_child_with_tag(joint_elem, "axis")
    origin_xyz = _parse_vector(
        origin_elem.attrib.get("xyz") if origin_elem is not None else None,
        [0.0, 0.0, 0.0],
    )
    origin_rpy = _parse_vector(
        origin_elem.attrib.get("rpy") if origin_elem is not None else None,
        [0.0, 0.0, 0.0],
    )
    axis = _parse_vector(
        axis_elem.attrib.get("xyz") if axis_elem is not None else None, [1.0, 0.0, 0.0]
    )

    root_rot, root_translation = _root_pose_matrix(cfg)
    joint_rot = _rpy_to_rotmat(origin_rpy)
    point_world = root_rot @ np.asarray(origin_xyz, dtype=np.float64) + root_translation
    axis_world = root_rot @ joint_rot @ np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis_world)
    if axis_norm > 1e-12:
        axis_world = axis_world / axis_norm

    return JointMetadata(
        name=name,
        type=joint_type,
        point_of_application=[float(v) for v in point_world.tolist()],
        axis_vector=[float(v) for v in axis_world.tolist()],
        coordinate_frame="render_world",
        urdf_origin_xyz=[float(v) for v in origin_xyz],
        urdf_origin_rpy=[float(v) for v in origin_rpy],
        urdf_axis=[float(v) for v in axis],
    )


def _extract_actor_id_to_link_name(articulation: Any) -> dict[int, str]:
    actor_id_to_link_name: dict[int, str] = {}
    links = list(articulation.get_links()) if hasattr(articulation, "get_links") else []

    for link in links:
        link_name = str(link.get_name()) if hasattr(link, "get_name") else ""
        if not link_name:
            continue

        ids: set[int] = set()
        for attr in ("get_id", "get_actor_id"):
            if not hasattr(link, attr):
                continue
            try:
                ids.add(int(getattr(link, attr)()))
            except Exception:  # noqa: BLE001
                pass

        for attr in ("id", "actor_id"):
            if not hasattr(link, attr):
                continue
            try:
                ids.add(int(getattr(link, attr)))
            except Exception:  # noqa: BLE001
                pass

        for actor_id in ids:
            if actor_id >= 0:
                actor_id_to_link_name[actor_id] = link_name

    return actor_id_to_link_name


def _build_actor_id_to_part(
    actor_id_to_link_name: dict[int, str],
    part_info: PartLinkInfo,
) -> dict[int, int]:
    static_links = set(part_info.static_links)
    moving_links = set(part_info.moving_links)

    actor_id_to_part: dict[int, int] = {}
    for actor_id, link_name in actor_id_to_link_name.items():
        if link_name in moving_links:
            actor_id_to_part[actor_id] = 1
        elif link_name in static_links:
            actor_id_to_part[actor_id] = 0

    return actor_id_to_part


def _camera_intrinsics(camera: CameraSpec) -> tuple[float, float, float, float]:
    if camera.has_intrinsics():
        assert camera.fx is not None
        assert camera.fy is not None
        assert camera.cx is not None
        assert camera.cy is not None
        return float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy)

    if camera.fov_y_deg is None:
        raise ValueError(f"Camera '{camera.name}' has no intrinsics or fov_y_deg")

    fy = float(camera.height) / (
        2.0 * math.tan(math.radians(float(camera.fov_y_deg)) / 2.0)
    )
    fx = fy
    cx = float(camera.width) / 2.0
    cy = float(camera.height) / 2.0
    return fx, fy, cx, cy


def _build_colmap_cameras(cameras: list[CameraSpec]) -> dict[int, Camera]:
    colmap_cameras: dict[int, Camera] = {}
    for camera_id, camera in enumerate(cameras, start=1):
        fx, fy, cx, cy = _camera_intrinsics(camera)
        colmap_cameras[camera_id] = Camera(
            id=camera_id,
            model="PINHOLE",
            width=int(camera.width),
            height=int(camera.height),
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
    return colmap_cameras


def _colmap_world_to_camera(camera: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    camera_to_world_sapien = camera_pose_matrix(camera)
    rotation_camera_to_world_sapien = camera_to_world_sapien[:3, :3]
    camera_center_world = camera_to_world_sapien[:3, 3]

    rotation_world_to_camera = (
        SAPIEN_CAMERA_TO_COLMAP_CAMERA @ rotation_camera_to_world_sapien.T
    )
    translation = -rotation_world_to_camera @ camera_center_world
    qvec = rotmat2qvec(rotation_world_to_camera)
    return qvec, translation


def _backproject_pixels_to_world(
    camera: CameraSpec,
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = _camera_intrinsics(camera)
    x_camera = ((xs.astype(np.float64) - cx) / fx) * depths
    y_camera = ((ys.astype(np.float64) - cy) / fy) * depths
    z_camera = depths
    points_camera = np.column_stack([x_camera, y_camera, z_camera])
    return (rotation_world_to_camera.T @ (points_camera - translation).T).T


def _part_labels_for_actor_labels(
    actor_labels: np.ndarray,
    actor_id_to_part: dict[int, int],
) -> np.ndarray:
    part_labels = np.full(actor_labels.shape, -1, dtype=np.int16)
    for actor_id, part_id in actor_id_to_part.items():
        part_labels[actor_labels == actor_id] = int(part_id)
    return part_labels


def _sample_valid_pixels(
    valid_mask: np.ndarray,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    sampled_mask = valid_mask
    if stride > 1:
        grid_mask = np.zeros(valid_mask.shape, dtype=bool)
        grid_mask[::stride, ::stride] = True
        sampled_mask = valid_mask & grid_mask

    ys, xs = np.nonzero(sampled_mask)
    if max_points > 0 and ys.shape[0] > max_points:
        keep = np.linspace(0, ys.shape[0] - 1, max_points, dtype=np.int64)
        ys = ys[keep]
        xs = xs[keep]
    return ys, xs


def _image_relative_name(camera_name: str, frame_idx: int) -> str:
    return f"{camera_name}/frame_{frame_idx:06d}.png"


def _write_or_link_image(
    model_output: Path,
    colmap_dir: Path,
    image_rel_name: str,
    rgb: np.ndarray,
    image_mode: str,
) -> None:
    if image_mode == "none":
        return

    dest = colmap_dir / "images" / image_rel_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = model_output / "rgb" / image_rel_name

    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        raise FileExistsError(f"Cannot replace non-file image output path: {dest}")

    if source.exists() and image_mode == "symlink":
        rel_target = os.path.relpath(source, start=dest.parent)
        dest.symlink_to(rel_target)
        return

    if source.exists() and image_mode == "copy":
        shutil.copy2(source, dest)
        return

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "imageio is required to write missing COLMAP image files. "
            "Install requirements or use --image-mode symlink/copy with existing RGB frames."
        ) from exc
    imageio.imwrite(dest, rgb)


def _write_part_ply(
    points3D: dict[int, Point3D],
    point_parts: dict[int, int],
    joint_metadata: dict[str, Any],
    ply_path: Path,
) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    point_ids = sorted(points3D.keys())
    joint_metadata_json = json.dumps(
        joint_metadata, sort_keys=True, separators=(",", ":")
    )

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment joint_metadata {joint_metadata_json}\n")
        f.write(f"element vertex {len(point_ids)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar part\n")
        f.write("end_header\n")

        for point_id in point_ids:
            pt = points3D[point_id]
            xyz = np.asarray(pt.xyz, dtype=np.float64)
            rgb = np.clip(np.asarray(pt.rgb, dtype=np.int64), 0, 255)
            part = int(point_parts[point_id])
            f.write(
                f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {part}\n"
            )


def _write_colmap_model(
    cameras: dict[int, Camera],
    images: dict[int, Image],
    points3D: dict[int, Point3D],
    sparse_dir: Path,
    model_format: str,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if model_format in {"txt", "both"}:
        write_model(cameras, images, points3D, str(sparse_dir), ext=".txt")
    if model_format in {"bin", "both"}:
        write_model(cameras, images, points3D, str(sparse_dir), ext=".bin")


def _rendered_joint_values(model_output: Path, cfg: RenderJobConfig) -> np.ndarray:
    """Use the exact rendered q(t), not an independently linearized trajectory."""
    count = cfg.resolved_num_frames()
    csv_path = model_output / "metadata" / "frame_values.csv"
    if csv_path.is_file():
        values = np.full(count, np.nan, dtype=np.float64)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or "frame_index" not in reader.fieldnames or cfg.joint_name not in reader.fieldnames:
                raise ValueError(f"Invalid trajectory CSV columns: {csv_path}")
            for row in reader:
                index = int(row["frame_index"])
                if index < 0 or index >= count or np.isfinite(values[index]):
                    raise ValueError(f"Invalid/duplicate frame index {index} in {csv_path}")
                values[index] = float(row[cfg.joint_name])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Incomplete/nonfinite trajectory in {csv_path}")
        return values
    if count == 1:
        return np.array([float(cfg.q_start)], dtype=np.float64)
    from sapien_urdf_renderer import _sample_joint_trajectory
    return _sample_joint_trajectory(cfg)


def export_model(
    model_output: Path,
    urdf_path: Path | None,
    args: argparse.Namespace,
) -> ExportStats:
    from sapien_urdf_renderer import SapienURDFRenderer

    model_output = model_output.expanduser().resolve()
    if not model_output.exists():
        raise FileNotFoundError(
            f"Model output directory does not exist: {model_output}"
        )

    cfg = _load_render_config(model_output, urdf_path=urdf_path)
    resolved_urdf = cfg.resolved_urdf_path()

    renderer = SapienURDFRenderer(
        background_color=tuple(float(v) for v in cfg.background_color),
        add_ground=cfg.add_ground,
    )
    renderer.load_articulation(str(resolved_urdf), root_pose=cfg.root_pose)
    renderer._validate_joint_for_rendering(cfg.joint_name)

    part_info = parse_part_link_info(resolved_urdf, cfg.joint_name)
    joint_metadata = asdict(parse_joint_metadata(resolved_urdf, cfg.joint_name, cfg))
    actor_id_to_link_name = _extract_actor_id_to_link_name(renderer.articulation)
    actor_id_to_part = _build_actor_id_to_part(actor_id_to_link_name, part_info)
    if not actor_id_to_part:
        raise RuntimeError(
            "Could not map SAPIEN actor IDs to URDF static/moving links. "
            "Cannot export a reliable part-labeled point cloud."
        )
    mapped_parts = set(actor_id_to_part.values())
    if mapped_parts != {0, 1}:
        raise RuntimeError(
            "SAPIEN actor IDs did not cover both URDF parts "
            f"(found parts={sorted(mapped_parts)}). Cannot export a reliable part-labeled point cloud."
        )

    colmap_dir = model_output / args.output_name
    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images" if args.image_mode != "none" else None
    ply_path = model_output / args.pointcloud_dir_name / args.ply_name

    colmap_cameras = _build_colmap_cameras(cfg.cameras)
    num_frames = cfg.resolved_num_frames()
    q_values = _rendered_joint_values(model_output, cfg)
    frame_indices = list(range(int(args.start_frame), num_frames, int(args.frame_step)))
    if args.max_frames is not None and int(args.max_frames) > 0:
        frame_indices = frame_indices[: int(args.max_frames)]

    colmap_images: dict[int, Image] = {}
    points3D: dict[int, Point3D] = {}
    point_parts: dict[int, int] = {}

    next_image_id = 1
    next_point_id = 1

    camera_id_by_name = {
        camera.name: idx for idx, camera in enumerate(cfg.cameras, start=1)
    }
    camera_extrinsics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for camera in cfg.cameras:
        qvec, tvec = _colmap_world_to_camera(camera)
        rotation_world_to_camera = qvec2rotmat(qvec)
        camera_extrinsics[camera.name] = (qvec, tvec, rotation_world_to_camera)

    for frame_idx in tqdm(
        frame_indices, desc=f"COLMAP export {model_output.name}", unit="frame"
    ):
        renderer.set_joint_value(cfg.joint_name, float(q_values[frame_idx]))

        for camera in cfg.cameras:
            image_id = next_image_id
            next_image_id += 1

            frame = renderer.render_frame(camera)
            rgb = frame["rgb"]
            depth_m = frame["depth_m"]
            actor_labels = frame["actor_labels"]
            part_labels = _part_labels_for_actor_labels(actor_labels, actor_id_to_part)

            valid = np.isfinite(depth_m) & (depth_m > 0.0) & (part_labels >= 0)
            ys, xs = _sample_valid_pixels(
                valid_mask=valid,
                stride=int(args.sample_stride),
                max_points=int(args.max_points_per_image),
            )

            image_rel_name = _image_relative_name(camera.name, frame_idx)
            _write_or_link_image(
                model_output=model_output,
                colmap_dir=colmap_dir,
                image_rel_name=image_rel_name,
                rgb=rgb,
                image_mode=args.image_mode,
            )

            qvec, tvec, rotation_world_to_camera = camera_extrinsics[camera.name]
            point3D_ids = np.empty(ys.shape[0], dtype=np.int64)
            xys = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

            if ys.shape[0] > 0:
                depths = depth_m[ys, xs].astype(np.float64)
                world_points = _backproject_pixels_to_world(
                    camera=camera,
                    xs=xs,
                    ys=ys,
                    depths=depths,
                    rotation_world_to_camera=rotation_world_to_camera,
                    translation=tvec,
                )
                colors = rgb[ys, xs, :3].astype(np.uint8)
                parts = part_labels[ys, xs].astype(np.uint8)

                for point_idx, (xyz, color, part) in enumerate(
                    zip(world_points, colors, parts)
                ):
                    point_id = next_point_id
                    next_point_id += 1
                    point3D_ids[point_idx] = point_id
                    points3D[point_id] = Point3D(
                        id=point_id,
                        xyz=np.asarray(xyz, dtype=np.float64),
                        rgb=np.asarray(color, dtype=np.uint8),
                        error=0.0,
                        image_ids=np.asarray([image_id], dtype=np.int32),
                        point2D_idxs=np.asarray([point_idx], dtype=np.int32),
                    )
                    point_parts[point_id] = int(part)

            colmap_images[image_id] = Image(
                id=image_id,
                qvec=np.asarray(qvec, dtype=np.float64),
                tvec=np.asarray(tvec, dtype=np.float64),
                camera_id=camera_id_by_name[camera.name],
                name=image_rel_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    if not points3D:
        raise RuntimeError(
            "No part-labeled 3D points were sampled. Check camera visibility, "
            "actor segmentation, and --sample-stride/--max-points-per-image."
        )

    part_counts = {
        "0": sum(1 for part in point_parts.values() if int(part) == 0),
        "1": sum(1 for part in point_parts.values() if int(part) == 1),
    }

    _write_colmap_model(
        cameras=colmap_cameras,
        images=colmap_images,
        points3D=points3D,
        sparse_dir=sparse_dir,
        model_format=args.model_format,
    )
    _write_part_ply(
        points3D=points3D,
        point_parts=point_parts,
        joint_metadata=joint_metadata,
        ply_path=ply_path,
    )
    joint_metadata_path = ply_path.parent / "joint_metadata.json"
    joint_metadata_path.write_text(
        json.dumps(joint_metadata, indent=2), encoding="utf-8"
    )

    stats = ExportStats(
        model_output=str(model_output),
        urdf_path=str(resolved_urdf),
        colmap_dir=str(colmap_dir),
        sparse_dir=str(sparse_dir),
        images_dir=str(images_dir) if images_dir is not None else None,
        pointcloud_ply=str(ply_path),
        num_cameras=len(colmap_cameras),
        num_images=len(colmap_images),
        num_points3D=len(points3D),
        point_part_counts=part_counts,
        model_format=args.model_format,
        image_mode=args.image_mode,
        sample_stride=int(args.sample_stride),
        max_points_per_image=int(args.max_points_per_image),
        frame_step=int(args.frame_step),
        max_frames=args.max_frames,
        joint_metadata=joint_metadata,
        part_link_info=asdict(part_info),
        actor_id_to_link_name={
            str(k): v for k, v in sorted(actor_id_to_link_name.items())
        },
        actor_id_to_part={str(k): v for k, v in sorted(actor_id_to_part.items())},
    )

    metadata_payload = asdict(stats)
    metadata_filename = getattr(args, "metadata_filename", "colmap_export.json")
    metadata_payload_path = model_output / "metadata" / metadata_filename
    metadata_payload_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload_path.write_text(
        json.dumps(metadata_payload, indent=2), encoding="utf-8"
    )

    export_summary_path = colmap_dir / "export_summary.json"
    export_summary_path.parent.mkdir(parents=True, exist_ok=True)
    export_summary_path.write_text(
        json.dumps(metadata_payload, indent=2), encoding="utf-8"
    )

    return stats


def main() -> int:
    args = parse_args()

    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.max_points_per_image < 0:
        raise ValueError("--max-points-per-image must be >= 0")
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0 when provided")

    model_outputs = _discover_model_outputs(args)
    if args.urdf is not None and len(model_outputs) != 1:
        raise ValueError("--urdf can only be used when exporting one --model-output")

    urdf_map = _build_urdf_map(args.urdf_dir)
    results: list[dict[str, Any]] = []
    failed = 0

    for model_output in model_outputs:
        try:
            urdf_path = _resolve_urdf_for_model(model_output, args, urdf_map)
            stats = export_model(
                model_output=model_output, urdf_path=urdf_path, args=args
            )
            payload = asdict(stats)
            payload["status"] = "ok"
            payload["export"] = "initial"
            results.append(payload)
            print(
                f"[OK] {model_output.name} (initial): "
                f"images={stats.num_images}, points3D={stats.num_points3D}, "
                f"colmap={stats.colmap_dir}, ply={stats.pointcloud_ply}"
            )

            # A separate, self-contained COLMAP reconstruction at t=T.
            # This never mixes points from different articulated poses.
            if not args.no_final_frame:
                cfg = _load_render_config(model_output, urdf_path=urdf_path)
                last_idx = cfg.resolved_num_frames() - 1
                if last_idx > 0:
                    final_args = argparse.Namespace(**vars(args))
                    final_args.start_frame = last_idx
                    final_args.max_frames = 1
                    final_args.output_name = f"{args.output_name}_final"
                    original_ply = Path(args.ply_name)
                    final_args.ply_name = f"{original_ply.stem}_final{original_ply.suffix}"
                    final_args.metadata_filename = "colmap_export_final.json"
                    final_stats = export_model(
                        model_output=model_output, urdf_path=urdf_path, args=final_args
                    )
                    final_payload = asdict(final_stats)
                    final_payload["status"] = "ok"
                    final_payload["export"] = "final"
                    final_payload["frame_index"] = last_idx
                    results.append(final_payload)
                    print(
                        f"[OK] {model_output.name} (final frame {last_idx}): "
                        f"images={final_stats.num_images}, points3D={final_stats.num_points3D}, "
                        f"colmap={final_stats.colmap_dir}, ply={final_stats.pointcloud_ply}"
                    )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append(
                {
                    "status": "failed",
                    "model_output": str(model_output),
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {model_output.name}: {exc}")

    if args.output_root:
        summary_path = (
            Path(args.output_root).expanduser().resolve() / "colmap_export_summary.json"
        )
        summary_path.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"Batch export summary written to: {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

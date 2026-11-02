from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config_types import CameraSpec, ConfigValidationError


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Camera config file does not exist: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".json"}:
            data = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            raise ConfigValidationError(
                f"Unsupported camera config file extension '{path.suffix}'. Use .json/.yaml/.yml"
            )

    if not isinstance(data, dict):
        raise ConfigValidationError("Camera config must be a JSON/YAML object")
    return data


def _normalize_camera_entry(
    entry: dict[str, Any],
    file_default_near: float | None,
    file_default_far: float | None,
    file_width: int | None,
    file_height: int | None,
) -> dict[str, Any]:
    """
    Normalize different camera schemas into CameraSpec-compatible keys.

    Supported input styles:
    - Native schema used by this project
    - Generator-style schema with nested `intrinsics` + `pose` + `extrinsics`
    """
    normalized: dict[str, Any] = dict(entry)

    intrinsics = entry.get("intrinsics")
    if isinstance(intrinsics, dict):
        for key in ("fx", "fy", "cx", "cy"):
            if key not in normalized and key in intrinsics:
                normalized[key] = intrinsics[key]
        if "width" not in normalized and intrinsics.get("width") is not None:
            normalized["width"] = intrinsics["width"]
        if "height" not in normalized and intrinsics.get("height") is not None:
            normalized["height"] = intrinsics["height"]

    pose = entry.get("pose")
    if isinstance(pose, dict):
        for key in ("position", "target", "up"):
            if key not in normalized and key in pose:
                normalized[key] = pose[key]

    extrinsics = entry.get("extrinsics")
    if isinstance(extrinsics, dict) and "pose_matrix_4x4" not in normalized:
        matrix = (
            extrinsics.get("T_world_camera")
            or extrinsics.get("pose_matrix_4x4")
            or extrinsics.get("camera_to_world")
        )
        if matrix is not None:
            normalized["pose_matrix_4x4"] = matrix

    if "near" not in normalized and file_default_near is not None:
        normalized["near"] = file_default_near
    if "far" not in normalized and file_default_far is not None:
        normalized["far"] = file_default_far
    if "width" not in normalized and file_width is not None:
        normalized["width"] = file_width
    if "height" not in normalized and file_height is not None:
        normalized["height"] = file_height

    return normalized


def load_camera_specs(
    config_path: str,
    image_width_override: int | None = None,
    image_height_override: int | None = None,
    default_near: float | None = None,
    default_far: float | None = None,
) -> list[CameraSpec]:
    path = Path(config_path).expanduser().resolve()
    data = _load_json_or_yaml(path)
    if "cameras" not in data or not isinstance(data["cameras"], list):
        raise ConfigValidationError("Camera config must contain a 'cameras' array")

    file_default_near = (
        float(default_near)
        if default_near is not None
        else float(data["default_near"])
        if data.get("default_near") is not None
        else float(data["near"])
        if data.get("near") is not None
        else None
    )
    file_default_far = (
        float(default_far)
        if default_far is not None
        else float(data["default_far"])
        if data.get("default_far") is not None
        else float(data["far"])
        if data.get("far") is not None
        else None
    )
    file_width = (
        int(image_width_override)
        if image_width_override is not None
        else int(data["image_width"])
        if data.get("image_width") is not None
        else int(data["width"])
        if data.get("width") is not None
        else None
    )
    file_height = (
        int(image_height_override)
        if image_height_override is not None
        else int(data["image_height"])
        if data.get("image_height") is not None
        else int(data["height"])
        if data.get("height") is not None
        else None
    )

    cameras: list[CameraSpec] = []
    for entry in data["cameras"]:
        if not isinstance(entry, dict):
            raise ConfigValidationError("Each camera entry must be an object")

        enriched = _normalize_camera_entry(
            entry=entry,
            file_default_near=file_default_near,
            file_default_far=file_default_far,
            file_width=file_width,
            file_height=file_height,
        )
        if image_width_override is not None:
            enriched["width"] = image_width_override
        if image_height_override is not None:
            enriched["height"] = image_height_override
        if file_default_near is not None and "near" not in enriched:
            enriched["near"] = file_default_near
        if file_default_far is not None and "far" not in enriched:
            enriched["far"] = file_default_far

        cameras.append(CameraSpec.from_dict(enriched))

    if not cameras:
        raise ConfigValidationError("No cameras provided")

    names = [cam.name for cam in cameras]
    if len(set(names)) != len(names):
        raise ConfigValidationError("Camera names must be unique")

    return cameras


def _normalize(vec: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ConfigValidationError(f"Cannot normalize zero-length vector for {label}")
    return vec / norm


def look_at_pose_matrix(position: list[float], target: list[float], up: list[float]) -> np.ndarray:
    """
    Build a camera-to-world matrix using SAPIEN's common x-forward, y-left, z-up convention.
    """
    pos = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up_vec = np.asarray(up, dtype=np.float64)

    forward = _normalize(tgt - pos, "forward")
    left = _normalize(np.cross(up_vec, forward), "left")
    true_up = _normalize(np.cross(forward, left), "up")

    rot = np.eye(3, dtype=np.float64)
    rot[:, 0] = forward
    rot[:, 1] = left
    rot[:, 2] = true_up

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def camera_pose_matrix(camera: CameraSpec) -> np.ndarray:
    if camera.pose_matrix_4x4 is not None:
        return np.asarray(camera.pose_matrix_4x4, dtype=np.float64)
    if camera.position is None or camera.target is None or camera.up is None:
        raise ConfigValidationError(
            f"Camera '{camera.name}' missing pose_matrix_4x4 or look-at fields"
        )
    return look_at_pose_matrix(camera.position, camera.target, camera.up)


def matrix3_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat

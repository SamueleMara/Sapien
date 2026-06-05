from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a render job configuration is invalid."""


@dataclass(slots=True)
class RootPoseSpec:
    """Root pose for placing the articulation in world coordinates."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion_wxyz: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootPoseSpec":
        if "position" not in data:
            raise ConfigValidationError("root_pose.position is required when root_pose is provided")
        quaternion = data.get("quaternion_wxyz", data.get("quaternion", [1.0, 0.0, 0.0, 0.0]))
        spec = cls(position=[float(v) for v in data["position"]], quaternion_wxyz=[float(v) for v in quaternion])
        spec.validate()
        return spec

    def validate(self) -> None:
        if len(self.position) != 3:
            raise ConfigValidationError("root_pose.position must have exactly 3 values")
        if len(self.quaternion_wxyz) != 4:
            raise ConfigValidationError("root_pose.quaternion_wxyz must have exactly 4 values")


@dataclass(slots=True)
class CameraSpec:
    """Single camera definition for rendering one view."""

    name: str
    width: int
    height: int
    near: float
    far: float
    fov_y_deg: float | None = None
    position: list[float] | None = None
    target: list[float] | None = None
    up: list[float] | None = None
    pose_matrix_4x4: list[list[float]] | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSpec":
        camera = cls(
            name=str(data["name"]),
            width=int(data["width"]),
            height=int(data["height"]),
            near=float(data["near"]),
            far=float(data["far"]),
            fov_y_deg=float(data["fov_y_deg"]) if data.get("fov_y_deg") is not None else None,
            position=[float(v) for v in data["position"]] if data.get("position") is not None else None,
            target=[float(v) for v in data["target"]] if data.get("target") is not None else None,
            up=[float(v) for v in data["up"]] if data.get("up") is not None else None,
            pose_matrix_4x4=[[float(v) for v in row] for row in data["pose_matrix_4x4"]]
            if data.get("pose_matrix_4x4") is not None
            else None,
            fx=float(data["fx"]) if data.get("fx") is not None else None,
            fy=float(data["fy"]) if data.get("fy") is not None else None,
            cx=float(data["cx"]) if data.get("cx") is not None else None,
            cy=float(data["cy"]) if data.get("cy") is not None else None,
        )
        camera.validate()
        return camera

    def has_intrinsics(self) -> bool:
        return all(v is not None for v in (self.fx, self.fy, self.cx, self.cy))

    def validate(self) -> None:
        if not self.name:
            raise ConfigValidationError("camera.name must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ConfigValidationError(f"camera '{self.name}' width/height must be > 0")
        if self.near <= 0:
            raise ConfigValidationError(f"camera '{self.name}' near must be > 0")
        if self.far <= self.near:
            raise ConfigValidationError(f"camera '{self.name}' far must be > near")

        has_pose_matrix = self.pose_matrix_4x4 is not None
        has_look_at = self.position is not None or self.target is not None or self.up is not None
        if has_pose_matrix and has_look_at:
            raise ConfigValidationError(
                f"camera '{self.name}' must specify either pose_matrix_4x4 or position/target/up, not both"
            )
        if not has_pose_matrix and not has_look_at:
            raise ConfigValidationError(
                f"camera '{self.name}' must define pose_matrix_4x4 or position/target/up look-at values"
            )

        if has_look_at:
            if self.position is None or self.target is None or self.up is None:
                raise ConfigValidationError(
                    f"camera '{self.name}' look-at mode requires position, target, and up"
                )
            if len(self.position) != 3 or len(self.target) != 3 or len(self.up) != 3:
                raise ConfigValidationError(
                    f"camera '{self.name}' position/target/up must each have exactly 3 values"
                )

        if has_pose_matrix:
            if len(self.pose_matrix_4x4) != 4 or any(len(row) != 4 for row in self.pose_matrix_4x4):
                raise ConfigValidationError(
                    f"camera '{self.name}' pose_matrix_4x4 must be a 4x4 matrix"
                )

        if self.fov_y_deg is None and not self.has_intrinsics():
            raise ConfigValidationError(
                f"camera '{self.name}' must define fov_y_deg or fx/fy/cx/cy intrinsics"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RenderJobConfig:
    urdf_path: str
    joint_name: str
    q_start: float
    q_end: float
    cameras: list[CameraSpec]
    output_dir: str
    num_frames: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    joint_space: bool = True
    root_pose: RootPoseSpec | None = None
    image_width: int | None = None
    image_height: int | None = None
    default_near: float | None = None
    default_far: float | None = None
    save_depth_png: bool = True
    save_depth_npy: bool = True
    save_video: bool = False
    video_fps: float | None = None
    video_codec: str = "libx264"
    mask_mode: str = "articulation_ids"
    background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dry_run: bool = False
    list_joints_only: bool = False
    add_ground: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        cameras_override: list[CameraSpec] | None = None,
    ) -> "RenderJobConfig":
        cameras: list[CameraSpec]
        if cameras_override is not None:
            cameras = cameras_override
        else:
            raw_cameras = data.get("cameras")
            if raw_cameras is None:
                raise ConfigValidationError("job config must contain a 'cameras' list or use cameras_override")
            cameras = [CameraSpec.from_dict(item) for item in raw_cameras]

        root_pose = RootPoseSpec.from_dict(data["root_pose"]) if data.get("root_pose") else None

        cfg = cls(
            urdf_path=str(data["urdf_path"]),
            joint_name=str(data["joint_name"]),
            q_start=float(data["q_start"]),
            q_end=float(data["q_end"]),
            num_frames=int(data["num_frames"]) if data.get("num_frames") is not None else None,
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            duration_sec=float(data["duration_sec"]) if data.get("duration_sec") is not None else None,
            cameras=cameras,
            output_dir=str(data["output_dir"]),
            joint_space=bool(data.get("joint_space", True)),
            root_pose=root_pose,
            image_width=int(data["image_width"]) if data.get("image_width") is not None else None,
            image_height=int(data["image_height"]) if data.get("image_height") is not None else None,
            default_near=float(data["default_near"]) if data.get("default_near") is not None else None,
            default_far=float(data["default_far"]) if data.get("default_far") is not None else None,
            save_depth_png=bool(data.get("save_depth_png", True)),
            save_depth_npy=bool(data.get("save_depth_npy", True)),
            save_video=bool(data.get("save_video", False)),
            video_fps=float(data["video_fps"]) if data.get("video_fps") is not None else None,
            video_codec=str(data.get("video_codec", "libx264")),
            mask_mode=str(data.get("mask_mode", "articulation_ids")),
            background_color=[float(v) for v in data.get("background_color", [0.0, 0.0, 0.0])],
            dry_run=bool(data.get("dry_run", False)),
            list_joints_only=bool(data.get("list_joints_only", False)),
            add_ground=bool(data.get("add_ground", False)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.urdf_path:
            raise ConfigValidationError("urdf_path is required")
        if not self.joint_name:
            raise ConfigValidationError("joint_name is required")
        if not self.output_dir:
            raise ConfigValidationError("output_dir is required")

        if not self.cameras:
            raise ConfigValidationError("At least one camera must be provided")
        for camera in self.cameras:
            camera.validate()

        mode_a = self.num_frames is not None
        mode_b = self.fps is not None or self.duration_sec is not None

        if mode_a and mode_b:
            raise ConfigValidationError(
                "Provide either num_frames OR fps+duration_sec, not both"
            )
        if not mode_a and not mode_b:
            raise ConfigValidationError(
                "Frame schedule is missing: provide num_frames OR (fps and duration_sec)"
            )

        if mode_a:
            if self.num_frames is None or self.num_frames < 2:
                raise ConfigValidationError("num_frames must be >= 2 to include both interpolation endpoints")

        if mode_b:
            if self.fps is None or self.duration_sec is None:
                raise ConfigValidationError("Both fps and duration_sec are required when num_frames is omitted")
            if self.fps <= 0 or self.duration_sec <= 0:
                raise ConfigValidationError("fps and duration_sec must be > 0")
            computed = round(self.fps * self.duration_sec)
            if computed < 2:
                raise ConfigValidationError(
                    "round(fps * duration_sec) is < 2, which cannot include both q_start and q_end"
                )

        if self.image_width is not None and self.image_width <= 0:
            raise ConfigValidationError("image_width must be > 0")
        if self.image_height is not None and self.image_height <= 0:
            raise ConfigValidationError("image_height must be > 0")

        if len(self.background_color) != 3:
            raise ConfigValidationError("background_color must have exactly 3 values")

    def resolved_num_frames(self) -> int:
        if self.num_frames is not None:
            return self.num_frames
        if self.fps is None or self.duration_sec is None:
            raise ConfigValidationError("Cannot resolve num_frames without fps and duration_sec")
        return round(self.fps * self.duration_sec)

    def resolved_video_fps(self) -> float:
        if self.video_fps is not None:
            return self.video_fps
        if self.fps is not None:
            return self.fps
        frames = self.resolved_num_frames()
        return float(max(frames - 1, 1))

    def resolved_urdf_path(self) -> Path:
        return Path(self.urdf_path).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

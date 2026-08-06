from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from punch_feedback_core import (
    FeatureError,
    ScoreResult,
    calculate_angle,
    normalized_wrist_speed,
    opposite_side,
    required_visibility,
    shoulder_width,
    side_landmark,
)


LANDMARK_NAMES_3D = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True)
class CameraModel:
    name: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_front_to_camera: np.ndarray
    translation_front_to_camera_mm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_matrix",
            np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "distortion",
            np.asarray(self.distortion, dtype=np.float64).reshape(-1, 1),
        )
        object.__setattr__(
            self,
            "rotation_front_to_camera",
            np.asarray(
                self.rotation_front_to_camera,
                dtype=np.float64,
            ).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "translation_front_to_camera_mm",
            np.asarray(
                self.translation_front_to_camera_mm,
                dtype=np.float64,
            ).reshape(3, 1),
        )

    @property
    def projection_matrix(self) -> np.ndarray:
        extrinsic = np.hstack(
            (
                self.rotation_front_to_camera,
                self.translation_front_to_camera_mm,
            )
        )
        return self.camera_matrix @ extrinsic

    def undistort_pixel(self, pixel: Sequence[float]) -> np.ndarray:
        point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        return cv2.undistortPoints(
            point,
            self.camera_matrix,
            self.distortion,
            P=self.camera_matrix,
        ).reshape(2)

    def camera_point(self, point_front_mm: np.ndarray) -> np.ndarray:
        point = np.asarray(point_front_mm, dtype=np.float64).reshape(3, 1)
        return (
            self.rotation_front_to_camera @ point
            + self.translation_front_to_camera_mm
        ).reshape(3)

    def project(self, point_front_mm: np.ndarray) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(self.rotation_front_to_camera)
        image_points, _ = cv2.projectPoints(
            np.asarray(point_front_mm, dtype=np.float64).reshape(1, 3),
            rvec,
            self.translation_front_to_camera_mm,
            self.camera_matrix,
            self.distortion,
        )
        return image_points.reshape(2)


@dataclass(frozen=True)
class ThreeCameraCalibration:
    image_width: int
    image_height: int
    cameras: Mapping[str, CameraModel]
    T_base_front_mm: np.ndarray | None = None

    def __post_init__(self) -> None:
        missing = {"left", "front", "right"} - set(self.cameras)
        if missing:
            raise ValueError(f"Calibration is missing cameras: {sorted(missing)}")
        for camera in self.cameras.values():
            rotation = camera.rotation_front_to_camera
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
                raise ValueError(f"Invalid rotation matrix for {camera.name}")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
                raise ValueError(f"Invalid rotation determinant for {camera.name}")
        if self.T_base_front_mm is not None:
            transform = np.asarray(self.T_base_front_mm, dtype=np.float64).reshape(4, 4)
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError("Invalid T_base_front_mm homogeneous transform")
            object.__setattr__(self, "T_base_front_mm", transform)

    def front_point_to_base(self, point_front_mm: Sequence[float]) -> np.ndarray:
        if self.T_base_front_mm is None:
            raise ValueError("Calibration does not contain a robot BASE transform")
        point = np.append(np.asarray(point_front_mm, dtype=np.float64).reshape(3), 1.0)
        return (self.T_base_front_mm @ point)[:3]

    def camera_pose_in_base(self, camera_name: str) -> np.ndarray:
        """Return T_base_camera for robot-world calibrations."""
        if self.T_base_front_mm is None:
            raise ValueError("Calibration does not contain a robot BASE transform")
        camera = self.cameras[camera_name]
        T_camera_front = np.eye(4, dtype=np.float64)
        T_camera_front[:3, :3] = camera.rotation_front_to_camera
        T_camera_front[:3, 3] = camera.translation_front_to_camera_mm.reshape(3)
        return self.T_base_front_mm @ np.linalg.inv(T_camera_front)


def _npz_scalar(data: Mapping[str, Any], key: str) -> int:
    return int(np.asarray(data[key]).reshape(()).item())


def _calibration_from_arrays(data: Mapping[str, Any]) -> ThreeCameraCalibration:
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros((3, 1), dtype=np.float64)
    cameras = {
        "front": CameraModel(
            "front",
            data["K_front"],
            data["D_front"],
            identity,
            zero,
        ),
        "left": CameraModel(
            "left",
            data["K_left"],
            data["D_left"],
            data["R_front_to_left"],
            data["T_front_to_left"],
        ),
        "right": CameraModel(
            "right",
            data["K_right"],
            data["D_right"],
            data["R_front_to_right"],
            data["T_front_to_right"],
        ),
    }
    return ThreeCameraCalibration(
        image_width=_npz_scalar(data, "image_width"),
        image_height=_npz_scalar(data, "image_height"),
        cameras=cameras,
    )


def load_three_camera_calibration(path: str | Path) -> ThreeCameraCalibration:
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    if calibration_path.suffix.lower() == ".npz":
        required = {
            "image_width",
            "image_height",
            "K_left",
            "D_left",
            "K_front",
            "D_front",
            "K_right",
            "D_right",
            "R_front_to_left",
            "T_front_to_left",
            "R_front_to_right",
            "T_front_to_right",
        }
        with np.load(calibration_path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"Calibration NPZ is missing keys: {sorted(missing)}"
                )
            arrays = {key: archive[key] for key in required}
        return _calibration_from_arrays(arrays)

    if calibration_path.suffix.lower() in (".yaml", ".yml"):
        with calibration_path.open(encoding="utf-8") as stream:
            yaml_data = yaml.safe_load(stream)
        if isinstance(yaml_data, dict) and "cameras" in yaml_data:
            return _load_robot_world_calibration(calibration_path, yaml_data)

        storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"Cannot open calibration YAML: {calibration_path}")
        matrix_keys = (
            "K_left",
            "D_left",
            "K_front",
            "D_front",
            "K_right",
            "D_right",
            "R_front_to_left",
            "T_front_to_left",
            "R_front_to_right",
            "T_front_to_right",
        )
        try:
            arrays: dict[str, Any] = {
                "image_width": int(storage.getNode("image_width").real()),
                "image_height": int(storage.getNode("image_height").real()),
            }
            for key in matrix_keys:
                matrix = storage.getNode(key).mat()
                if matrix is None:
                    raise ValueError(f"Calibration YAML is missing key: {key}")
                arrays[key] = matrix
        finally:
            storage.release()
        return _calibration_from_arrays(arrays)

    raise ValueError("Calibration must be an .npz, .yaml, or .yml file")


def _resolve_intrinsic_path(robot_world_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path).expanduser()
    if configured.is_absolute() and configured.is_file():
        return configured
    candidates = (
        Path.cwd() / configured,
        robot_world_path.parents[2] / configured,
        robot_world_path.parent / configured,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Intrinsic file referenced by {robot_world_path} was not found: {configured_path}"
    )


def _load_robot_world_calibration(
    robot_world_path: Path,
    data: Mapping[str, Any],
) -> ThreeCameraCalibration:
    """Adapt robot-world camera poses to the front-frame triangulation model."""
    camera_entries = data["cameras"]
    required = {"left", "front", "right"}
    missing = required - set(camera_entries)
    if missing:
        raise ValueError(f"Robot-world calibration is missing cameras: {sorted(missing)}")

    transforms = {
        name: np.asarray(camera_entries[name]["T_base_camera_mm"], dtype=np.float64).reshape(4, 4)
        for name in required
    }
    T_base_front = transforms["front"]
    cameras: dict[str, CameraModel] = {}
    image_sizes: set[tuple[int, int]] = set()
    for name in ("front", "left", "right"):
        entry = camera_entries[name]
        intrinsic_path = _resolve_intrinsic_path(
            robot_world_path, str(entry["intrinsic_file"])
        )
        with intrinsic_path.open(encoding="utf-8") as stream:
            intrinsic = yaml.safe_load(stream)
        image_size = (int(intrinsic["image_width"]), int(intrinsic["image_height"]))
        image_sizes.add(image_size)
        T_camera_front = np.linalg.inv(transforms[name]) @ T_base_front
        cameras[name] = CameraModel(
            name=name,
            camera_matrix=intrinsic["camera_matrix"],
            distortion=intrinsic["distortion_coefficients"],
            rotation_front_to_camera=T_camera_front[:3, :3],
            translation_front_to_camera_mm=T_camera_front[:3, 3],
        )
    if len(image_sizes) != 1:
        raise ValueError(f"All cameras must use one image size, got: {sorted(image_sizes)}")
    image_width, image_height = image_sizes.pop()
    return ThreeCameraCalibration(
        image_width=image_width,
        image_height=image_height,
        cameras=cameras,
        T_base_front_mm=T_base_front,
    )


@dataclass(frozen=True)
class LandmarkObservation2D:
    pixel: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class TriangulatedPoint:
    xyz_mm: np.ndarray
    confidence: float
    reprojection_rms_px: float
    cameras: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "xyz_mm",
            np.asarray(self.xyz_mm, dtype=np.float64).reshape(3),
        )


def _triangulate_dlt(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, LandmarkObservation2D],
    camera_names: Sequence[str],
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for name in camera_names:
        camera = calibration.cameras[name]
        pixel = camera.undistort_pixel(observations[name].pixel)
        projection = camera.projection_matrix
        weight = np.sqrt(max(float(observations[name].confidence), 1e-3))
        rows.append(weight * (pixel[0] * projection[2] - projection[0]))
        rows.append(weight * (pixel[1] * projection[2] - projection[1]))

    if len(rows) < 4:
        return None
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) <= 1e-9:
        return None
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)):
        return None
    return point


def triangulate_landmark(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, LandmarkObservation2D],
    max_reprojection_error_px: float = 10.0,
) -> TriangulatedPoint | None:
    available = tuple(
        name
        for name in ("left", "front", "right")
        if name in observations and observations[name].confidence > 0.0
    )
    if len(available) < 2:
        return None

    candidates: list[tuple[int, float, float, np.ndarray, tuple[str, ...]]] = []
    subsets = list(combinations(available, 2))
    if len(available) == 3:
        subsets.append(available)

    for subset in subsets:
        point = _triangulate_dlt(calibration, observations, subset)
        if point is None:
            continue
        if any(
            calibration.cameras[name].camera_point(point)[2] <= 1e-6
            for name in subset
        ):
            continue

        errors = {
            name: float(
                np.linalg.norm(
                    calibration.cameras[name].project(point)
                    - np.asarray(observations[name].pixel, dtype=np.float64)
                )
            )
            for name in available
        }
        inliers = tuple(
            name
            for name in available
            if errors[name] <= max_reprojection_error_px
            and calibration.cameras[name].camera_point(point)[2] > 0.0
        )
        if len(inliers) < 2:
            continue
        weights = np.asarray(
            [max(observations[name].confidence, 1e-3) for name in inliers],
            dtype=np.float64,
        )
        squared = np.asarray([errors[name] ** 2 for name in inliers])
        rms = float(np.sqrt(np.average(squared, weights=weights)))
        all_mean = float(np.mean(list(errors.values())))
        candidates.append((len(inliers), rms, all_mean, point, inliers))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, _, initial_point, initial_inliers = candidates[0]
    refined = _triangulate_dlt(calibration, observations, initial_inliers)
    point = initial_point if refined is None else refined
    errors = np.asarray(
        [
            np.linalg.norm(
                calibration.cameras[name].project(point)
                - np.asarray(observations[name].pixel, dtype=np.float64)
            )
            for name in initial_inliers
        ],
        dtype=np.float64,
    )
    rms = float(np.sqrt(np.mean(errors**2)))
    mean_visibility = float(
        np.mean([observations[name].confidence for name in initial_inliers])
    )
    confidence = mean_visibility * float(
        np.exp(-rms / max(max_reprojection_error_px, 1e-6))
    )
    return TriangulatedPoint(
        xyz_mm=point,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        reprojection_rms_px=rms,
        cameras=tuple(initial_inliers),
    )


def fuse_realsense_depth(
    triangulated_mm: np.ndarray,
    depth_point_mm: np.ndarray | None,
    weight: float,
    max_disagreement_mm: float,
) -> tuple[np.ndarray, bool]:
    triangulated = np.asarray(triangulated_mm, dtype=np.float64).reshape(3)
    if depth_point_mm is None:
        return triangulated, False
    depth_point = np.asarray(depth_point_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(depth_point)) or depth_point[2] <= 0.0:
        return triangulated, False
    if np.linalg.norm(depth_point - triangulated) > max_disagreement_mm:
        return triangulated, False
    blend = float(np.clip(weight, 0.0, 1.0))
    return triangulated * (1.0 - blend) + depth_point * blend, True


@dataclass(frozen=True)
class Landmark3D:
    x_mm: float
    y_mm: float
    z_mm: float
    confidence: float = 1.0
    reprojection_error_px: float = 0.0
    camera_count: int = 3

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray((self.x_mm, self.y_mm, self.z_mm), dtype=np.float64)


@dataclass
class PoseSample3D:
    stamp_s: float
    landmarks: Mapping[str, Landmark3D]
    front_pose: Any | None = None
    sync_spread_ms: float = 0.0
    depth_fused_count: int = 0


@dataclass
class PunchEvent3D:
    side: str
    samples: list[PoseSample3D]
    max_speed: float


@dataclass
class ClassifiedPunch3D:
    punch_type: str
    side: str
    confidence: float
    key_sample: PoseSample3D
    motion_features: dict[str, float]
    classification_reason: str
    candidate_scores: dict[str, float]


def shoulder_center_3d(sample: PoseSample3D) -> np.ndarray:
    return (
        sample.landmarks["left_shoulder"].xyz
        + sample.landmarks["right_shoulder"].xyz
    ) * 0.5


def hip_center_3d(sample: PoseSample3D) -> np.ndarray:
    return (
        sample.landmarks["left_hip"].xyz
        + sample.landmarks["right_hip"].xyz
    ) * 0.5


def shoulder_width_3d(sample: PoseSample3D) -> float:
    return max(
        float(
            np.linalg.norm(
                sample.landmarks["right_shoulder"].xyz
                - sample.landmarks["left_shoulder"].xyz
            )
        ),
        1e-3,
    )


@dataclass(frozen=True)
class BodyFrame3D:
    origin: np.ndarray
    right_axis: np.ndarray
    up_axis: np.ndarray
    forward_axis: np.ndarray
    shoulder_width_mm: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def body_frame_3d(
    sample: PoseSample3D,
    forward_target_front_mm: np.ndarray | None = None,
) -> BodyFrame3D:
    left = sample.landmarks["left_shoulder"].xyz
    right = sample.landmarks["right_shoulder"].xyz
    origin = (left + right) * 0.5
    right_axis = _unit(right - left)
    raw_up = origin - hip_center_3d(sample)
    orthogonal_up = raw_up - np.dot(raw_up, right_axis) * right_axis
    up_axis = _unit(orthogonal_up)
    forward_axis = _unit(np.cross(right_axis, up_axis))
    target = (
        np.zeros(3, dtype=np.float64)
        if forward_target_front_mm is None
        else np.asarray(forward_target_front_mm, dtype=np.float64).reshape(3)
    )
    if np.dot(forward_axis, target - origin) < 0.0:
        forward_axis = -forward_axis
    return BodyFrame3D(
        origin=origin,
        right_axis=right_axis,
        up_axis=up_axis,
        forward_axis=forward_axis,
        shoulder_width_mm=shoulder_width_3d(sample),
    )


def point_in_body_frame(
    point_front_mm: np.ndarray,
    frame: BodyFrame3D,
) -> np.ndarray:
    relative = np.asarray(point_front_mm, dtype=np.float64).reshape(3) - frame.origin
    return np.asarray(
        (
            np.dot(relative, frame.right_axis),
            np.dot(relative, frame.up_axis),
            np.dot(relative, frame.forward_axis),
        ),
        dtype=np.float64,
    ) / frame.shoulder_width_mm


def wrist_in_body_frame(sample: PoseSample3D, side: str) -> np.ndarray:
    frame = body_frame_3d(sample)
    return point_in_body_frame(
        sample.landmarks[side_landmark(side, "wrist")].xyz,
        frame,
    )


def elbow_angle_3d(sample: PoseSample3D, side: str) -> float:
    return calculate_angle(
        sample.landmarks[side_landmark(side, "shoulder")].xyz,
        sample.landmarks[side_landmark(side, "elbow")].xyz,
        sample.landmarks[side_landmark(side, "wrist")].xyz,
    )


def required_confidence_3d(sample: PoseSample3D, side: str) -> float:
    other = opposite_side(side)
    names = (
        "nose",
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
        f"{other}_wrist",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    return min(sample.landmarks[name].confidence for name in names)


def normalized_wrist_speed_3d(
    previous: PoseSample3D,
    current: PoseSample3D,
    side: str,
) -> float:
    dt = current.stamp_s - previous.stamp_s
    if dt <= 1e-5 or dt > 0.75:
        return 0.0
    return float(
        np.linalg.norm(
            wrist_in_body_frame(current, side)
            - wrist_in_body_frame(previous, side)
        )
        / dt
    )


class PunchDetector3D:
    """Slow-motion-friendly three-dimensional punch state machine."""

    def __init__(self, settings: Mapping[str, Any]):
        self.guard_speed = float(settings.get("guard_speed", 0.45))
        self.guard_min_visibility = float(
            settings.get("guard_min_visibility", 0.30)
        )
        self.start_speed = float(settings.get("start_speed", 0.25))
        self.end_speed = float(settings.get("end_speed", 0.12))
        self.speed_window = max(int(settings.get("speed_window", 3)), 1)
        self.start_displacement_ratio = float(
            settings.get("start_displacement_ratio", 0.10)
        )
        self.start_extension_deg = float(settings.get("start_extension_deg", 8.0))
        self.slow_start_displacement_ratio = float(
            settings.get("slow_start_displacement_ratio", 0.22)
        )
        self.slow_start_speed = float(settings.get("slow_start_speed", 0.05))
        self.slow_start_extension_deg = float(
            settings.get("slow_start_extension_deg", 12.0)
        )
        self.start_confirm_frames = max(
            int(settings.get("start_confirm_frames", 2)),
            1,
        )
        self.ready_frames = max(int(settings.get("ready_frames", 3)), 1)
        self.end_frames = max(int(settings.get("end_frames", 4)), 1)
        self.end_step_displacement_ratio = float(
            settings.get("end_step_displacement_ratio", 0.015)
        )
        self.min_duration_s = float(settings.get("min_duration_s", 0.20))
        self.max_duration_s = float(settings.get("max_duration_s", 3.0))
        self.cooldown_s = float(settings.get("cooldown_s", 0.7))
        self.min_confidence = float(settings.get("min_confidence", 0.18))
        self.guard_max_wrist_face_ratio = float(
            settings.get("guard_max_wrist_face_ratio", 2.6)
        )
        self.max_lost_frames = max(int(settings.get("max_lost_frames", 6)), 1)

        self._state = "WAIT_GUARD"
        self.previous: PoseSample3D | None = None
        self.guard_baseline: PoseSample3D | None = None
        self.active_side: str | None = None
        self.active_started_at: float | None = None
        self.samples: list[PoseSample3D] = []
        self.max_speed = 0.0
        self.stable_frames = 0
        self.slow_frames = 0
        self.lost_frames = 0
        self.cooldown_until = 0.0
        self.start_confirm = {"left": 0, "right": 0}
        self.speed_history = {
            side: deque(maxlen=self.speed_window) for side in ("left", "right")
        }
        self.guard_speed_history = {
            side: deque(maxlen=self.speed_window) for side in ("left", "right")
        }
        self._telemetry = {
            "left_speed": 0.0,
            "right_speed": 0.0,
            "left_displacement": 0.0,
            "right_displacement": 0.0,
            "left_elbow_delta": 0.0,
            "right_elbow_delta": 0.0,
            "left_guard_speed": 0.0,
            "right_guard_speed": 0.0,
            "left_guard_face_ratio": float("inf"),
            "right_guard_face_ratio": float("inf"),
            "guard_quality": 0.0,
            "guard_quality_limit": self.guard_min_visibility,
            "guard_speed_limit": self.guard_speed,
            "guard_face_limit": self.guard_max_wrist_face_ratio,
            "guard_source_2d": 0.0,
        }

    @property
    def state(self) -> str:
        if self._state == "ACTIVE" and self.active_side is not None:
            return f"ACTIVE_{self.active_side.upper()}"
        return self._state

    @property
    def telemetry(self) -> dict[str, float]:
        values = dict(self._telemetry)
        values.update(
            {
                "guard_frames": float(self.stable_frames),
                "ready_frames": float(self.ready_frames),
                "left_start_confirm": float(self.start_confirm["left"]),
                "right_start_confirm": float(self.start_confirm["right"]),
                "start_confirm_frames": float(self.start_confirm_frames),
            }
        )
        return values

    def reset(self) -> None:
        self.__init__(
            {
                "guard_speed": self.guard_speed,
                "guard_min_visibility": self.guard_min_visibility,
                "start_speed": self.start_speed,
                "end_speed": self.end_speed,
                "speed_window": self.speed_window,
                "start_displacement_ratio": self.start_displacement_ratio,
                "start_extension_deg": self.start_extension_deg,
                "slow_start_displacement_ratio": self.slow_start_displacement_ratio,
                "slow_start_speed": self.slow_start_speed,
                "slow_start_extension_deg": self.slow_start_extension_deg,
                "start_confirm_frames": self.start_confirm_frames,
                "ready_frames": self.ready_frames,
                "end_frames": self.end_frames,
                "end_step_displacement_ratio": self.end_step_displacement_ratio,
                "min_duration_s": self.min_duration_s,
                "max_duration_s": self.max_duration_s,
                "cooldown_s": self.cooldown_s,
                "min_confidence": self.min_confidence,
                "guard_max_wrist_face_ratio": self.guard_max_wrist_face_ratio,
                "max_lost_frames": self.max_lost_frames,
            }
        )

    def _pose_visible(self, sample: PoseSample3D) -> bool:
        return all(
            required_confidence_3d(sample, side) >= self.min_confidence
            for side in ("left", "right")
        )

    def _guard_valid(self, sample: PoseSample3D) -> bool:
        del sample
        return bool(
            self._telemetry["guard_quality"]
            >= self._telemetry["guard_quality_limit"]
            and all(
                self._telemetry[f"{side}_guard_face_ratio"]
                <= self.guard_max_wrist_face_ratio
                for side in ("left", "right")
            )
        )

    def _guard_pose_metrics(
        self,
        sample: PoseSample3D,
    ) -> tuple[float, dict[str, float], float, bool]:
        """Return guard quality and wrist/face ratios.

        The live node always supplies ``front_pose``. Using that 2D pose for
        guard acquisition avoids treating triangulation/depth jitter as body
        movement. Synthetic/core callers without it retain the 3D fallback.
        """
        if sample.front_pose is not None:
            front_pose = sample.front_pose
            scale = shoulder_width(front_pose)
            nose = front_pose.landmarks["nose"].xy
            quality = min(
                required_visibility(front_pose, side)
                for side in ("left", "right")
            )
            ratios = {
                side: float(
                    np.linalg.norm(
                        front_pose.landmarks[f"{side}_wrist"].xy - nose
                    )
                    / scale
                )
                for side in ("left", "right")
            }
            return quality, ratios, self.guard_min_visibility, True

        nose = sample.landmarks["nose"].xyz
        scale = shoulder_width_3d(sample)
        quality = min(
            required_confidence_3d(sample, side) for side in ("left", "right")
        )
        ratios = {
            side: float(
                np.linalg.norm(sample.landmarks[f"{side}_wrist"].xyz - nose)
                / scale
            )
            for side in ("left", "right")
        }
        return quality, ratios, self.min_confidence, False

    def _speeds(
        self,
        previous: PoseSample3D,
        current: PoseSample3D,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for side in ("left", "right"):
            self.speed_history[side].append(
                normalized_wrist_speed_3d(previous, current, side)
            )
            result[side] = float(np.median(self.speed_history[side]))
        return result

    def _guard_speeds(
        self,
        previous: PoseSample3D,
        current: PoseSample3D,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        use_front_2d = (
            previous.front_pose is not None and current.front_pose is not None
        )
        for side in ("left", "right"):
            if use_front_2d:
                raw_speed = normalized_wrist_speed(
                    previous.front_pose,
                    current.front_pose,
                    side,
                )
            else:
                raw_speed = normalized_wrist_speed_3d(previous, current, side)
            self.guard_speed_history[side].append(raw_speed)
            result[side] = float(np.median(self.guard_speed_history[side]))
        return result

    def _refresh_telemetry(
        self,
        sample: PoseSample3D,
        speeds: Mapping[str, float],
        guard_speeds: Mapping[str, float],
    ) -> None:
        guard_quality, guard_ratios, quality_limit, using_2d = (
            self._guard_pose_metrics(sample)
        )
        self._telemetry["guard_quality"] = float(guard_quality)
        self._telemetry["guard_quality_limit"] = float(quality_limit)
        self._telemetry["guard_source_2d"] = float(using_2d)
        for side in ("left", "right"):
            displacement = 0.0
            elbow_delta = 0.0
            if self.guard_baseline is not None:
                displacement = float(
                    np.linalg.norm(
                        wrist_in_body_frame(sample, side)
                        - wrist_in_body_frame(self.guard_baseline, side)
                    )
                )
                elbow_delta = max(
                    0.0,
                    elbow_angle_3d(sample, side)
                    - elbow_angle_3d(self.guard_baseline, side),
                )
            self._telemetry[f"{side}_speed"] = float(speeds[side])
            self._telemetry[f"{side}_displacement"] = displacement
            self._telemetry[f"{side}_elbow_delta"] = elbow_delta
            self._telemetry[f"{side}_guard_speed"] = float(guard_speeds[side])
            self._telemetry[f"{side}_guard_face_ratio"] = float(
                guard_ratios[side]
            )

    def _start_side(
        self,
        sample: PoseSample3D,
        speeds: Mapping[str, float],
    ) -> str | None:
        candidates: list[str] = []
        for side in ("left", "right"):
            displacement = self._telemetry[f"{side}_displacement"]
            elbow_delta = self._telemetry[f"{side}_elbow_delta"]
            normal_start = (
                speeds[side] >= self.start_speed
                and (
                    displacement >= self.start_displacement_ratio
                    or elbow_delta >= self.start_extension_deg
                )
            )
            slow_start = (
                displacement >= self.slow_start_displacement_ratio
                and (
                    elbow_delta >= self.slow_start_extension_deg
                    or speeds[side] >= self.slow_start_speed
                )
            )
            visible = required_confidence_3d(sample, side) >= self.min_confidence
            if visible and (normal_start or slow_start):
                self.start_confirm[side] += 1
            else:
                self.start_confirm[side] = 0
            if self.start_confirm[side] >= self.start_confirm_frames:
                candidates.append(side)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda side: self._telemetry[f"{side}_displacement"],
        )

    def _return_to_wait(self) -> None:
        self._state = "WAIT_GUARD"
        self.guard_baseline = None
        self.active_side = None
        self.active_started_at = None
        self.samples = []
        self.max_speed = 0.0
        self.stable_frames = 0
        self.slow_frames = 0
        self.start_confirm = {"left": 0, "right": 0}

    def update(self, sample: PoseSample3D) -> PunchEvent3D | None:
        if self.previous is None:
            self.previous = sample
            return None

        speeds = self._speeds(self.previous, sample)
        guard_speeds = self._guard_speeds(self.previous, sample)
        visible = self._pose_visible(sample)
        self.lost_frames = 0 if visible else self.lost_frames + 1
        self._refresh_telemetry(sample, speeds, guard_speeds)

        if self._state == "COOLDOWN":
            if sample.stamp_s < self.cooldown_until:
                self.previous = sample
                return None
            self._return_to_wait()

        if self._state == "WAIT_GUARD":
            if (
                self._guard_valid(sample)
                and max(guard_speeds.values()) <= self.guard_speed
            ):
                self.stable_frames += 1
            else:
                self.stable_frames = max(0, self.stable_frames - 1)
            if self.stable_frames >= self.ready_frames:
                self._state = "READY"
                self.guard_baseline = sample
            self.previous = sample
            return None

        if self._state == "READY":
            if self.lost_frames >= self.max_lost_frames:
                self._return_to_wait()
                self.previous = sample
                return None
            side = self._start_side(sample, speeds)
            if side is not None:
                self._state = "ACTIVE"
                self.active_side = side
                self.active_started_at = sample.stamp_s
                self.samples = [self.guard_baseline or self.previous, sample]
                self.max_speed = float(speeds[side])
                self.slow_frames = 0
            # READY is deliberately latched to its original guard sample.
            # Updating that baseline here would erase a gradual slow punch.
            self.previous = sample
            return None

        if self._state != "ACTIVE" or self.active_side is None:
            self._return_to_wait()
            self.previous = sample
            return None

        side = self.active_side
        speed = float(speeds[side])
        step_displacement = float(
            np.linalg.norm(
                wrist_in_body_frame(sample, side)
                - wrist_in_body_frame(self.previous, side)
            )
        )
        self.samples.append(sample)
        self.max_speed = max(self.max_speed, speed)
        stopped = (
            speed <= self.end_speed
            and step_displacement <= self.end_step_displacement_ratio
        )
        self.slow_frames = self.slow_frames + 1 if stopped else 0
        active_started_at = (
            self.samples[0].stamp_s
            if self.active_started_at is None
            else self.active_started_at
        )
        duration = sample.stamp_s - active_started_at
        finished = (
            duration >= self.min_duration_s and self.slow_frames >= self.end_frames
        ) or duration >= self.max_duration_s
        self.previous = sample
        if not finished:
            return None

        event = PunchEvent3D(side, list(self.samples), self.max_speed)
        self._state = "COOLDOWN"
        self.cooldown_until = sample.stamp_s + self.cooldown_s
        self.active_side = None
        self.active_started_at = None
        self.samples = []
        self.guard_baseline = None
        self.stable_frames = 0
        self.slow_frames = 0
        self.start_confirm = {"left": 0, "right": 0}
        return event


def _first_near_peak_index(values: np.ndarray, fraction: float) -> int:
    if values.size == 0:
        return 0
    peak = float(np.max(values))
    if peak <= 1e-9:
        return 0
    indices = np.flatnonzero(values >= peak * fraction)
    return int(indices[0]) if indices.size else int(np.argmax(values))


def _trajectory_geometry_3d(
    points: np.ndarray,
) -> tuple[float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 1.0
    path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    direct_vector = points[-1] - points[0]
    direct = float(np.linalg.norm(direct_vector))
    if path_length <= 1e-9 or direct <= 1e-9:
        return path_length, direct, 0.0, 1.0
    linearity = float(np.clip(direct / path_length, 0.0, 1.0))
    relative = points - points[0]
    perpendicular = np.linalg.norm(np.cross(relative, direct_vector), axis=1) / direct
    curvature = float(np.max(perpendicular) / direct)
    return path_length, direct, linearity, curvature


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def classify_punch_3d(
    event: PunchEvent3D,
    settings: Mapping[str, Any],
) -> ClassifiedPunch3D:
    if len(event.samples) < 2:
        raise ValueError("3D punch event must contain at least two samples")
    side = event.side
    body_points = np.asarray(
        [wrist_in_body_frame(sample, side) for sample in event.samples],
        dtype=np.float64,
    )
    elbow_angles_all = np.asarray(
        [elbow_angle_3d(sample, side) for sample in event.samples],
        dtype=np.float64,
    )
    displacement_all = np.linalg.norm(body_points - body_points[0], axis=1)
    forward_all = body_points[:, 2] - body_points[0, 2]
    extension_all = np.maximum(elbow_angles_all - elbow_angles_all[0], 0.0)

    peak_fraction = float(np.clip(settings.get("impact_peak_fraction", 0.92), 0.5, 1.0))
    impact_index = max(
        _first_near_peak_index(displacement_all, peak_fraction),
        _first_near_peak_index(np.maximum(forward_all, 0.0), peak_fraction),
        _first_near_peak_index(extension_all, peak_fraction),
    )
    impact_index = min(max(impact_index, 1), len(event.samples) - 1)

    points = body_points[: impact_index + 1]
    deltas = points - points[0]
    elbow_angles = elbow_angles_all[: impact_index + 1]
    distances = np.linalg.norm(deltas, axis=1)
    peak_travel = max(float(np.max(distances)), 1e-6)
    lateral_travel = float(np.max(np.abs(deltas[:, 0])))
    upward_travel = float(np.max(deltas[:, 1]))
    forward_travel = float(np.max(deltas[:, 2]))
    lateral_ratio = lateral_travel / peak_travel
    upward_ratio = max(upward_travel, 0.0) / peak_travel
    forward_ratio = max(forward_travel, 0.0) / peak_travel
    outward_sign = 1.0 if side == "right" else -1.0
    outward_positions = outward_sign * deltas[:, 0]
    outward_peak_index = int(np.argmax(outward_positions))
    outward_travel = max(float(outward_positions[outward_peak_index]), 0.0)
    inward_return = max(
        outward_travel - float(outward_positions[-1]),
        0.0,
    )
    path_length, direct_travel, linearity, curvature = _trajectory_geometry_3d(points)
    max_elbow = float(np.max(elbow_angles))
    extension_gain = max_elbow - float(elbow_angles[0])
    impact_elbow = float(elbow_angles[-1])

    impact_sample = event.samples[impact_index]
    impact_frame = body_frame_3d(impact_sample)
    wrist = impact_sample.landmarks[side_landmark(side, "wrist")].xyz
    elbow = impact_sample.landmarks[side_landmark(side, "elbow")].xyz
    wrist_above_elbow = float(
        np.dot(wrist - elbow, impact_frame.up_axis)
        / impact_frame.shoulder_width_mm
    )

    time_values = np.asarray(
        [sample.stamp_s for sample in event.samples[: impact_index + 1]],
        dtype=np.float64,
    )
    velocities: list[np.ndarray] = []
    for index in range(1, len(points)):
        dt = time_values[index] - time_values[index - 1]
        if 1e-5 < dt <= 0.75:
            velocities.append((points[index] - points[index - 1]) / dt)
    velocity_array = np.asarray(velocities) if velocities else np.zeros((1, 3))
    peak_forward_speed = float(np.max(velocity_array[:, 2]))
    peak_lateral_speed = float(np.max(np.abs(velocity_array[:, 0])))
    peak_upward_speed = float(np.max(velocity_array[:, 1]))

    straight_forward_min = float(settings.get("straight_forward_ratio_min", 0.42))
    straight_linearity_min = float(settings.get("straight_linearity_min", 0.84))
    straight_curvature_max = float(settings.get("straight_curvature_max", 0.22))
    straight_elbow_min = float(settings.get("straight_elbow_angle_min", 138.0))
    straight_extension_min = float(settings.get("straight_extension_gain_min", 8.0))

    hook_lateral_min = float(settings.get("hook_lateral_ratio_min", 0.42))
    hook_curvature_min = float(settings.get("hook_curvature_min", 0.16))
    hook_elbow_max = float(settings.get("hook_elbow_angle_max", 158.0))
    hook_outward_min = float(settings.get("hook_outward_travel_min", 0.12))
    hook_inward_min = float(settings.get("hook_inward_return_min", 0.08))

    upper_upward_min = float(settings.get("uppercut_upward_ratio_min", 0.42))
    upper_upward_travel_min = float(
        settings.get("uppercut_upward_travel_min", 0.18)
    )
    upper_elbow_max = float(settings.get("uppercut_elbow_angle_max", 158.0))

    straight_score = (
        0.32 * _ramp(forward_ratio, straight_forward_min * 0.65, 0.75)
        + 0.28 * _ramp(linearity, straight_linearity_min * 0.90, 0.98)
        + 0.16 * (1.0 - _ramp(curvature, straight_curvature_max * 0.5, 0.45))
        + 0.14 * _ramp(max_elbow, straight_elbow_min - 18.0, 168.0)
        + 0.10 * _ramp(extension_gain, straight_extension_min * 0.5, 35.0)
    )
    hook_score = (
        0.26 * _ramp(lateral_ratio, hook_lateral_min * 0.65, 0.80)
        + 0.22 * _ramp(curvature, hook_curvature_min * 0.65, 0.45)
        + 0.14 * (1.0 - _ramp(impact_elbow, hook_elbow_max - 35.0, 175.0))
        + 0.12 * _ramp(lateral_travel, 0.12, 0.65)
        + 0.16 * _ramp(outward_travel, hook_outward_min * 0.65, 0.45)
        + 0.10 * _ramp(inward_return, hook_inward_min * 0.65, 0.35)
    )
    uppercut_score = (
        0.38 * _ramp(upward_ratio, upper_upward_min * 0.65, 0.80)
        + 0.24 * _ramp(upward_travel, upper_upward_travel_min * 0.65, 0.60)
        + 0.14 * _ramp(wrist_above_elbow, 0.0, 0.40)
        + 0.14 * (1.0 - _ramp(impact_elbow, upper_elbow_max - 35.0, 175.0))
        + 0.10 * _ramp(forward_ratio, 0.10, 0.55)
    )

    straight_gate = (
        forward_ratio >= straight_forward_min
        and linearity >= straight_linearity_min
        and curvature <= straight_curvature_max
        and max_elbow >= straight_elbow_min
        and extension_gain >= straight_extension_min
    )
    hook_gate = (
        lateral_ratio >= hook_lateral_min
        and curvature >= hook_curvature_min
        and impact_elbow <= hook_elbow_max
        and outward_travel >= hook_outward_min
        and inward_return >= hook_inward_min
        and lateral_travel >= max(upward_travel, 0.0) * 0.90
    )
    uppercut_gate = (
        upward_ratio >= upper_upward_min
        and upward_travel >= upper_upward_travel_min
        and impact_elbow <= upper_elbow_max
        and upward_travel >= lateral_travel * 0.90
    )
    if not straight_gate:
        straight_score *= 0.72
    if not hook_gate:
        hook_score *= 0.72
    if not uppercut_gate:
        uppercut_score *= 0.72

    candidate_scores = {
        "straight": float(np.clip(straight_score, 0.0, 1.0)),
        "hook": float(np.clip(hook_score, 0.0, 1.0)),
        "uppercut": float(np.clip(uppercut_score, 0.0, 1.0)),
    }
    ordered = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    punch_type, winner = ordered[0]
    margin = winner - ordered[1][1]
    confidence = float(np.clip(0.40 + 0.45 * winner + 0.60 * margin, 0.0, 0.95))
    reasons = {
        "straight": "straight_3d_forward_linear_vector",
        "hook": "hook_3d_lateral_curved_vector",
        "uppercut": "uppercut_3d_upward_vector",
    }

    motion_features = {
        "delta_x_lateral_ratio": float(deltas[-1, 0]),
        "delta_y_up_ratio": float(deltas[-1, 1]),
        "delta_z_forward_ratio": float(deltas[-1, 2]),
        "lateral_travel_ratio": lateral_travel,
        "hook_outward_travel_ratio": outward_travel,
        "hook_inward_return_ratio": inward_return,
        "upward_travel_ratio": upward_travel,
        "forward_travel_ratio": forward_travel,
        "lateral_component_ratio": lateral_ratio,
        "upward_component_ratio": upward_ratio,
        "forward_component_ratio": forward_ratio,
        "path_length_ratio": path_length,
        "direct_travel_ratio": direct_travel,
        "path_linearity": linearity,
        "path_curvature_ratio": curvature,
        "max_elbow_angle_deg": max_elbow,
        "impact_elbow_angle_deg": impact_elbow,
        "extension_gain_deg": extension_gain,
        "wrist_above_elbow_ratio": wrist_above_elbow,
        "peak_forward_speed_ratio_s": peak_forward_speed,
        "peak_lateral_speed_ratio_s": peak_lateral_speed,
        "peak_upward_speed_ratio_s": peak_upward_speed,
        "straight_candidate_score": candidate_scores["straight"],
        "hook_candidate_score": candidate_scores["hook"],
        "uppercut_candidate_score": candidate_scores["uppercut"],
        "classification_margin": margin,
        "impact_sample_index": float(impact_index),
        "event_sample_count": float(len(event.samples)),
        "recovery_frames_ignored": float(len(event.samples) - impact_index - 1),
        "max_speed_body_widths_s": float(event.max_speed),
        "mean_sync_spread_ms": float(
            np.mean([sample.sync_spread_ms for sample in event.samples])
        ),
        "mean_depth_fused_joints": float(
            np.mean([sample.depth_fused_count for sample in event.samples])
        ),
    }
    return ClassifiedPunch3D(
        punch_type=punch_type,
        side=side,
        confidence=confidence,
        key_sample=impact_sample,
        motion_features=motion_features,
        classification_reason=reasons[punch_type],
        candidate_scores=candidate_scores,
    )


def _point_line_distance_3d(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    line = line_end - line_start
    length = float(np.linalg.norm(line))
    if length <= 1e-9:
        return 0.0
    return float(np.linalg.norm(np.cross(point - line_start, line)) / length)


def extract_form_features_3d(
    classified: ClassifiedPunch3D,
) -> dict[str, float]:
    sample = classified.key_sample
    side = classified.side
    other = opposite_side(side)
    scale = shoulder_width_3d(sample)
    shoulder = sample.landmarks[f"{side}_shoulder"].xyz
    elbow = sample.landmarks[f"{side}_elbow"].xyz
    wrist = sample.landmarks[f"{side}_wrist"].xyz
    guard = sample.landmarks[f"{other}_wrist"].xyz
    nose = sample.landmarks["nose"].xyz
    frame = body_frame_3d(sample)

    values = {
        "strike_elbow_angle_deg": calculate_angle(shoulder, elbow, wrist),
        "elbow_flare_ratio": _point_line_distance_3d(
            elbow,
            shoulder,
            wrist,
        )
        / scale,
        "guard_to_face_ratio": float(np.linalg.norm(guard - nose) / scale),
        "strike_wrist_height_ratio": float(
            np.dot(wrist - shoulder, frame.up_axis) / scale
        ),
        "wrist_elbow_height_ratio": float(
            np.dot(wrist - elbow, frame.up_axis) / scale
        ),
    }
    values.update(classified.motion_features)
    return values


def score_punch_3d(
    classified: ClassifiedPunch3D,
    profiles: Mapping[str, Any],
    feedback_settings: Mapping[str, Any],
) -> ScoreResult:
    try:
        profile = profiles[classified.punch_type]
    except KeyError as error:
        raise ValueError(
            f"Missing 3D reference profile: {classified.punch_type}"
        ) from error

    values = extract_form_features_3d(classified)
    errors: list[FeatureError] = []
    weighted_penalty = 0.0
    total_weight = 0.0
    for feature, reference in profile["features"].items():
        if feature not in values:
            raise ValueError(f"Unsupported 3D form feature in YAML: {feature}")
        target = float(reference["target"])
        tolerance = max(float(reference["tolerance"]), 1e-6)
        weight = max(float(reference["weight"]), 0.0)
        value = float(values[feature])
        error_ratio = abs(value - target) / tolerance
        errors.append(
            FeatureError(
                feature=feature,
                joint=str(reference["joint"]),
                code=str(reference["code"]),
                value=value,
                target=target,
                tolerance=tolerance,
                weight=weight,
                error_ratio=error_ratio,
            )
        )
        weighted_penalty += weight * min(error_ratio, 1.0)
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError(f"3D profile has zero total weight: {classified.punch_type}")

    score = 100.0 * (1.0 - weighted_penalty / total_weight)
    joint_threshold = float(feedback_settings.get("joint_error_threshold", 1.0))
    score_threshold = float(feedback_settings.get("score_threshold", 30.0))
    violations = [error for error in errors if error.error_ratio >= joint_threshold]
    feedback_required = score < score_threshold or bool(violations)
    if feedback_required and not violations:
        violations = sorted(
            errors,
            key=lambda error: error.error_ratio,
            reverse=True,
        )[:2]
    return ScoreResult(
        total_score=float(np.clip(score, 0.0, 100.0)),
        feedback_required=feedback_required,
        feature_errors=errors,
        violations=violations,
    )

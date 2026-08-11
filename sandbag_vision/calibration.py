from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import yaml

from .types import CAMERAS


@dataclass(frozen=True)
class CameraModel:
    name: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    T_base_camera_mm: np.ndarray
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_matrix", np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "distortion", np.asarray(self.distortion, dtype=np.float64).reshape(-1, 1))
        transform = np.asarray(self.T_base_camera_mm, dtype=np.float64).reshape(4, 4)
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError(f"{self.name}: invalid homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3):
            raise ValueError(f"{self.name}: invalid camera rotation")
        object.__setattr__(self, "T_base_camera_mm", transform)
        object.__setattr__(self, "_T_camera_base_mm", np.linalg.inv(transform))

    @property
    def T_camera_base_mm(self) -> np.ndarray:
        return self._T_camera_base_mm.copy()

    @property
    def projection_base_normalized(self) -> np.ndarray:
        return self._T_camera_base_mm[:3, :]

    @property
    def origin_base_mm(self) -> np.ndarray:
        return self.T_base_camera_mm[:3, 3].copy()

    def base_to_camera(self, point_base_mm: Sequence[float]) -> np.ndarray:
        homogeneous = np.append(np.asarray(point_base_mm, dtype=np.float64).reshape(3), 1.0)
        return (self._T_camera_base_mm @ homogeneous)[:3]

    def camera_to_base(self, point_camera_mm: Sequence[float]) -> np.ndarray:
        homogeneous = np.append(np.asarray(point_camera_mm, dtype=np.float64).reshape(3), 1.0)
        return (self.T_base_camera_mm @ homogeneous)[:3]

    def undistort_normalized(self, pixel: Sequence[float]) -> np.ndarray:
        return cv2.undistortPoints(
            np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
            self.camera_matrix,
            self.distortion,
        ).reshape(2)

    def project_base(self, point_base_mm: Sequence[float]) -> np.ndarray:
        point_camera = self.base_to_camera(point_base_mm)
        if point_camera[2] <= 1e-9:
            return np.asarray((np.nan, np.nan), dtype=np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        projected, _ = cv2.projectPoints(
            point_camera.reshape(1, 3),
            rvec,
            tvec,
            self.camera_matrix,
            self.distortion,
        )
        return projected.reshape(2)

    def ray_base(self, pixel: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.undistort_normalized(pixel)
        direction_camera = np.asarray((normalized[0], normalized[1], 1.0), dtype=np.float64)
        direction_camera /= np.linalg.norm(direction_camera)
        direction_base = self.T_base_camera_mm[:3, :3] @ direction_camera
        direction_base /= np.linalg.norm(direction_base)
        return self.origin_base_mm, direction_base

    def depth_pixel_to_base(self, pixel: Sequence[float], depth_mm: float) -> np.ndarray:
        normalized = self.undistort_normalized(pixel)
        point_camera = np.asarray(
            (normalized[0] * depth_mm, normalized[1] * depth_mm, depth_mm),
            dtype=np.float64,
        )
        return self.camera_to_base(point_camera)


@dataclass(frozen=True)
class ThreeCameraCalibration:
    cameras: Mapping[str, CameraModel]

    def __post_init__(self) -> None:
        missing = set(CAMERAS) - set(self.cameras)
        if missing:
            raise ValueError(f"calibration is missing cameras: {sorted(missing)}")

    def project_workspace(
        self,
        camera_name: str,
        workspace: Mapping[str, Sequence[float]],
        margin_ratio: float = 0.08,
    ) -> tuple[int, int, int, int] | None:
        camera = self.cameras[camera_name]
        corners = np.asarray(
            [
                (x, y, z)
                for x in workspace["x"]
                for y in workspace["y"]
                for z in workspace["z"]
            ],
            dtype=np.float64,
        )
        visible = [camera.project_base(point) for point in corners if camera.base_to_camera(point)[2] > 1.0]
        visible = [point for point in visible if np.all(np.isfinite(point))]
        if not visible:
            return None
        points = np.asarray(visible)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        margin_x = (x2 - x1) * max(float(margin_ratio), 0.0)
        margin_y = (y2 - y1) * max(float(margin_ratio), 0.0)
        x1 = int(np.clip(np.floor(x1 - margin_x), 0, camera.image_width - 2))
        y1 = int(np.clip(np.floor(y1 - margin_y), 0, camera.image_height - 2))
        x2 = int(np.clip(np.ceil(x2 + margin_x), x1 + 1, camera.image_width))
        y2 = int(np.clip(np.ceil(y2 + margin_y), y1 + 1, camera.image_height))
        return (x1, y1, x2, y2)


def _resolve_intrinsic_path(world_path: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    candidates = (
        path,
        world_path.parent / path,
        world_path.parents[1] / path,
        world_path.parents[2] / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"intrinsic file not found: {configured}")


def load_robot_world(path: str | Path) -> ThreeCameraCalibration:
    world_path = Path(path).expanduser().resolve()
    with world_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or "cameras" not in data:
        raise ValueError(f"not a robot-world calibration: {world_path}")
    models: dict[str, CameraModel] = {}
    for name in CAMERAS:
        entry = data["cameras"].get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"camera {name!r} is missing from {world_path}")
        intrinsic_path = _resolve_intrinsic_path(world_path, str(entry["intrinsic_file"]))
        with intrinsic_path.open(encoding="utf-8") as stream:
            intrinsic = yaml.safe_load(stream)
        models[name] = CameraModel(
            name=name,
            camera_matrix=intrinsic["camera_matrix"],
            distortion=intrinsic["distortion_coefficients"],
            T_base_camera_mm=entry["T_base_camera_mm"],
            image_width=int(intrinsic["image_width"]),
            image_height=int(intrinsic["image_height"]),
        )
    return ThreeCameraCalibration(models)


def workspace_contains(point_base_mm: Sequence[float], workspace: Mapping[str, Sequence[float]]) -> bool:
    point = np.asarray(point_base_mm, dtype=np.float64).reshape(3)
    return all(float(workspace[axis][0]) <= point[index] <= float(workspace[axis][1]) for index, axis in enumerate(("x", "y", "z")))

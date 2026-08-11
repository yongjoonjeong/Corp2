from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .common import (
    load_yaml,
    matrix_to_doosan_zyz_degrees,
    normalize_vector,
    transform_point,
    transform_vector,
)


_AXIS_MAP = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class MittPose:
    position_base_mm: np.ndarray
    rotation_base_tcp: np.ndarray
    doosan_posx_mm_deg: np.ndarray
    desired_surface_normal_base: np.ndarray
    punch_direction_base: np.ndarray


class RobotWorldTransformer:
    def __init__(self, calibration_path: str | Path) -> None:
        data = load_yaml(calibration_path)
        self.data = data
        self.T_base_camera: dict[str, np.ndarray] = {
            name: np.asarray(entry["T_base_camera_mm"], dtype=np.float64).reshape(4, 4)
            for name, entry in data["cameras"].items()
        }

    def camera_names(self) -> list[str]:
        return sorted(self.T_base_camera)

    def point_camera_to_base(
        self,
        camera_name: str,
        point_camera: Sequence[float],
    ) -> np.ndarray:
        return transform_point(self.T_base_camera[camera_name], point_camera)

    def vector_camera_to_base(
        self,
        camera_name: str,
        vector_camera: Sequence[float],
    ) -> np.ndarray:
        return transform_vector(self.T_base_camera[camera_name], vector_camera)

    def make_mitt_pose(
        self,
        camera_name: str,
        impact_point_camera_mm: Sequence[float],
        punch_direction_camera: Sequence[float],
        surface_normal_axis: str = "+Z",
        local_up_axis: str = "+Y",
        base_up: Sequence[float] = (0.0, 0.0, 1.0),
        stand_off_mm: float = 0.0,
    ) -> MittPose:
        if surface_normal_axis not in _AXIS_MAP or local_up_axis not in _AXIS_MAP:
            raise ValueError("Axes must be one of +X, -X, +Y, -Y, +Z, -Z")
        normal_local = _AXIS_MAP[surface_normal_axis]
        up_local_seed = _AXIS_MAP[local_up_axis]
        if abs(float(np.dot(normal_local, up_local_seed))) > 0.5:
            raise ValueError("surface_normal_axis and local_up_axis must be perpendicular")

        point_base = self.point_camera_to_base(camera_name, impact_point_camera_mm)
        punch_base = normalize_vector(
            self.vector_camera_to_base(camera_name, punch_direction_camera)
        )
        # The mitt's outward normal faces the incoming fist.
        desired_normal_base = -punch_base
        base_up_vector = normalize_vector(base_up)
        projected_up = base_up_vector - np.dot(base_up_vector, desired_normal_base) * desired_normal_base
        if np.linalg.norm(projected_up) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0])
            projected_up = fallback - np.dot(fallback, desired_normal_base) * desired_normal_base
        desired_up_base = normalize_vector(projected_up)

        local_right = normalize_vector(np.cross(up_local_seed, normal_local))
        local_up = normalize_vector(np.cross(normal_local, local_right))
        base_right = normalize_vector(np.cross(desired_up_base, desired_normal_base))
        base_up_final = normalize_vector(np.cross(desired_normal_base, base_right))
        local_basis = np.column_stack((local_right, local_up, normal_local))
        base_basis = np.column_stack((base_right, base_up_final, desired_normal_base))
        R_base_tcp = base_basis @ local_basis.T

        # Positive stand-off moves the mitt farther along the punch travel direction
        # (typically toward the robot side of the predicted impact point).
        target_position = point_base + float(stand_off_mm) * punch_base
        abc = matrix_to_doosan_zyz_degrees(R_base_tcp)
        posx = np.concatenate([target_position, abc])
        return MittPose(
            position_base_mm=target_position,
            rotation_base_tcp=R_base_tcp,
            doosan_posx_mm_deg=posx,
            desired_surface_normal_base=desired_normal_base,
            punch_direction_base=punch_base,
        )

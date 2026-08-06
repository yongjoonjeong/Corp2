from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robot_calibration.common import (
    doosan_zyz_degrees_to_matrix,
    invert_transform,
    make_transform,
    matrix_to_doosan_zyz_degrees,
    transform_error,
)
from robot_calibration.external_solver import ExternalSample, solve_robot_world_calibration


def random_transform(rng: np.random.Generator, translation_scale: float = 300.0) -> np.ndarray:
    rotation = Rotation.from_rotvec(rng.normal(size=3) * 0.7).as_matrix()
    translation = rng.uniform(-translation_scale, translation_scale, size=3)
    return make_transform(rotation, translation)


def test_joint_external_solver_recovers_transforms() -> None:
    rng = np.random.default_rng(7)
    T_flange_board_true = make_transform(
        Rotation.from_euler("xyz", [15, -25, 35], degrees=True).as_matrix(),
        [20, -35, 120],
    )
    cameras_true = {
        "front": make_transform(
            Rotation.from_euler("xyz", [170, 5, 85], degrees=True).as_matrix(),
            [-500, 200, 1100],
        ),
        "left": make_transform(
            Rotation.from_euler("xyz", [160, -20, 35], degrees=True).as_matrix(),
            [-250, -900, 900],
        ),
        "right": make_transform(
            Rotation.from_euler("xyz", [175, 20, 140], degrees=True).as_matrix(),
            [-300, 850, 950],
        ),
    }
    samples: list[ExternalSample] = []
    for camera_name, T_base_camera in cameras_true.items():
        for index in range(22):
            T_base_flange = random_transform(rng, 450.0)
            T_camera_board = (
                invert_transform(T_base_camera)
                @ T_base_flange
                @ T_flange_board_true
            )
            samples.append(
                ExternalSample(
                    camera_name=camera_name,
                    sample_id=f"{camera_name}_{index}",
                    T_base_flange_mm=T_base_flange,
                    T_camera_board_mm=T_camera_board,
                    reprojection_error_px=0.2,
                    corner_count=50,
                )
            )
    result = solve_robot_world_calibration(
        samples,
        minimum_samples_per_camera=12,
        maximum_reprojection_error_px=1.0,
        outlier_translation_mm=1.0,
        outlier_rotation_deg=0.5,
    )
    t_error, r_error = transform_error(T_flange_board_true, result.T_flange_board_mm)
    assert t_error < 1e-4
    assert r_error < 1e-5
    for name, expected in cameras_true.items():
        t_error, r_error = transform_error(expected, result.T_base_camera_mm[name])
        assert t_error < 1e-4
        assert r_error < 1e-5


def test_doosan_zyz_round_trip_matrix() -> None:
    expected = doosan_zyz_degrees_to_matrix([90.0, -90.0, 90.0])
    abc = matrix_to_doosan_zyz_degrees(expected)
    actual = doosan_zyz_degrees_to_matrix(abc)
    assert np.allclose(expected, actual, atol=1e-8)

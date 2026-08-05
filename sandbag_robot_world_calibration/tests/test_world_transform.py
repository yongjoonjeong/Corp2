from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_calibration.world_transform import RobotWorldTransformer


def test_point_vector_and_mitt_normal(tmp_path: Path) -> None:
    calibration = {
        "cameras": {
            "front": {
                "T_base_camera_mm": np.eye(4).tolist(),
            }
        }
    }
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(calibration), encoding="utf-8")
    transformer = RobotWorldTransformer(path)
    point = transformer.point_camera_to_base("front", [10, 20, 30])
    assert np.allclose(point, [10, 20, 30])
    pose = transformer.make_mitt_pose(
        "front",
        impact_point_camera_mm=[100, 200, 300],
        punch_direction_camera=[0, 0, 1],
        surface_normal_axis="+Z",
        local_up_axis="+Y",
    )
    assert np.allclose(pose.desired_surface_normal_base, [0, 0, -1])
    mapped_normal = pose.rotation_base_tcp @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(mapped_normal, [0, 0, -1], atol=1e-8)
    assert np.allclose(pose.position_base_mm, [100, 200, 300])

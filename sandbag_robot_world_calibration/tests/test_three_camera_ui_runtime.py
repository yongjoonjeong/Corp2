from collections import deque

import numpy as np

from punch_feedback_3d_core import PoseSample3D
from three_camera_punch_feedback_node import (
    alignment_status_flags,
    append_bounded_trajectory,
    build_runtime_status_payload,
    center_within_gate,
    compose_lightweight_preview,
    rate_limit_due,
    render_robot_base_3d_map,
    retained_3d_sample_for_map,
    robot_base_map_bounds,
    sanitize_trajectory_points,
    trajectory_snapshot,
)


def test_rate_limit_supports_first_run_interval_and_disabled_state() -> None:
    assert rate_limit_due(0.0, 100.0, 6.0)
    assert not rate_limit_due(100.0, 100.10, 6.0)
    assert rate_limit_due(100.0, 100.17, 6.0)
    assert not rate_limit_due(0.0, 100.0, 0.0)


def test_map_sample_ttl_allows_brief_dropout_but_never_unlocked_target() -> None:
    sample = PoseSample3D(stamp_s=10.0, landmarks={})

    assert retained_3d_sample_for_map(
        sample,
        10.0,
        10.4,
        0.4,
        target_locked=True,
    ) is sample
    assert retained_3d_sample_for_map(
        sample,
        10.0,
        10.401,
        0.4,
        target_locked=True,
    ) is None
    assert retained_3d_sample_for_map(
        sample,
        10.0,
        10.1,
        0.4,
        target_locked=False,
    ) is None


def test_center_gate_uses_front_detection_target_region() -> None:
    assert center_within_gate((0.50, 0.52))
    assert center_within_gate((0.72, 0.77))
    assert not center_within_gate((0.73, 0.52))
    assert not center_within_gate((0.50, 0.78))
    assert not center_within_gate((np.nan, 0.52))


def test_alignment_status_requires_complete_3d_sample() -> None:
    assert alignment_status_flags(
        front_detection_present=True,
        sample_3d_valid=True,
        center_gate_passed=True,
    ) == (True, True)
    assert alignment_status_flags(
        front_detection_present=True,
        sample_3d_valid=False,
        center_gate_passed=True,
    ) == (False, False)
    assert alignment_status_flags(
        front_detection_present=True,
        sample_3d_valid=True,
        center_gate_passed=False,
    ) == (True, False)


def test_bounded_trajectory_prunes_by_time_and_point_count() -> None:
    history: deque[tuple[float, np.ndarray]] = deque()
    for stamp in range(6):
        append_bounded_trajectory(
            history,
            float(stamp),
            (stamp, stamp + 1, stamp + 2),
            maximum_age_s=4.0,
            maximum_points=3,
        )
    snapshot = trajectory_snapshot(
        history,
        5.0,
        maximum_age_s=4.0,
        maximum_points=3,
    )
    np.testing.assert_array_equal(
        snapshot,
        np.asarray(((3, 4, 5), (4, 5, 6), (5, 6, 7))),
    )

    expired = trajectory_snapshot(
        history,
        10.0,
        maximum_age_s=1.5,
        maximum_points=3,
    )
    assert expired.shape == (0, 3)


def test_renderer_sanitizes_and_caps_trajectory_points() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (np.nan, 1.0, 2.0),
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
            (3.0, 3.0, 3.0),
        )
    )

    sanitized = sanitize_trajectory_points(points, 2)

    np.testing.assert_array_equal(
        sanitized,
        np.asarray(((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))),
    )


def test_lightweight_preview_contains_status_and_three_camera_strip() -> None:
    camera_views = {
        "left": np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8),
        "front": np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8),
        "right": np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8),
    }

    preview = compose_lightweight_preview(
        camera_views,
        480,
        120,
        "STATE READY | TARGET LOCKED",
    )

    assert preview.shape == (162, 480, 3)
    assert np.count_nonzero(preview[:42] != 18) > 0
    assert preview[80, 80, 0] > 200
    assert preview[80, 240, 1] > 200
    assert preview[80, 400, 2] > 200


def test_status_payload_preserves_latest_ko_ui_contract_and_3d_quality() -> None:
    payload = build_runtime_status_payload(
        pose_detected=True,
        centered=True,
        target_state="LOCKED",
        detector_state="READY",
        telemetry={"guard_frames": 6, "ready_frames": 6},
        sample_3d_valid=True,
        pose_detected_by_camera={"left": True, "front": True, "right": True},
        people_count_by_camera={"left": 2, "front": 3, "right": 1},
        sync_spread_ms=12.345,
        left_sync_offset_ms=-3.2,
        right_sync_offset_ms=5.1,
        reprojection_error_px=1.23456,
        depth_fused_joints=8,
        missing_pose_frames=0,
        sync_drop_count=4,
    )

    assert payload["pose_detected"] is True
    assert payload["centered"] is True
    assert payload["target_locked"] is True
    assert payload["detector_state"] == "READY"
    assert payload["guard_frames"] == 6
    assert payload["ready_frames"] == 6
    assert payload["sync_spread_ms"] == 12.35
    assert payload["reprojection_error_px"] == 1.235
    assert payload["pose_detected_by_camera"] == {
        "left": True,
        "front": True,
        "right": True,
    }


class _FakeCalibration:
    T_base_front_mm = np.eye(4, dtype=np.float64)

    def camera_pose_in_base(self, name: str) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = {
            "left": (-500.0, 300.0, 900.0),
            "front": (0.0, -800.0, 1000.0),
            "right": (500.0, 300.0, 900.0),
        }[name]
        return pose

    def front_point_to_base(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64)


def test_robot_base_map_renders_bounded_left_and_right_wrist_trails() -> None:
    calibration = _FakeCalibration()
    without_trails = render_robot_base_3d_map(
        calibration,
        None,
        640,
        480,
    )
    with_trails = render_robot_base_3d_map(
        calibration,
        None,
        640,
        480,
        wrist_trajectories_base_mm={
            "left": np.asarray(
                ((-250.0, 100.0, 1000.0), (-100.0, 50.0, 1100.0))
            ),
            "right": np.asarray(
                ((250.0, 100.0, 1000.0), (100.0, 50.0, 1100.0))
            ),
        },
        trajectory_point_limit=2,
    )

    assert with_trails.shape == (480, 640, 3)
    assert np.count_nonzero(with_trails != without_trails) > 0


def test_robot_base_map_keeps_static_projection_when_dynamic_cloud_moves() -> None:
    calibration = _FakeCalibration()
    bounds = robot_base_map_bounds(calibration)
    assert bounds.x_min_mm <= -2000.0
    assert bounds.x_max_mm >= 2000.0
    assert bounds.y_min_mm <= -2200.0
    assert bounds.y_max_mm >= 1200.0
    assert bounds.z_max_mm >= 2400.0

    near_cloud = render_robot_base_3d_map(
        calibration,
        None,
        640,
        480,
        np.asarray(((0.0, 0.0, 500.0),)),
        np.asarray(((1, 2, 3),), dtype=np.uint8),
    )
    distant_cloud = render_robot_base_3d_map(
        calibration,
        None,
        640,
        480,
        np.asarray(((10000.0, 10000.0, 10000.0),)),
        np.asarray(((1, 2, 3),), dtype=np.uint8),
    )

    camera_color = np.asarray((255, 180, 20), dtype=np.uint8)
    near_pixels = np.argwhere(np.all(near_cloud == camera_color, axis=2))
    distant_pixels = np.argwhere(np.all(distant_cloud == camera_color, axis=2))
    assert len(near_pixels) > 0
    np.testing.assert_array_equal(near_pixels, distant_pixels)

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from punch_feedback_core import Landmark2D, PoseSample
from punch_feedback_3d_core import (
    CameraModel,
    Landmark3D,
    LandmarkObservation2D,
    PoseSample3D,
    PunchDetector3D,
    PunchEvent3D,
    ThreeCameraCalibration,
    classify_punch_3d,
    fuse_realsense_depth,
    load_three_camera_calibration,
    score_punch_3d,
    triangulate_landmark,
)


def synthetic_calibration() -> ThreeCameraCalibration:
    camera_matrix = np.asarray(
        ((800.0, 0.0, 320.0), (0.0, 800.0, 240.0), (0.0, 0.0, 1.0))
    )
    distortion = np.zeros((5, 1), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return ThreeCameraCalibration(
        image_width=640,
        image_height=480,
        cameras={
            "front": CameraModel(
                "front", camera_matrix, distortion, identity, np.zeros((3, 1))
            ),
            # X_camera = X_front - camera_center.
            "left": CameraModel(
                "left", camera_matrix, distortion, identity, [[300.0], [0.0], [0.0]]
            ),
            "right": CameraModel(
                "right", camera_matrix, distortion, identity, [[-300.0], [0.0], [0.0]]
            ),
        },
    )


def body_to_front(body_xyz) -> np.ndarray:
    """Synthetic person: +X right, +Y up, +Z toward front camera."""
    x, y, z = np.asarray(body_xyz, dtype=np.float64)
    return np.asarray((400.0 * x, -400.0 * y, 2000.0 - 400.0 * z))


def bent_elbow(shoulder, wrist, angle_deg: float) -> np.ndarray:
    shoulder = np.asarray(shoulder, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)
    direct = wrist - shoulder
    length = float(np.linalg.norm(direct))
    midpoint = (shoulder + wrist) * 0.5
    if length <= 1e-6 or angle_deg >= 179.9:
        return midpoint
    direction = direct / length
    perpendicular = np.cross(direction, np.asarray((0.0, 1.0, 0.0)))
    if np.linalg.norm(perpendicular) <= 1e-6:
        perpendicular = np.cross(direction, np.asarray((1.0, 0.0, 0.0)))
    perpendicular /= np.linalg.norm(perpendicular)
    offset = 0.5 * length / np.tan(np.radians(angle_deg) * 0.5)
    return midpoint + perpendicular * offset


def pose_sample(
    stamp_s: float,
    strike_wrist_body,
    elbow_angle: float,
    side: str = "right",
) -> PoseSample3D:
    points = {
        "nose": body_to_front((0.0, 0.35, 0.0)),
        "left_shoulder": body_to_front((-0.5, 0.0, 0.0)),
        "right_shoulder": body_to_front((0.5, 0.0, 0.0)),
        "left_hip": body_to_front((-0.35, -1.0, 0.0)),
        "right_hip": body_to_front((0.35, -1.0, 0.0)),
        "left_wrist": body_to_front(
            strike_wrist_body if side == "left" else (-0.25, 0.25, 0.05)
        ),
        "right_wrist": body_to_front(
            strike_wrist_body if side == "right" else (0.25, 0.25, 0.05)
        ),
    }
    points["left_elbow"] = bent_elbow(
        points["left_shoulder"],
        points["left_wrist"],
        elbow_angle if side == "left" else 105.0,
    )
    points["right_elbow"] = bent_elbow(
        points["right_shoulder"],
        points["right_wrist"],
        elbow_angle if side == "right" else 105.0,
    )
    landmarks = {
        name: Landmark3D(*xyz, confidence=0.95, reprojection_error_px=0.5)
        for name, xyz in points.items()
    }
    return PoseSample3D(stamp_s=stamp_s, landmarks=landmarks)


def front_guard_pose(stamp_s: float) -> PoseSample:
    points = {
        "nose": (0.50, 0.30),
        "left_shoulder": (0.40, 0.50),
        "right_shoulder": (0.60, 0.50),
        "left_elbow": (0.38, 0.43),
        "right_elbow": (0.62, 0.43),
        "left_wrist": (0.45, 0.42),
        "right_wrist": (0.55, 0.42),
        "left_hip": (0.43, 0.78),
        "right_hip": (0.57, 0.78),
    }
    landmarks = {
        name: Landmark2D(x, y, visibility=0.95, z=0.0)
        for name, (x, y) in points.items()
    }
    return PoseSample(
        stamp_s=stamp_s,
        landmarks=landmarks,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def event_from_path(path, elbow_angles, side: str = "right") -> PunchEvent3D:
    samples = [
        pose_sample(index * 0.1, point, elbow_angles[index], side=side)
        for index, point in enumerate(path)
    ]
    return PunchEvent3D(side=side, samples=samples, max_speed=1.0)


class CalibrationAndTriangulationTests(unittest.TestCase):
    def test_triangulates_known_point_from_three_cameras(self):
        calibration = synthetic_calibration()
        expected = np.asarray((80.0, -50.0, 2000.0))
        observations = {
            name: LandmarkObservation2D(
                tuple(camera.project(expected)), confidence=0.95
            )
            for name, camera in calibration.cameras.items()
        }
        result = triangulate_landmark(calibration, observations, 2.0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.xyz_mm, expected, atol=1e-5)
        self.assertEqual(set(result.cameras), {"left", "front", "right"})

    def test_rejects_one_camera_outlier(self):
        calibration = synthetic_calibration()
        expected = np.asarray((80.0, -50.0, 2000.0))
        observations = {
            name: LandmarkObservation2D(
                tuple(camera.project(expected)), confidence=0.95
            )
            for name, camera in calibration.cameras.items()
        }
        right = observations["right"]
        observations["right"] = LandmarkObservation2D(
            (right.pixel[0] + 120.0, right.pixel[1] - 80.0), 0.95
        )
        result = triangulate_landmark(calibration, observations, 4.0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.xyz_mm, expected, atol=1e-4)
        self.assertEqual(set(result.cameras), {"left", "front"})

    def test_loads_calibrator_npz_format(self):
        calibration = synthetic_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three_camera_charuco_calibration.npz"
            np.savez(
                path,
                image_width=640,
                image_height=480,
                K_left=calibration.cameras["left"].camera_matrix,
                D_left=calibration.cameras["left"].distortion,
                K_front=calibration.cameras["front"].camera_matrix,
                D_front=calibration.cameras["front"].distortion,
                K_right=calibration.cameras["right"].camera_matrix,
                D_right=calibration.cameras["right"].distortion,
                R_front_to_left=np.eye(3),
                T_front_to_left=[[300.0], [0.0], [0.0]],
                R_front_to_right=np.eye(3),
                T_front_to_right=[[-300.0], [0.0], [0.0]],
            )
            loaded = load_three_camera_calibration(path)
        self.assertEqual((loaded.image_width, loaded.image_height), (640, 480))
        np.testing.assert_allclose(
            loaded.cameras["right"].translation_front_to_camera_mm,
            [[-300.0], [0.0], [0.0]],
        )

    def test_loads_robot_world_yaml_and_transforms_to_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intrinsics = root / "intrinsics"
            intrinsics.mkdir()
            camera_matrix = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
            for name in ("front", "left", "right"):
                (intrinsics / f"{name}.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "image_width": 640,
                            "image_height": 480,
                            "camera_matrix": camera_matrix,
                            "distortion_coefficients": [0.0] * 5,
                        }
                    ),
                    encoding="utf-8",
                )
            T_base_front = np.eye(4)
            T_base_front[:3, 3] = (100.0, 200.0, 300.0)
            entries = {}
            for name, center_x in (("front", 0.0), ("left", -300.0), ("right", 300.0)):
                transform = np.eye(4)
                transform[:3, 3] = T_base_front[:3, 3] + (center_x, 0.0, 0.0)
                entries[name] = {
                    "T_base_camera_mm": transform.tolist(),
                    "intrinsic_file": str(intrinsics / f"{name}.yaml"),
                }
            path = root / "robot_world.yaml"
            path.write_text(yaml.safe_dump({"cameras": entries}), encoding="utf-8")

            loaded = load_three_camera_calibration(path)

        np.testing.assert_allclose(
            loaded.cameras["left"].translation_front_to_camera_mm.reshape(3),
            (300.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            loaded.cameras["right"].translation_front_to_camera_mm.reshape(3),
            (-300.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            loaded.front_point_to_base((10.0, 20.0, 30.0)),
            (110.0, 220.0, 330.0),
        )

    def test_depth_fusion_only_accepts_consistent_depth(self):
        triangulated = np.asarray((100.0, 20.0, 2000.0))
        fused, used = fuse_realsense_depth(
            triangulated, np.asarray((110.0, 10.0, 2020.0)), 0.25, 100.0
        )
        self.assertTrue(used)
        np.testing.assert_allclose(fused, (102.5, 17.5, 2005.0))
        unchanged, used = fuse_realsense_depth(
            triangulated, np.asarray((900.0, 20.0, 2000.0)), 0.25, 100.0
        )
        self.assertFalse(used)
        np.testing.assert_allclose(unchanged, triangulated)


class Classification3DTests(unittest.TestCase):
    def test_forward_linear_extension_is_straight(self):
        event = event_from_path(
            ((0.35, 0.15, 0.05), (0.37, 0.15, 0.30), (0.39, 0.16, 0.62), (0.40, 0.16, 0.95)),
            (100.0, 125.0, 150.0, 168.0),
        )
        result = classify_punch_3d(event, {})
        self.assertEqual(result.punch_type, "straight")
        self.assertGreater(result.motion_features["forward_component_ratio"], 0.9)

    def test_lateral_curved_bent_arm_path_is_hook(self):
        event = event_from_path(
            ((0.35, 0.10, 0.05), (0.72, 0.11, 0.20), (0.92, 0.12, 0.43), (0.70, 0.12, 0.68), (0.36, 0.12, 0.78)),
            (98.0, 100.0, 102.0, 104.0, 106.0),
        )
        result = classify_punch_3d(event, {})
        self.assertEqual(result.punch_type, "hook")
        self.assertGreater(result.motion_features["path_curvature_ratio"], 0.16)
        self.assertGreater(result.motion_features["hook_outward_travel_ratio"], 0.5)
        self.assertGreater(result.motion_features["hook_inward_return_ratio"], 0.5)

    def test_upward_bent_arm_path_is_uppercut(self):
        event = event_from_path(
            ((0.34, -0.18, 0.05), (0.36, 0.02, 0.14), (0.37, 0.32, 0.25), (0.38, 0.70, 0.36)),
            (82.0, 88.0, 96.0, 105.0),
        )
        result = classify_punch_3d(event, {})
        self.assertEqual(result.punch_type, "uppercut")
        self.assertGreater(result.motion_features["upward_component_ratio"], 0.8)

    def test_hook_uses_late_contact_instead_of_generic_chamber_peak(self):
        event = event_from_path(
            (
                (0.35, 0.10, 0.05),
                (0.55, 0.10, 0.25),
                (0.85, 0.11, 0.50),
                (1.15, 0.11, 0.78),
                (0.95, 0.12, 0.82),
                (0.65, 0.12, 0.84),
                (0.32, 0.12, 0.86),
                (0.34, 0.11, 0.20),
            ),
            (100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 105.0),
        )
        result = classify_punch_3d(event, {})
        features = result.motion_features

        self.assertEqual(result.punch_type, "hook")
        self.assertEqual(result.classification_reason, "hook_3d_outside_to_inside_contact")
        self.assertEqual(features["hook_ordered_sweep_pass"], 1.0)
        self.assertLess(
            features["generic_impact_sample_index"],
            features["hook_impact_sample_index"],
        )
        self.assertEqual(features["impact_sample_index"], 6.0)
        self.assertIs(result.key_sample, event.samples[6])

    def test_moderate_ordered_hook_still_passes(self):
        event = event_from_path(
            (
                (0.35, 0.10, 0.05),
                (0.50, 0.10, 0.20),
                (0.70, 0.11, 0.38),
                (0.85, 0.11, 0.56),
                (0.75, 0.12, 0.64),
                (0.55, 0.12, 0.70),
                (0.38, 0.12, 0.74),
            ),
            (98.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0),
        )
        result = classify_punch_3d(event, {})

        self.assertEqual(result.punch_type, "hook")
        self.assertEqual(result.motion_features["hook_ordered_sweep_pass"], 1.0)
        self.assertGreaterEqual(
            result.motion_features["hook_inward_sweep_ratio"], 0.35
        )

    def test_monotonic_and_wide_forward_paths_remain_straight(self):
        paths = (
            # Monotonic movement from an outside guard toward body centre is
            # not an outside chamber followed by an inward Hook.
            (
                (0.45, 0.15, 0.05),
                (0.35, 0.15, 0.20),
                (0.25, 0.15, 0.40),
                (0.15, 0.15, 0.62),
                (0.05, 0.16, 0.82),
                (-0.05, 0.16, 1.00),
            ),
            # A straight aimed off-centre has large X travel but no reversal.
            (
                (0.35, 0.15, 0.05),
                (0.48, 0.15, 0.20),
                (0.61, 0.15, 0.40),
                (0.74, 0.15, 0.62),
                (0.87, 0.16, 0.82),
                (1.00, 0.16, 1.00),
            ),
        )
        for path in paths:
            with self.subTest(path=path[-1]):
                result = classify_punch_3d(
                    event_from_path(path, (100.0, 118.0, 136.0, 150.0, 160.0, 168.0)),
                    {},
                )
                self.assertEqual(result.punch_type, "straight")
                self.assertEqual(
                    result.motion_features["hook_ordered_sweep_pass"], 0.0
                )

    def test_endpoint_wrist_jitter_does_not_move_straight_impact(self):
        event = event_from_path(
            (
                (0.35, 0.15, 0.05),
                (0.36, 0.15, 0.20),
                (0.37, 0.15, 0.40),
                (0.38, 0.15, 0.62),
                (0.39, 0.15, 0.78),
                (0.40, 0.16, 0.95),
                (1.80, -1.20, -0.40),
            ),
            (100.0, 118.0, 136.0, 150.0, 160.0, 168.0, 40.0),
        )
        result = classify_punch_3d(event, {})

        self.assertEqual(result.punch_type, "straight")
        self.assertLess(result.motion_features["impact_sample_index"], 6.0)

    def test_current_shoulder_translation_is_removed_in_guard_frame(self):
        path = (
            (0.35, 0.15, 0.05),
            (0.36, 0.15, 0.20),
            (0.37, 0.15, 0.40),
            (0.38, 0.15, 0.62),
            (0.39, 0.16, 0.82),
            (0.40, 0.16, 1.00),
        )
        angles = (100.0, 118.0, 136.0, 150.0, 160.0, 168.0)
        stationary = event_from_path(path, angles)
        translated_samples = []
        for index, sample in enumerate(stationary.samples):
            translation = np.asarray((35.0 * index, -18.0 * index, 22.0 * index))
            translated_samples.append(
                PoseSample3D(
                    stamp_s=sample.stamp_s,
                    landmarks={
                        name: Landmark3D(
                            *(landmark.xyz + translation),
                            confidence=landmark.confidence,
                            reprojection_error_px=landmark.reprojection_error_px,
                            camera_count=landmark.camera_count,
                        )
                        for name, landmark in sample.landmarks.items()
                    },
                )
            )
        translated = PunchEvent3D("right", translated_samples, 1.0)

        reference = classify_punch_3d(stationary, {})
        result = classify_punch_3d(translated, {})

        self.assertEqual(result.punch_type, "straight")
        self.assertEqual(result.punch_type, reference.punch_type)
        self.assertAlmostEqual(
            result.motion_features["forward_travel_ratio"],
            reference.motion_features["forward_travel_ratio"],
            places=6,
        )
        self.assertEqual(result.motion_features["body_frame_guard_locked"], 1.0)

    def test_uppercut_uses_low_chamber_to_apex_contact(self):
        event = event_from_path(
            (
                (0.34, 0.15, 0.05),
                (0.35, -0.10, 0.10),
                (0.36, -0.55, 0.15),
                (0.37, -0.52, 0.16),
                (0.38, -0.05, 0.15),
                (0.39, 0.30, 0.15),
                (0.40, 0.60, 0.15),
                (0.39, 0.58, 0.14),
                (0.38, 0.20, 0.10),
            ),
            (100.0, 98.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0),
        )
        result = classify_punch_3d(event, {})
        features = result.motion_features

        self.assertEqual(result.punch_type, "uppercut")
        self.assertEqual(features["uppercut_strict_pass"], 1.0)
        self.assertLess(
            features["generic_impact_sample_index"],
            features["uppercut_impact_sample_index"],
        )
        self.assertEqual(features["impact_sample_index"], 6.0)
        self.assertIs(result.key_sample, event.samples[6])

    def test_left_hook_is_mirror_of_right_hook(self):
        right_path = (
            (0.35, 0.10, 0.05),
            (0.55, 0.10, 0.20),
            (0.78, 0.11, 0.38),
            (0.92, 0.11, 0.56),
            (0.78, 0.12, 0.64),
            (0.55, 0.12, 0.70),
            (0.32, 0.12, 0.74),
        )
        left_path = tuple((-x, y, z) for x, y, z in right_path)
        angles = (98.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0)

        right = classify_punch_3d(event_from_path(right_path, angles), {})
        left = classify_punch_3d(
            event_from_path(left_path, angles, side="left"),
            {},
        )

        self.assertEqual((right.punch_type, left.punch_type), ("hook", "hook"))
        self.assertEqual(right.motion_features["hook_ordered_sweep_pass"], 1.0)
        self.assertEqual(left.motion_features["hook_ordered_sweep_pass"], 1.0)
        self.assertAlmostEqual(
            right.motion_features["hook_inward_sweep_ratio"],
            left.motion_features["hook_inward_sweep_ratio"],
            places=6,
        )

    def test_temporary_yaml_scores_every_3d_class(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "temporary_form_reference_3d.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            reference = yaml.safe_load(stream)
        events = (
            event_from_path(
                (
                    (0.35, 0.15, 0.05),
                    (0.37, 0.15, 0.30),
                    (0.39, 0.16, 0.62),
                    (0.40, 0.16, 0.95),
                ),
                (100.0, 125.0, 150.0, 168.0),
            ),
            event_from_path(
                (
                    (0.35, 0.10, 0.05),
                    (0.72, 0.11, 0.20),
                    (0.92, 0.12, 0.43),
                    (0.70, 0.12, 0.68),
                    (0.36, 0.12, 0.78),
                ),
                (98.0, 100.0, 102.0, 104.0, 106.0),
            ),
            event_from_path(
                (
                    (0.34, -0.18, 0.05),
                    (0.36, 0.02, 0.14),
                    (0.37, 0.32, 0.25),
                    (0.38, 0.70, 0.36),
                ),
                (82.0, 88.0, 96.0, 105.0),
            ),
        )
        self.assertEqual(
            {
                classify_punch_3d(event, reference["classification"]).punch_type
                for event in events
            },
            {"straight", "hook", "uppercut"},
        )
        for event in events:
            classified = classify_punch_3d(event, reference["classification"])
            score = score_punch_3d(
                classified, reference["profiles"], reference["feedback"]
            )
            self.assertGreaterEqual(score.total_score, 0.0)
            self.assertLessEqual(score.total_score, 100.0)


class SlowDetector3DTests(unittest.TestCase):
    def test_front_2d_guard_ignores_noisy_3d_depth(self):
        detector = PunchDetector3D(
            {
                "ready_frames": 3,
                "guard_speed": 0.05,
                "guard_min_visibility": 0.50,
                "guard_max_wrist_face_ratio": 1.50,
                "speed_window": 1,
            }
        )
        for index, depth in enumerate((0.05, 0.45, -0.25, 0.55)):
            stamp = index * 0.2
            sample = pose_sample(stamp, (0.35, 0.15, depth), 100.0)
            sample.front_pose = front_guard_pose(stamp)
            detector.update(sample)

        self.assertEqual(detector.state, "READY")
        self.assertEqual(detector.telemetry["guard_source_2d"], 1.0)
        self.assertLessEqual(detector.telemetry["right_guard_speed"], 0.05)
        self.assertGreater(detector.telemetry["right_speed"], 0.05)

    def test_slow_displacement_and_extension_can_start_punch(self):
        detector = PunchDetector3D(
            {
                "ready_frames": 3,
                "guard_speed": 0.05,
                "start_speed": 99.0,
                "slow_start_displacement_ratio": 0.20,
                "slow_start_extension_deg": 10.0,
                "start_confirm_frames": 2,
                "speed_window": 1,
                "end_speed": 0.05,
                "end_frames": 2,
                "min_duration_s": 0.1,
            }
        )
        stamp = 0.0
        for _ in range(5):
            detector.update(pose_sample(stamp, (0.35, 0.15, 0.05), 100.0))
            stamp += 0.2
        self.assertEqual(detector.state, "READY")

        # Deliberately below the configured start speed; accumulated motion starts it.
        detector.update(pose_sample(stamp, (0.36, 0.15, 0.28), 115.0))
        stamp += 0.5
        detector.update(pose_sample(stamp, (0.37, 0.15, 0.34), 118.0))
        stamp += 0.5
        self.assertEqual(detector.state, "ACTIVE_RIGHT")

        event = None
        for _ in range(3):
            event = detector.update(pose_sample(stamp, (0.37, 0.15, 0.34), 118.0)) or event
            stamp += 0.2
        self.assertIsNotNone(event)
        self.assertEqual(event.side, "right")

    def test_ready_baseline_does_not_follow_a_gradual_punch(self):
        detector = PunchDetector3D(
            {
                "ready_frames": 2,
                "guard_speed": 0.10,
                "start_speed": 99.0,
                "slow_start_displacement_ratio": 0.20,
                "slow_start_extension_deg": 10.0,
                "start_confirm_frames": 2,
                "speed_window": 1,
            }
        )
        stamp = 0.0
        for _ in range(4):
            detector.update(pose_sample(stamp, (0.35, 0.15, 0.05), 100.0))
            stamp += 0.5
        self.assertEqual(detector.state, "READY")

        for step in range(1, 8):
            detector.update(
                pose_sample(
                    stamp,
                    (0.35, 0.15, 0.05 + step * 0.04),
                    100.0 + step * 2.0,
                )
            )
            stamp += 0.5
        self.assertEqual(detector.state, "ACTIVE_RIGHT")

    def test_slow_bent_arm_hook_can_start_without_elbow_extension(self):
        detector = PunchDetector3D(
            {
                "ready_frames": 2,
                "guard_speed": 0.05,
                "start_speed": 0.50,
                "slow_start_speed": 0.06,
                "slow_start_displacement_ratio": 0.20,
                "slow_start_extension_deg": 10.0,
                "start_confirm_frames": 2,
                "speed_window": 1,
            }
        )
        stamp = 0.0
        for _ in range(4):
            detector.update(pose_sample(stamp, (0.35, 0.15, 0.05), 100.0))
            stamp += 0.5
        for step in range(1, 7):
            detector.update(
                pose_sample(stamp, (0.35 + step * 0.05, 0.15, 0.05), 100.0)
            )
            stamp += 0.5
        self.assertEqual(detector.state, "ACTIVE_RIGHT")


if __name__ == "__main__":
    unittest.main()

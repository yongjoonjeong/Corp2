import unittest

from punch_feedback_core import (
    ClassifiedPunch,
    Landmark2D,
    PoseSample,
    PunchDetector,
    PunchEvent,
    classify_punch,
    extract_form_features,
    normalized_wrist_speed,
    score_punch,
)


def make_sample(stamp_s, overrides=None, depths=None):
    points = {
        "nose": (0.50, 0.20),
        "left_shoulder": (0.40, 0.40),
        "right_shoulder": (0.60, 0.40),
        "left_elbow": (0.35, 0.50),
        "right_elbow": (0.65, 0.50),
        "left_wrist": (0.45, 0.28),
        "right_wrist": (0.62, 0.58),
        "left_hip": (0.44, 0.70),
        "right_hip": (0.56, 0.70),
    }
    points.update(overrides or {})
    depths = depths or {}
    return PoseSample(
        stamp_s=stamp_s,
        landmarks={
            name: Landmark2D(
                x=x,
                y=y,
                visibility=1.0,
                z=float(depths.get(name, 0.0)),
            )
            for name, (x, y) in points.items()
        },
    )


class PunchDetectorTests(unittest.TestCase):
    def test_whole_body_translation_does_not_create_wrist_speed(self):
        first = make_sample(0.0)
        translated = {
            name: (landmark.x + 0.10, landmark.y + 0.08)
            for name, landmark in first.landmarks.items()
        }
        second = make_sample(0.04, translated)

        self.assertAlmostEqual(
            normalized_wrist_speed(first, second, "right"),
            0.0,
            places=6,
        )

    def test_whole_body_camera_approach_does_not_create_wrist_speed(self):
        first = make_sample(0.0)
        image_center = (0.50, 0.50)
        scale = 1.25
        approached = {
            name: (
                image_center[0] + (landmark.x - image_center[0]) * scale,
                image_center[1] + (landmark.y - image_center[1]) * scale,
            )
            for name, landmark in first.landmarks.items()
        }
        second = make_sample(0.04, approached)

        self.assertAlmostEqual(
            normalized_wrist_speed(first, second, "right"),
            0.0,
            places=6,
        )

    def test_emits_one_right_hand_event_after_speed_burst(self):
        detector = PunchDetector(
            {
                "start_speed": 1.0,
                "guard_speed": 0.4,
                "end_speed": 0.4,
                "ready_frames": 3,
                "end_frames": 3,
                "min_duration_s": 0.05,
                "max_duration_s": 1.0,
                "cooldown_s": 0.2,
                "min_visibility": 0.5,
                "speed_window": 1,
                "start_displacement_ratio": 0.1,
                "start_extension_deg": 10.0,
                "start_confirm_frames": 1,
                "guard_max_wrist_face_ratio": 3.0,
            }
        )

        samples = [make_sample(index * 0.04) for index in range(5)]
        samples.extend(
            [
                make_sample(0.20, {"right_wrist": (0.70, 0.55)}),
                make_sample(0.24, {"right_wrist": (0.82, 0.48)}),
                make_sample(0.28, {"right_wrist": (0.88, 0.43)}),
                make_sample(0.32, {"right_wrist": (0.88, 0.43)}),
                make_sample(0.36, {"right_wrist": (0.88, 0.43)}),
                make_sample(0.40, {"right_wrist": (0.88, 0.43)}),
            ]
        )

        events = [event for sample in samples if (event := detector.update(sample))]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].side, "right")
        self.assertGreater(events[0].max_speed, 1.0)

    def test_ready_is_latched_during_gradual_acceleration(self):
        detector = PunchDetector(
            {
                "start_speed": 1.0,
                "guard_speed": 0.35,
                "end_speed": 0.45,
                "ready_frames": 3,
                "end_frames": 3,
                "min_duration_s": 0.05,
                "max_duration_s": 1.0,
                "cooldown_s": 0.2,
                "min_visibility": 0.5,
                "speed_window": 1,
                "start_displacement_ratio": 0.1,
                "start_extension_deg": 10.0,
                "start_confirm_frames": 1,
                "guard_max_wrist_face_ratio": 3.0,
            }
        )

        for index in range(5):
            detector.update(make_sample(index * 0.04))
        self.assertEqual(detector.state, "READY")

        # Faster than guard_speed but slower than start_speed. READY must remain latched.
        detector.update(make_sample(0.20, {"right_wrist": (0.6248, 0.58)}))
        self.assertEqual(detector.state, "READY")

        detector.update(make_sample(0.24, {"right_wrist": (0.66, 0.55)}))
        self.assertEqual(detector.state, "ACTIVE_RIGHT")

    def test_three_frame_median_rejects_single_speed_spike(self):
        detector = PunchDetector(
            {
                "start_speed": 1.0,
                "guard_speed": 0.35,
                "end_speed": 0.45,
                "ready_frames": 3,
                "speed_window": 3,
                "start_displacement_ratio": 0.1,
                "start_confirm_frames": 1,
                "guard_max_wrist_face_ratio": 3.0,
            }
        )

        for index in range(5):
            detector.update(make_sample(index * 0.04))
        self.assertEqual(detector.state, "READY")

        detector.update(make_sample(0.20, {"right_wrist": (0.72, 0.55)}))
        self.assertEqual(detector.state, "READY")
        self.assertLess(detector.telemetry["right_speed"], 1.0)


class ClassificationTests(unittest.TestCase):
    def test_clear_elbow_extension_is_straight(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.62, 0.58),
            },
        )
        extended = make_sample(
            0.2,
            {
                "right_elbow": (0.75, 0.40),
                "right_wrist": (0.90, 0.40),
            },
        )
        event = PunchEvent("right", [start, extended], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "straight_min_peak_elbow_angle": 150.0,
                "straight_min_extension_gain_deg": 25.0,
            },
        )
        self.assertEqual(classified.punch_type, "straight")
        self.assertEqual(
            classified.classification_reason,
            "straight_linear_outbound_path",
        )
        self.assertEqual(
            classified.motion_features["straight_extension_pass"],
            1.0,
        )

    def test_outside_to_inside_lateral_arc_is_hook(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.70, 0.65),
            },
        )
        outside = make_sample(
            0.1,
            {
                "right_elbow": (0.72, 0.50),
                "right_wrist": (0.85, 0.52),
            },
        )
        inside = make_sample(
            0.2,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.45, 0.50),
            },
        )
        event = PunchEvent("right", [start, outside, inside], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "uppercut_min_upward_ratio": 0.30,
                "uppercut_max_elbow_angle": 145.0,
                "uppercut_vertical_dominance": 1.05,
                "uppercut_min_wrist_above_elbow_ratio": 0.10,
                "hook_min_lateral_ratio": 0.35,
                "hook_min_inward_ratio": 0.15,
                "hook_lateral_dominance": 1.0,
                "hook_max_elbow_angle": 145.0,
                "hook_max_wrist_elbow_height_ratio": 0.35,
                "hook_max_extension_gain_deg": 70.0,
            },
        )
        self.assertEqual(classified.punch_type, "hook")
        self.assertEqual(
            classified.classification_reason,
            "hook_outside_to_inside",
        )
        self.assertIs(classified.key_sample, inside)
        self.assertGreater(classified.motion_features["inward_ratio"], 0.15)
        self.assertGreater(
            classified.motion_features["lateral_ratio"],
            classified.motion_features["upward_ratio"],
        )

    def test_forward_depth_and_partial_extension_take_straight_priority(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.62, 0.58),
            },
        )
        outside = make_sample(
            0.1,
            {
                # Keep projected extension below the relaxed 2D threshold so
                # this test specifically exercises the depth path.
                "right_elbow": (0.655, 0.495),
                "right_wrist": (0.638, 0.575),
            },
            {"right_wrist": -0.05},
        )
        impact = make_sample(
            0.2,
            {
                "right_elbow": (0.66, 0.49),
                "right_wrist": (0.655, 0.57),
            },
            {"right_wrist": -0.10},
        )
        event = PunchEvent("right", [start, outside, impact], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "straight_min_peak_elbow_angle": 140.0,
                "straight_min_extension_gain_deg": 15.0,
                "straight_min_depth_gain_ratio": 0.18,
                "straight_depth_min_peak_elbow_angle": 125.0,
                "straight_depth_min_extension_gain_deg": 8.0,
                "hook_min_lateral_ratio": 0.35,
                "hook_min_inward_ratio": 0.15,
                "hook_lateral_dominance": 1.15,
                "hook_max_elbow_angle": 145.0,
            },
        )
        self.assertEqual(classified.punch_type, "straight")
        self.assertEqual(
            classified.classification_reason,
            "straight_forward_depth_and_extension",
        )
        self.assertEqual(classified.motion_features["straight_depth_pass"], 1.0)
        self.assertEqual(
            classified.motion_features["straight_2d_extension_pass"],
            0.0,
        )
        self.assertGreater(
            classified.motion_features["forward_depth_gain_ratio"],
            0.18,
        )

    def test_curved_path_overrides_strong_elbow_extension_as_hook(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.62, 0.58),
            },
        )
        arc = make_sample(
            0.1,
            {
                "right_elbow": (0.70, 0.43),
                "right_wrist": (0.70, 0.35),
            },
        )
        impact = make_sample(
            0.2,
            {
                "right_elbow": (0.75, 0.48),
                "right_wrist": (0.90, 0.50),
            },
        )
        event = PunchEvent("right", [start, arc, impact], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "hook_min_lateral_ratio": 0.38,
                "hook_lateral_dominance": 1.20,
                "hook_strong_arc_min_curvature_ratio": 0.25,
                "hook_strong_arc_max_linearity": 0.82,
                "hook_strong_arc_max_elbow_angle": 160.0,
                "straight_priority_peak_elbow_angle": 150.0,
                "straight_priority_extension_gain_deg": 25.0,
            },
        )
        self.assertEqual(classified.punch_type, "hook")
        self.assertEqual(
            classified.classification_reason,
            "hook_curved_outbound_path",
        )
        self.assertEqual(classified.motion_features["hook_priority_pass"], 1.0)
        self.assertGreater(
            classified.motion_features["path_curvature_ratio"],
            0.25,
        )

    def test_return_to_guard_is_ignored_for_straight_classification(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.62, 0.58),
            },
        )
        outbound = make_sample(
            0.1,
            {
                "right_elbow": (0.68, 0.45),
                "right_wrist": (0.75, 0.49),
            },
        )
        impact = make_sample(
            0.2,
            {
                "right_elbow": (0.75, 0.40),
                "right_wrist": (0.90, 0.40),
            },
        )
        recovery = make_sample(
            0.3,
            {
                "right_elbow": (0.67, 0.48),
                "right_wrist": (0.70, 0.55),
            },
        )
        guard = make_sample(0.4)
        event = PunchEvent(
            "right",
            [start, outbound, impact, recovery, guard],
            max_speed=4.0,
        )
        classified = classify_punch(event, {})

        self.assertEqual(classified.punch_type, "straight")
        self.assertEqual(
            classified.classification_reason,
            "straight_linear_outbound_path",
        )
        self.assertEqual(classified.motion_features["inward_ratio"], 0.0)
        self.assertGreaterEqual(
            classified.motion_features["recovery_frames_ignored"],
            2.0,
        )
        self.assertEqual(classified.motion_features["hook_strict_pass"], 0.0)

    def test_low_planar_motion_with_partial_extension_is_straight(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.65, 0.50),
                "right_wrist": (0.62, 0.58),
            },
        )
        impact = make_sample(
            0.2,
            {
                "right_elbow": (0.66, 0.49),
                "right_wrist": (0.655, 0.57),
            },
        )
        event = PunchEvent("right", [start, impact], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "straight_min_peak_elbow_angle": 138.0,
                "straight_min_extension_gain_deg": 12.0,
                "straight_max_planar_travel_ratio": 0.32,
                "straight_planar_min_peak_elbow_angle": 132.0,
                "straight_planar_min_extension_gain_deg": 8.0,
            },
        )
        self.assertEqual(classified.punch_type, "straight")
        self.assertEqual(
            classified.classification_reason,
            "straight_low_planar_motion_and_extension",
        )
        self.assertEqual(
            classified.motion_features["straight_low_planar_pass"],
            1.0,
        )
        self.assertEqual(
            classified.motion_features["straight_2d_extension_pass"],
            0.0,
        )
        self.assertLess(
            classified.motion_features["planar_travel_ratio"],
            0.32,
        )

    def test_marginal_lateral_arc_no_longer_passes_hook_fallback(self):
        start = make_sample(
            0.0,
            {
                "right_elbow": (0.72, 0.45),
                "right_wrist": (0.70, 0.55),
            },
        )
        outside = make_sample(
            0.1,
            {
                "right_elbow": (0.74, 0.42),
                "right_wrist": (0.762, 0.53),
            },
        )
        inside = make_sample(
            0.2,
            {
                "right_elbow": (0.73, 0.42),
                "right_wrist": (0.742, 0.53),
            },
        )
        event = PunchEvent("right", [start, outside, inside], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "hook_min_lateral_ratio": 0.38,
                "hook_min_inward_ratio": 0.18,
                "hook_lateral_dominance": 1.20,
                "hook_max_elbow_angle": 142.0,
                "hook_fallback_lateral_scale": 0.85,
                "hook_fallback_inward_scale": 0.75,
                "hook_fallback_lateral_dominance": 1.05,
                "hook_fallback_elbow_slack_deg": 5.0,
                "straight_max_planar_travel_ratio": 0.32,
            },
        )
        self.assertNotEqual(classified.punch_type, "hook")
        self.assertEqual(classified.motion_features["hook_strict_pass"], 0.0)
        self.assertEqual(classified.motion_features["hook_fallback_pass"], 0.0)
        self.assertGreater(
            classified.motion_features["planar_travel_ratio"],
            0.32,
        )

    def test_upward_bent_arm_motion_is_uppercut(self):
        start = make_sample(0.0, {"right_wrist": (0.62, 0.65)})
        key = make_sample(
            0.1,
            {
                "right_elbow": (0.62, 0.50),
                "right_wrist": (0.62, 0.30),
            },
        )
        event = PunchEvent("right", [start, key], max_speed=4.0)
        classified = classify_punch(
            event,
            {
                "uppercut_min_upward_ratio": 0.30,
                "uppercut_max_elbow_angle": 145.0,
                "hook_min_lateral_ratio": 0.45,
                "hook_max_elbow_angle": 145.0,
            },
        )
        self.assertEqual(classified.punch_type, "uppercut")
        self.assertEqual(
            classified.classification_reason,
            "uppercut_vertical_dominant",
        )
        self.assertIs(classified.key_sample, key)


class ScoringTests(unittest.TestCase):
    def test_exact_reference_scores_100(self):
        sample = make_sample(0.0)
        actual_angle = extract_form_features(sample, "right")[
            "strike_elbow_angle_deg"
        ]
        classified = ClassifiedPunch(
            punch_type="straight",
            side="right",
            confidence=0.8,
            key_sample=sample,
            motion_features={},
        )
        profiles = {
            "straight": {
                "features": {
                    "strike_elbow_angle_deg": {
                        "target": actual_angle,
                        "tolerance": 20.0,
                        "weight": 1.0,
                        "joint": "strike_elbow",
                        "code": "elbow_not_extended",
                    }
                }
            }
        }
        result = score_punch(
            classified,
            profiles,
            {"score_threshold": 75.0, "joint_error_threshold": 1.0},
        )
        self.assertAlmostEqual(result.total_score, 100.0)
        self.assertFalse(result.feedback_required)

    def test_large_error_requests_feedback(self):
        sample = make_sample(0.0)
        classified = ClassifiedPunch(
            punch_type="straight",
            side="right",
            confidence=0.8,
            key_sample=sample,
            motion_features={},
        )
        profiles = {
            "straight": {
                "features": {
                    "guard_to_face_ratio": {
                        "target": 0.0,
                        "tolerance": 0.1,
                        "weight": 1.0,
                        "joint": "guard_wrist",
                        "code": "guard_dropped",
                    }
                }
            }
        }
        result = score_punch(
            classified,
            profiles,
            {"score_threshold": 75.0, "joint_error_threshold": 1.0},
        )
        self.assertLess(result.total_score, 75.0)
        self.assertTrue(result.feedback_required)
        self.assertEqual(result.violations[0].code, "guard_dropped")


if __name__ == "__main__":
    unittest.main()

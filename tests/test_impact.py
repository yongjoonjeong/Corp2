from sandbag_vision.impact import FrontImpactDetector2D
from sandbag_vision.types import Landmark2D, PoseSample


def pose(stamp_ns: int, left_wrist_x: float, left_elbow_x: float = 205.0) -> PoseSample:
    points = {
        "nose": (250.0, 210.0),
        "left_shoulder": (200.0, 300.0),
        "right_shoulder": (300.0, 300.0),
        "left_elbow": (left_elbow_x, 270.0),
        "right_elbow": (290.0, 270.0),
        "left_wrist": (left_wrist_x, 240.0),
        "right_wrist": (280.0, 240.0),
        "left_hip": (215.0, 400.0),
        "right_hip": (285.0, 400.0),
    }
    return PoseSample(
        "front",
        stamp_ns,
        stamp_ns,
        (640, 480),
        (0, 0, 640, 480),
        {name: Landmark2D(pixel, 0.95) for name, pixel in points.items()},
        2.0,
    )


def test_contact_is_reported_at_reach_peak_not_early_path() -> None:
    detector = FrontImpactDetector2D(
        {
            "minimum_landmark_confidence": 0.3,
            "guard_max_wrist_nose_ratio": 1.85,
            "guard_max_speed_body_s": 0.6,
            "guard_ready_frames": 3,
            "start_speed_body_s": 0.8,
            "start_displacement_body": 0.08,
            "start_extension_gain_deg": 4.0,
            "start_confirm_frames": 1,
            "minimum_duration_s": 0.06,
            "maximum_duration_s": 1.2,
            "minimum_contact_displacement_body": 0.12,
            "minimum_contact_extension_gain_deg": 8.0,
            "minimum_contact_displacement_without_extension_body": 0.30,
            "minimum_peak_speed_body_s": 0.8,
            "deceleration_ratio": 0.65,
            "reversal_cosine": 0.1,
            "contact_settle_frames": 2,
            "impact_display_s": 0.12,
            "cooldown_s": 0.5,
            "max_lost_frames": 5,
            "fist_extension_forearm_ratio": 0.35,
            "require_mitt_roi": True,
            "mitt_roi_normalized": [0.60, 0.44, 0.65, 0.52],
        }
    )
    stamp = 1_000_000_000
    for _ in range(4):
        assert detector.update(pose(stamp, 220.0)) is None
        stamp += 40_000_000
    assert detector.state == "READY"
    event = None
    trajectory = [235.0, 270.0, 315.0, 350.0, 350.0, 342.0, 335.0]
    stamps = []
    for index, x in enumerate(trajectory):
        stamps.append(stamp)
        event = detector.update(pose(stamp, x, 210.0 + index * 3.0)) or event
        stamp += 40_000_000
    assert event is not None
    assert event.side == "left"
    assert event.stamp_ns == stamps[3]
    assert abs(event.wrist_pixel_front[0] - 350.0) < 1e-6
    assert event.metadata["contact_pixel_front_x"] > event.wrist_pixel_front[0]
    assert event.detected_stamp_ns > event.stamp_ns
    assert detector.state == "IMPACT"

    detector.update(pose(stamp + 120_000_000, 220.0))
    assert detector.state == "COOLDOWN"


def test_motion_that_never_reaches_required_mitt_zone_is_rejected() -> None:
    detector = FrontImpactDetector2D(
        {
            "minimum_landmark_confidence": 0.3,
            "guard_max_wrist_nose_ratio": 1.85,
            "guard_max_speed_body_s": 0.6,
            "guard_ready_frames": 3,
            "start_speed_body_s": 0.8,
            "start_displacement_body": 0.08,
            "start_extension_gain_deg": 4.0,
            "start_confirm_frames": 1,
            "minimum_duration_s": 0.06,
            "maximum_duration_s": 1.2,
            "minimum_contact_displacement_body": 0.12,
            "minimum_contact_extension_gain_deg": 8.0,
            "minimum_contact_displacement_without_extension_body": 0.30,
            "minimum_peak_speed_body_s": 0.8,
            "deceleration_ratio": 0.65,
            "reversal_cosine": 0.1,
            "contact_settle_frames": 2,
            "impact_display_s": 0.12,
            "cooldown_s": 0.5,
            "max_lost_frames": 5,
            "fist_extension_forearm_ratio": 0.35,
            "require_mitt_roi": True,
            "mitt_roi_normalized": [0.80, 0.44, 0.95, 0.52],
        }
    )
    stamp = 1_000_000_000
    for _ in range(4):
        assert detector.update(pose(stamp, 220.0)) is None
        stamp += 40_000_000
    states = []
    for index, x in enumerate([235.0, 270.0, 315.0, 350.0, 350.0, 342.0, 335.0]):
        assert detector.update(pose(stamp, x, 210.0 + index * 3.0)) is None
        states.append(detector.state)
        stamp += 40_000_000
    for _ in range(3):
        assert detector.update(pose(stamp, 335.0, 228.0)) is None
        states.append(detector.state)
        stamp += 40_000_000
    assert "ACTIVE" in states
    assert detector.state == "READY"

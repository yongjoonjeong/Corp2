import pytest

from boxing_integration.user_mitt_calibration import (
    VisionTarget,
    apply_tool_xy_correction,
    calculate_vision_target_calibration,
    hand_for_punch_role,
    predict_vision_target_pose,
)


def test_orthodox_and_southpaw_roles_use_the_correct_hands():
    assert hand_for_punch_role("right", "jab") == "left"
    assert hand_for_punch_role("right", "straight") == "right"
    assert hand_for_punch_role("left", "jab") == "right"
    assert hand_for_punch_role("left", "straight") == "left"


def target(base, x, y):
    return VisionTarget(apply_tool_xy_correction(base, x, y))


def test_ten_vision_targets_produce_a_robust_center_and_reject_one_outlier():
    base = (100, 200, 300, 0, 0, 0)
    samples = [target(base, 12 + index % 2, -8 - index % 2) for index in range(9)]
    samples.append(target(base, 90, 70))

    result = calculate_vision_target_calibration(samples, base)

    assert result.sample_count == 10
    assert result.accepted_sample_count == 9
    assert result.correction_x_mm == pytest.approx(12.0)
    assert result.correction_y_mm == pytest.approx(-8.0)


def test_correction_is_limited_to_five_centimetres_per_axis():
    base = (100, 200, 300, 0, 0, 0)
    result = calculate_vision_target_calibration(
        [target(base, 80, -70)] * 10, base
    )

    assert result.correction_x_mm == 50.0
    assert result.correction_y_mm == -50.0
    assert result.correction_limited is True


def test_tool_xy_correction_preserves_orientation_and_moves_in_base():
    pose = apply_tool_xy_correction((100, 200, 300, 0, 0, 0), 20, -10)
    assert pose == pytest.approx((120, 190, 300, 0, 0, 0))


def test_vision_vector_predicts_full_target_without_late_policy():
    result = predict_vision_target_pose(
        (100, 200, 300, 0, 0, 0),
        (100, 200, 300, 0, 0, 0),
        (120, 170, 0),
        (100, 50, 1000),
    )

    assert result is not None
    pose, time_ms = result
    assert time_ms == pytest.approx(300.0)
    assert pose == pytest.approx((150, 185, 300, 0, 0, 0))


def test_vision_target_is_bounded_to_five_centimetres():
    result = predict_vision_target_pose(
        (100, 200, 300, 0, 0, 0),
        (100, 200, 300, 0, 0, 0),
        (200, 100, 0),
        (100, -100, 1000),
    )

    assert result is not None
    pose, _ = result
    assert pose == pytest.approx((150, 150, 300, 0, 0, 0))

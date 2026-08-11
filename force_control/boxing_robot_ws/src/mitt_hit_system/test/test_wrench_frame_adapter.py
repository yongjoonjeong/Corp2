import math

import pytest

from mitt_hit_system.wrench_frame_adapter import (
    correct_base_wrench_to_tool,
    normalize_rotation_matrix,
    rotation_distance_degrees,
    rotation_from_zyz_degrees,
)


def test_identity_rotation_preserves_all_wrench_axes() -> None:
    wrench = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    assert correct_base_wrench_to_tool(wrench, identity) == wrench


def test_rotation_transforms_force_and_moment_with_same_transpose() -> None:
    # TOOL X points along BASE Y, TOOL Y along -BASE X, and Z is unchanged.
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    base_wrench = (0.0, 12.0, 0.0, 0.0, 0.0, 0.8)

    corrected = correct_base_wrench_to_tool(base_wrench, rotation)

    assert corrected == pytest.approx((12.0, 0.0, 0.0, 0.0, 0.0, 0.8))


def test_frame_check_pose_maps_base_normal_to_negative_tool_z() -> None:
    # A valid pose representative of the measured +BASE-Y -> -TOOL-Z mapping.
    rotation = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    corrected = correct_base_wrench_to_tool(
        (0.0, 20.0, 0.0, 1.0, 0.0, 0.0), rotation
    )

    assert corrected == pytest.approx((0.0, 0.0, -20.0, 1.0, 0.0, 0.0))


def test_zero_zyz_angles_produce_identity_rotation() -> None:
    matrix = rotation_from_zyz_degrees((0.0, 0.0, 0.0))
    assert matrix[0] == pytest.approx((1.0, 0.0, 0.0))
    assert matrix[1] == pytest.approx((0.0, 1.0, 0.0))
    assert matrix[2] == pytest.approx((0.0, 0.0, 1.0))


def test_zyz_rotation_maps_tool_x_to_base_y() -> None:
    rotation = rotation_from_zyz_degrees((90.0, 0.0, 0.0))
    corrected = correct_base_wrench_to_tool(
        (0.0, 12.0, 0.0, 0.0, 0.0, 0.8), rotation
    )
    assert corrected == pytest.approx((12.0, 0.0, 0.0, 0.0, 0.0, 0.8))


def test_rotation_distance_is_shortest_physical_angle() -> None:
    identity = rotation_from_zyz_degrees((0.0, 0.0, 0.0))
    quarter_turn = rotation_from_zyz_degrees((90.0, 0.0, 0.0))

    assert rotation_distance_degrees(identity, identity) == pytest.approx(0.0)
    assert rotation_distance_degrees(identity, quarter_turn) == pytest.approx(90.0)


def test_rotation_distance_handles_euler_wraparound() -> None:
    positive = rotation_from_zyz_degrees((179.0, 0.0, 0.0))
    negative = rotation_from_zyz_degrees((-179.0, 0.0, 0.0))

    assert rotation_distance_degrees(positive, negative) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "matrix",
    (
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0, 0.0),) * 3,
        ((math.nan, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ),
)
def test_invalid_rotation_matrix_is_rejected(matrix: object) -> None:
    with pytest.raises(ValueError):
        normalize_rotation_matrix(matrix)  # type: ignore[arg-type]

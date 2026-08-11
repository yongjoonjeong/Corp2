import pytest

from mitt_hit_system.mitt_pose_planner import MittPosePlanner, PersonMeasurement, PunchType
from mitt_hit_system.wrench_frame_adapter import rotation_from_zyz_degrees


REFERENCE = (216.30, -711.58, 328.37, 175.10, 89.74, 83.08)


def test_reference_user_keeps_final_photographed_pose():
    plan = MittPosePlanner().plan(PersonMeasurement(1730.0, 666.0))
    assert plan.tcp_pose_mm_deg == pytest.approx(REFERENCE)


def test_reach_is_still_applied_from_final_reference_pose():
    planner = MittPosePlanner()
    short = planner.plan(PersonMeasurement(1730.0, 600.0))
    reference = planner.plan(PersonMeasurement(1730.0, 666.0))
    assert short.strike_range_mm == pytest.approx(600.0)
    assert short.tcp_pose_mm_deg[:2] != pytest.approx(reference.tcp_pose_mm_deg[:2])


def test_height_is_still_applied_from_final_reference_pose():
    plan = MittPosePlanner().plan(PersonMeasurement(1800.0, 666.0))
    assert plan.target_height_mm == pytest.approx(328.37 + 70.0 * 0.818)


@pytest.mark.parametrize("punch", list(PunchType))
def test_all_punch_types_remain_supported(punch):
    MittPosePlanner().plan(PersonMeasurement(1730.0, 666.0), punch)


def test_hooks_move_closer_and_turn_face_in_opposite_directions():
    planner = MittPosePlanner(hook_face_angle_deg=55.0)
    straight = planner.plan(PersonMeasurement(1730.0, 666.0), PunchType.STRAIGHT)
    left = planner.plan(PersonMeasurement(1730.0, 666.0), PunchType.LEFT_HOOK)
    right = planner.plan(PersonMeasurement(1730.0, 666.0), PunchType.RIGHT_HOOK)

    assert left.strike_range_mm == pytest.approx(666.0 * 0.70)
    assert right.strike_range_mm == pytest.approx(left.strike_range_mm)
    assert left.tcp_pose_mm_deg[:2] == pytest.approx(right.tcp_pose_mm_deg[:2])
    assert left.face_normal_base != pytest.approx(straight.face_normal_base)
    assert right.face_normal_base != pytest.approx(straight.face_normal_base)
    assert left.face_normal_base != pytest.approx(right.face_normal_base)
    # Punch-side names are from the boxer's viewpoint, not the robot's.
    assert left.face_normal_base[1] > 0.0
    assert right.face_normal_base[1] < 0.0


def test_uppercut_keeps_horizontal_position_moves_lower_and_points_face_down():
    planner = MittPosePlanner()
    uppercut = planner.plan(PersonMeasurement(1730.0, 666.0), PunchType.UPPERCUT)

    assert uppercut.strike_range_mm == pytest.approx(666.0 * 0.55)
    assert uppercut.tcp_pose_mm_deg[:2] == pytest.approx(REFERENCE[:2])
    assert uppercut.target_height_mm == pytest.approx(328.37 + 1730.0 * (0.72 - 0.818))
    assert uppercut.face_normal_base == pytest.approx((0.0, 0.0, -1.0), abs=1e-7)


@pytest.mark.parametrize("punch", [PunchType.LEFT_HOOK, PunchType.RIGHT_HOOK, PunchType.UPPERCUT])
def test_planned_euler_angles_reproduce_the_target_face_normal(punch):
    plan = MittPosePlanner().plan(PersonMeasurement(1730.0, 666.0), punch)
    rotation = rotation_from_zyz_degrees(plan.tcp_pose_mm_deg[3:])
    normal = tuple(rotation[row][2] for row in range(3))
    assert normal == pytest.approx(plan.face_normal_base, abs=1e-7)

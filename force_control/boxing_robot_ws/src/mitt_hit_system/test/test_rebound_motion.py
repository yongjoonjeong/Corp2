from types import SimpleNamespace

import pytest

from mitt_hit_system.rebound_motion import (
    ReboundMotionConfig,
    ReboundMotionController,
    ReboundPhase,
)
from mitt_hit_system.mitt_pose_planner import (
    MittPosePlanner,
    PersonMeasurement,
    PunchType,
)


class Future:
    def __init__(self, response):
        self.response = response

    def done(self):
        return True

    def result(self):
        return self.response


class Client:
    def __init__(self, success=True, ready=True):
        self.success = success
        self.ready = ready
        self.calls = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.calls.append(request)
        return Future(SimpleNamespace(success=self.success))


def request():
    return SimpleNamespace()


def controller():
    move = Client()
    stop = Client()
    value = ReboundMotionController(
        ReboundMotionConfig(
            enabled=True,
            minimum_distance_mm=5.0,
            maximum_distance_mm=20.0,
            force_for_maximum_distance_n=80.0,
            retreat_velocity_mm_s=30.0,
            retreat_acceleration_mm_s2=60.0,
            return_velocity_mm_s=20.0,
            return_acceleration_mm_s2=40.0,
            target_tolerance_mm=0.5,
            motion_timeout_sec=3.0,
            tool_z_direction_sign=-1,
        ),
        move,
        stop,
        move_line_request_factory=request,
        move_stop_request_factory=request,
    )
    return value, move, stop


def test_force_maps_to_bounded_rebound_distance():
    value, _, _ = controller()

    assert value.distance_for_force(0.0) == pytest.approx(5.0)
    assert value.distance_for_force(40.0) == pytest.approx(10.0)
    assert value.distance_for_force(100.0) == pytest.approx(20.0)


def test_retreat_and_return_use_async_absolute_base_moveline():
    value, move, _ = controller()

    success, _ = value.start_retreat(80.0, (10, 20, 30, 0, 0, 0))

    assert success
    assert value.phase is ReboundPhase.RETREATING
    assert move.calls[0].pos == pytest.approx([10, 20, 10, 0, 0, 0])
    assert move.calls[0].vel == pytest.approx([30, 10])
    assert move.calls[0].acc == pytest.approx([60, 20])
    assert move.calls[0].ref == 0
    assert move.calls[0].mode == 0
    assert move.calls[0].sync_type == 1
    assert value.retreat_target_reached((10, 20, 10.4, 0, 0, 0))

    success, _ = value.start_return()

    assert success
    assert value.phase is ReboundPhase.RETURNING
    assert move.calls[1].pos == pytest.approx([10, 20, 30, 0, 0, 0])
    assert move.calls[1].vel == pytest.approx([20, 10])


def test_hook_rebounds_follow_each_punch_direction():
    planner = MittPosePlanner()
    person = PersonMeasurement(1730.0, 666.0)
    deltas = {}

    for punch in (PunchType.LEFT_HOOK, PunchType.RIGHT_HOOK):
        plan = planner.plan(person, punch)
        value, move, _ = controller()
        assert value.start_retreat(80.0, plan.tcp_pose_mm_deg)[0]
        deltas[punch] = tuple(
            move.calls[0].pos[index] - plan.tcp_pose_mm_deg[index]
            for index in range(3)
        )
        assert deltas[punch] == pytest.approx(
            tuple(-20.0 * component for component in plan.face_normal_base),
            abs=1e-6,
        )

    assert deltas[PunchType.LEFT_HOOK][1] < 0.0
    assert deltas[PunchType.RIGHT_HOOK][1] > 0.0


def test_uppercut_rebound_moves_upward_in_base():
    plan = MittPosePlanner().plan(
        PersonMeasurement(1730.0, 666.0), PunchType.UPPERCUT
    )
    value, move, _ = controller()

    assert value.start_retreat(80.0, plan.tcp_pose_mm_deg)[0]
    delta = tuple(
        move.calls[0].pos[index] - plan.tcp_pose_mm_deg[index]
        for index in range(3)
    )

    assert delta == pytest.approx((0.0, 0.0, 20.0), abs=1e-6)


def test_retreat_completion_uses_commanded_axis_progress_with_compliance_offset():
    value, _, _ = controller()
    assert value.start_retreat(80.0, (10, 20, 30, 0, 0, 0))[0]

    # The 2 mm transverse X offset must not hide completed Tool-Z travel.
    assert value.retreat_target_reached((12, 20, 10.4, 0, 0, 0))
    assert not value.retreat_target_reached((12, 20, 11.0, 0, 0, 0))


def test_soft_stop_uses_only_dr_ssto_and_clears_motion():
    value, _, stop = controller()
    assert value.start_retreat(10.0, (0, 0, 0, 0, 0, 0))[0]

    success, _ = value.soft_stop()

    assert success
    assert stop.calls[0].stop_mode == 2
    assert value.phase is ReboundPhase.IDLE
    assert value.allows_state_moving


def test_enabled_config_rejects_non_soft_stop_mode():
    with pytest.raises(ValueError, match="DR_SSTO"):
        ReboundMotionController(
            ReboundMotionConfig(
                enabled=True,
                minimum_distance_mm=5.0,
                maximum_distance_mm=5.0,
                force_for_maximum_distance_n=80.0,
                retreat_velocity_mm_s=30.0,
                retreat_acceleration_mm_s2=60.0,
                return_velocity_mm_s=20.0,
                return_acceleration_mm_s2=40.0,
                target_tolerance_mm=0.5,
                motion_timeout_sec=3.0,
                soft_stop_mode=1,
            )
        )

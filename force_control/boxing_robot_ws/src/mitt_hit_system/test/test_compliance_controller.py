from types import SimpleNamespace

import pytest

from mitt_hit_system.compliance_controller import ComplianceConfig, ComplianceController


class FakeFuture:
    def __init__(self, response):
        self._response = response

    def done(self):
        return True

    def result(self):
        return self._response


class FakeClient:
    def __init__(self, response=None, *, ready=True):
        self.response = response
        self.ready = ready
        self.calls = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.calls.append(request)
        response = (
            self.response.pop(0)
            if isinstance(self.response, list)
            else self.response
        )
        return FakeFuture(response)


def request():
    return SimpleNamespace()


def make_controller(
    *,
    activation_success=True,
    ready=True,
    require_normal=False,
    speed_mode=0,
):
    activate = FakeClient(SimpleNamespace(success=activation_success), ready=ready)
    release = FakeClient(SimpleNamespace(success=True), ready=ready)
    rt = FakeClient(
        SimpleNamespace(
            data=SimpleNamespace(
                robot_state=1,
                actual_tcp_position=(100.0, 200.0, 300.0, 0.0, 0.0, 0.0),
            )
        ),
        ready=ready,
    )
    speed = FakeClient(
        SimpleNamespace(success=True, speed_mode=speed_mode), ready=ready
    )
    controller = ComplianceController(
        ComplianceConfig(
            enabled=True,
            stiffness_verified=True,
            stiffness=(1, 2, 3, 4, 5, 6),
            maximum_tcp_displacement_mm=5.0,
            maximum_tcp_angular_displacement_deg=30.0,
            maximum_activation_tcp_displacement_mm=5.0,
            maximum_activation_tcp_angular_displacement_deg=30.0,
            release_retry_interval_sec=0.0,
            require_normal_speed_mode=require_normal,
        ),
        activate,
        release,
        rt,
        speed,
        activate_request_factory=request,
        release_request_factory=request,
        rt_request_factory=request,
        speed_mode_request_factory=request,
    )
    return controller, activate, release


def test_disabled_adapter_never_calls_services():
    controller = ComplianceController(ComplianceConfig())
    assert controller.services_ready()
    assert controller.robot_is_standby()
    assert controller.enable()[0]
    assert controller.release()[0]


def test_fake_future_adapter_uses_tool_reference_and_verified_stiffness():
    controller, activate, _ = make_controller()
    assert controller.robot_is_standby()
    assert controller.enable()[0]
    assert activate.calls[0].ref == 1
    assert activate.calls[0].stx == [1, 2, 3, 4, 5, 6]
    assert controller.active


def test_normal_speed_mode_gate_accepts_normal_mode():
    controller, _, _ = make_controller(require_normal=True, speed_mode=0)

    assert controller.services_ready()
    assert controller.robot_is_standby()


def test_normal_speed_mode_gate_rejects_reduced_mode():
    controller, _, _ = make_controller(require_normal=True, speed_mode=1)

    with pytest.raises(RuntimeError, match="REDUCED"):
        controller.robot_is_standby()


def test_unverified_stiffness_blocks_activation_before_service_call():
    activate = FakeClient(SimpleNamespace(success=True))
    controller = ComplianceController(
        ComplianceConfig(enabled=True),
        activate,
        FakeClient(SimpleNamespace(success=True)),
        FakeClient(SimpleNamespace(data=SimpleNamespace(robot_state=1))),
        activate_request_factory=request,
        release_request_factory=request,
        rt_request_factory=request,
    )
    assert not controller.enable()[0]
    assert activate.calls == []


def test_unset_displacement_limit_blocks_activation_before_service_call():
    activate = FakeClient(SimpleNamespace(success=True))
    controller = ComplianceController(
        ComplianceConfig(
            enabled=True,
            stiffness_verified=True,
            stiffness=(1, 2, 3, 4, 5, 6),
        ),
        activate,
        FakeClient(SimpleNamespace(success=True)),
        FakeClient(
            SimpleNamespace(
                data=SimpleNamespace(
                    robot_state=1,
                    actual_tcp_position=(0.0,) * 6,
                )
            )
        ),
        activate_request_factory=request,
        release_request_factory=request,
        rt_request_factory=request,
    )
    assert controller.robot_is_standby()
    success, message = controller.enable()
    assert not success
    assert "explicitly configured" in message
    assert activate.calls == []


def test_unset_angular_displacement_limit_blocks_activation_before_service_call():
    controller, activate, _ = make_controller()
    controller.config = ComplianceConfig(
        enabled=True,
        stiffness_verified=True,
        stiffness=(1, 2, 3, 4, 5, 6),
        maximum_tcp_displacement_mm=5.0,
        maximum_activation_tcp_displacement_mm=5.0,
        maximum_activation_tcp_angular_displacement_deg=30.0,
    )
    assert controller.robot_is_standby()

    success, message = controller.enable()

    assert not success
    assert "angular displacement" in message
    assert activate.calls == []


def test_tcp_displacement_watchdog_uses_activation_reference():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()
    controller._rt_client.response = SimpleNamespace(
        data=SimpleNamespace(
            robot_state=1,
            actual_tcp_position=(103.0, 204.0, 300.0, 0.0, 0.0, 0.0),
        )
    )

    at_limit = controller.check_rt_health()

    assert at_limit.healthy
    assert at_limit.tcp_displacement_mm == 5.0

    controller._rt_client.response.data.actual_tcp_position = (
        106.0,
        200.0,
        300.0,
        0.0,
        0.0,
        0.0,
    )
    exceeded = controller.check_rt_health()
    assert not exceeded.healthy
    assert "exceeded" in exceeded.detail


def test_moving_state_is_allowed_only_during_bounded_activation():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()
    controller._rt_client.response = SimpleNamespace(
        data=SimpleNamespace(
            robot_state=2,
            actual_tcp_position=(100.1, 200.0, 300.0, 0.0, 0.0, 0.0),
        )
    )

    activation = controller.check_rt_health(allow_moving=True)
    after_activation = controller.check_rt_health(allow_moving=False)

    assert activation.healthy
    assert activation.robot_state == 2
    assert "bounded compliance activation" in activation.detail
    assert not after_activation.healthy
    assert controller.observed_robot_state_counts == {"1": 1, "2": 2}


def test_bounded_activation_does_not_disable_displacement_watchdog():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()
    controller._rt_client.response = SimpleNamespace(
        data=SimpleNamespace(
            robot_state=2,
            actual_tcp_position=(106.0, 200.0, 300.0, 0.0, 0.0, 0.0),
        )
    )

    health = controller.check_rt_health(allow_moving=True)

    assert not health.healthy
    assert health.tcp_displacement_mm == 6.0
    assert "exceeded" in health.detail


def test_direct_rt_observation_uses_same_displacement_logic():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()

    health = controller.observe_rt_state(
        1,
        (103.0, 204.0, 300.0, 0.0, 0.0, 0.0),
        allow_moving=False,
    )

    assert health.healthy
    assert health.tcp_displacement_mm == 5.0


def test_tcp_angular_displacement_watchdog_uses_physical_rotation_distance():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()

    at_limit = controller.observe_rt_state(
        1, (100.0, 200.0, 300.0, 30.0, 0.0, 0.0)
    )
    exceeded = controller.observe_rt_state(
        1, (100.0, 200.0, 300.0, 31.0, 0.0, 0.0)
    )

    assert at_limit.healthy
    assert at_limit.tcp_angular_displacement_deg == pytest.approx(30.0)
    assert not exceeded.healthy
    assert "angular displacement exceeded" in exceeded.detail


def test_activation_limits_are_replaced_only_after_settled_reference_capture():
    controller, _, _ = make_controller()
    controller.config = ComplianceConfig(
        enabled=True,
        stiffness_verified=True,
        stiffness=(1, 2, 3, 4, 5, 6),
        maximum_tcp_displacement_mm=40.0,
        maximum_tcp_angular_displacement_deg=2.0,
        maximum_activation_tcp_displacement_mm=1.0,
        maximum_activation_tcp_angular_displacement_deg=0.3,
    )
    assert controller.robot_is_standby()
    assert controller.enable()[0]

    activation_fault = controller.observe_rt_state(
        1, (102.0, 200.0, 300.0, 0.0, 0.0, 0.0)
    )
    controller.recapture_reference_tcp_position(
        (100.0, 200.0, 300.0, 0.0, 0.0, 0.0)
    )
    punch_travel = controller.observe_rt_state(
        1, (120.0, 200.0, 300.0, 0.0, 0.0, 0.0)
    )

    assert not activation_fault.healthy
    assert "activation limit" in activation_fault.detail
    assert punch_travel.healthy
    assert punch_travel.tcp_displacement_mm == 20.0


def test_settled_compliance_pose_replaces_activation_reference():
    controller, _, _ = make_controller()
    assert controller.robot_is_standby()
    assert controller.enable()[0]
    assert controller.observe_rt_state(
        1, (104.0, 200.0, 300.0, 0.0, 0.0, 0.0)
    ).tcp_displacement_mm == 4.0

    reference = controller.recapture_reference_tcp_position(
        (104.0, 201.0, 302.0, 10.0, 20.0, 30.0)
    )

    assert reference == (104.0, 201.0, 302.0)
    assert controller.maximum_observed_displacement_mm == 0.0
    assert controller.maximum_observed_angular_displacement_deg == 0.0
    settled = controller.observe_rt_state(
        1, (104.0, 201.0, 302.0, 10.0, 20.0, 30.0)
    )
    assert settled.healthy
    assert settled.tcp_displacement_mm == 0.0


def test_reference_recapture_rejects_inactive_or_invalid_pose():
    controller, _, _ = make_controller()

    try:
        controller.recapture_reference_tcp_position((1.0, 2.0, 3.0))
    except RuntimeError as error:
        assert "active" in str(error)
    else:
        raise AssertionError("inactive compliance reference recapture was accepted")

    assert controller.robot_is_standby()
    assert controller.enable()[0]
    try:
        controller.recapture_reference_tcp_position((1.0, 2.0))
    except ValueError as error:
        assert "six" in str(error)
    else:
        raise AssertionError("invalid TCP pose was accepted")


def test_release_retries_then_succeeds_without_locking_start():
    controller, _, release = make_controller()
    assert controller.robot_is_standby()
    assert controller.enable()[0]
    release.response = [
        SimpleNamespace(success=False),
        SimpleNamespace(success=True),
    ]

    success, message = controller.release()

    assert success
    assert "attempt 2" in message
    assert len(release.calls) == 2
    assert not controller.release_failed_locked


def test_release_exhaustion_locks_future_activation():
    controller, activate, release = make_controller()
    assert controller.robot_is_standby()
    assert controller.enable()[0]
    release.response = SimpleNamespace(success=False)

    success, message = controller.release()

    assert not success
    assert "Start is locked" in message
    assert len(release.calls) == 3
    assert controller.release_failed_locked
    assert not controller.enable()[0]
    assert len(activate.calls) == 1
    assert not controller.release()[0]
    assert len(release.calls) == 3

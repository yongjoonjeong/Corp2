from types import SimpleNamespace

from mitt_hit_system.compliance_preflight_node import evaluate_preflight


def test_preflight_rejects_unavailable_service_without_rt_state():
    result = evaluate_preflight(services_ready=False, response=None)
    assert not result.success
    assert result.robot_state is None


def test_preflight_requires_robot_standby():
    response = SimpleNamespace(data=SimpleNamespace(robot_state=3))
    result = evaluate_preflight(services_ready=True, response=response)
    assert not result.success
    assert result.robot_state == 3


def test_preflight_passes_only_for_standby():
    response = SimpleNamespace(data=SimpleNamespace(robot_state=1))
    result = evaluate_preflight(services_ready=True, response=response)
    assert result.success
    assert result.robot_state == 1


def test_preflight_reports_normal_speed_mode():
    response = SimpleNamespace(data=SimpleNamespace(robot_state=1))
    speed_response = SimpleNamespace(success=True, speed_mode=0)

    result = evaluate_preflight(
        services_ready=True,
        response=response,
        speed_response=speed_response,
        require_normal_speed_mode=True,
    )

    assert result.success
    assert result.speed_mode == 0
    assert "NORMAL(0)" in result.detail


def test_punch_preflight_rejects_reduced_speed_mode():
    response = SimpleNamespace(data=SimpleNamespace(robot_state=1))
    speed_response = SimpleNamespace(success=True, speed_mode=1)

    result = evaluate_preflight(
        services_ready=True,
        response=response,
        speed_response=speed_response,
        require_normal_speed_mode=True,
    )

    assert not result.success
    assert result.speed_mode == 1
    assert "requires NORMAL" in result.detail

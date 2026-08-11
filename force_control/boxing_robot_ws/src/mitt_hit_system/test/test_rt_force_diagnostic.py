import math

from builtin_interfaces.msg import Time
import pytest

import mitt_hit_system.rt_force_diagnostic_node as diagnostic_node
from mitt_hit_system.rt_force_diagnostic_node import make_rt_sample, make_wrench
from rclpy.executors import ExternalShutdownException


def test_make_wrench_maps_six_rt_axes() -> None:
    message = make_wrench(Time(sec=1), "rt_test", [1, 2, 3, 4, 5, 6])
    assert message.header.frame_id == "rt_test"
    assert message.wrench.force.x == 1.0
    assert message.wrench.force.y == 2.0
    assert message.wrench.force.z == 3.0
    assert message.wrench.torque.x == 4.0
    assert message.wrench.torque.y == 5.0
    assert message.wrench.torque.z == 6.0


def test_make_rt_sample_keeps_pose_wrench_state_synchronized() -> None:
    message = make_rt_sample(
        Time(),
        "mitt_tool_corrected",
        (1, 2, 3, 4, 5, 6),
        (10, 20, 30, 40, 50, 60),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        1,
        0.25,
    )

    assert tuple(message.corrected_wrench) == (1, 2, 3, 4, 5, 6)
    assert tuple(message.tcp_pose_mm_deg) == (10, 20, 30, 40, 50, 60)
    assert message.robot_state == 1
    assert message.singularity == 0.25


@pytest.mark.parametrize("values", ([0] * 5, [0] * 7, [0, 0, math.nan, 0, 0, 0]))
def test_make_wrench_rejects_invalid_rt_axes(values: list[float]) -> None:
    with pytest.raises(ValueError):
        make_wrench(Time(), "rt_test", values)


def test_main_treats_ros_external_shutdown_as_clean(monkeypatch) -> None:
    calls = []

    class FakeNode:
        def destroy_node(self):
            calls.append("destroy")

    monkeypatch.setattr(
        diagnostic_node.rclpy,
        "init",
        lambda args=None: calls.append("init"),
    )
    monkeypatch.setattr(diagnostic_node, "RtForceDiagnosticNode", FakeNode)
    monkeypatch.setattr(
        diagnostic_node.rclpy,
        "spin",
        lambda node: (_ for _ in ()).throw(ExternalShutdownException()),
    )
    monkeypatch.setattr(diagnostic_node.rclpy, "ok", lambda: False)

    diagnostic_node.main()

    assert calls == ["init", "destroy"]

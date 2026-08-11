import json
from pathlib import Path
import threading
import time

from boxing_interfaces.msg import HitResult, RtMittSample, SystemState
from boxing_interfaces.srv import StartHitTest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from mitt_hit_system.hit_analyzer_node import HitAnalyzerNode


MS = 1_000_000


def wait_for(predicate, timeout_sec=3.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def publish_wrench(publisher, timestamp_ns, fz):
    message = RtMittSample()
    message.frame_id = "mitt_tool_corrected"
    message.stamp.sec = timestamp_ns // 1_000_000_000
    message.stamp.nanosec = timestamp_ns % 1_000_000_000
    message.corrected_wrench = [0.0, 0.0, fz, 0.0, 0.0, 0.0]
    message.tcp_pose_mm_deg = [0.0] * 6
    message.tcp_velocity_mm_deg_s = [0.0] * 6
    message.robot_state = 1
    publisher.publish(message)
    time.sleep(0.015)


def test_ros_services_topics_and_automatic_completion(tmp_path: Path):
    rclpy.init()
    analyzer = HitAnalyzerNode(
        parameter_overrides=[
            Parameter("calibration_duration_ms", value=90.0),
            Parameter("minimum_calibration_samples", value=10),
            Parameter("record_dir", value=str(tmp_path)),
            Parameter("save_contact_samples", value=True),
            Parameter("compliance_enabled", value=False),
        ]
    )
    probe = Node("hit_analyzer_integration_probe")
    publisher = probe.create_publisher(
        RtMittSample, "/mitt/rt_sample", 10
    )
    states = []
    results = []
    probe.create_subscription(SystemState, "/mitt/system_state", states.append, 10)
    probe.create_subscription(HitResult, "/mitt/hit_result", results.append, 10)
    start_client = probe.create_client(StartHitTest, "/mitt/start_test")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(analyzer)
    executor.add_node(probe)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    try:
        assert wait_for(lambda: publisher.get_subscription_count() == 1)
        assert wait_for(start_client.service_is_ready)

        for index in range(10):
            publish_wrench(publisher, index * 10 * MS, 0.0)
        assert wait_for(lambda: any(state.state == "READY" for state in states))

        request = StartHitTest.Request()
        request.target_hit_count = 1
        request.auto_recover = False
        future = start_client.call_async(request)
        assert wait_for(future.done)
        assert future.result().success
        assert wait_for(
            lambda: any(state.state == "WAITING_FOR_HIT" for state in states)
        )

        for timestamp_ms, fz in (
            (200, 0.0),
            (210, -12.0),
            (220, -20.0),
            (230, -16.0),
            (240, -2.0),
            (260, 0.0),
        ):
            publish_wrench(publisher, timestamp_ms * MS, fz)

        assert wait_for(lambda: len(results) == 1)
        assert wait_for(
            lambda: any(state.state == "TEST_COMPLETE" for state in states)
        )
        final = next(state for state in reversed(states) if state.state == "TEST_COMPLETE")
        assert not final.accepting_hits
        assert not final.compliance_enabled
        assert results[0].power_score == 0.0

        sessions = list(tmp_path.glob("*_session.json"))
        assert len(sessions) == 1
        document = json.loads(sessions[0].read_text(encoding="utf-8"))
        assert len(document["hits"]) == 1
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        analyzer.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()

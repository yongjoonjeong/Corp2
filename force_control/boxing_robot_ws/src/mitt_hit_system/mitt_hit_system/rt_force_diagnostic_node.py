"""Read-only diagnostic for Doosan real-time force fields."""

import math
from typing import Any, Sequence

from boxing_interfaces.msg import RtMittSample
from dsr_msgs2.srv import ReadDataRt
from geometry_msgs.msg import WrenchStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from mitt_hit_system.wrench_frame_adapter import (
    fuse_doosan_rt_wrench_to_tool,
    rotation_from_zyz_degrees,
)


def make_wrench(stamp: object, frame_id: str, values: Sequence[float]) -> WrenchStamped:
    axes = tuple(float(value) for value in values)
    if len(axes) != 6 or not all(math.isfinite(value) for value in axes):
        raise ValueError("wrench must contain six finite values")
    message = WrenchStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = axes[:3]
    message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = axes[3:]
    return message


def make_rt_sample(
    stamp: object,
    frame_id: str,
    corrected_wrench: Sequence[float],
    tcp_pose: Sequence[float],
    tcp_velocity: Sequence[float],
    robot_state: int,
    singularity: float,
) -> RtMittSample:
    wrench = tuple(float(value) for value in corrected_wrench)
    pose = tuple(float(value) for value in tcp_pose)
    velocity = tuple(float(value) for value in tcp_velocity)
    if any(len(values) != 6 for values in (wrench, pose, velocity)):
        raise ValueError("RT wrench, pose, and velocity must contain six values")
    if not all(
        math.isfinite(value)
        for values in (wrench, pose, velocity)
        for value in values
    ) or not math.isfinite(float(singularity)):
        raise ValueError("RT sample values must be finite")
    message = RtMittSample()
    message.stamp = stamp
    message.frame_id = frame_id
    message.corrected_wrench = list(wrench)
    message.tcp_pose_mm_deg = list(pose)
    message.tcp_velocity_mm_deg_s = list(velocity)
    message.robot_state = int(robot_state)
    message.singularity = float(singularity)
    return message


class RtForceDiagnosticNode(Node):
    """Poll ReadDataRt without invoking motion, compliance, or safety setters."""

    def __init__(self) -> None:
        super().__init__("rt_force_diagnostic")
        self.declare_parameter("service_name", "/dsr01/realtime/read_data_rt")
        self.declare_parameter("poll_period_ms", 2.0)
        self.declare_parameter("request_timeout_ms", 100.0)
        self.declare_parameter("corrected_topic", "/mitt/wrench_rt_corrected")
        self.declare_parameter("corrected_frame_id", "mitt_tool_corrected")
        self.declare_parameter("rt_sample_topic", "/mitt/rt_sample")
        service_name = str(self.get_parameter("service_name").value)
        self._period_ms = float(self.get_parameter("poll_period_ms").value)
        self._timeout_ms = float(self.get_parameter("request_timeout_ms").value)
        if self._period_ms <= 0.0 or self._timeout_ms <= self._period_ms:
            raise ValueError("invalid RT diagnostic timing parameters")
        self._client = self.create_client(ReadDataRt, service_name)
        rt_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._external_publisher = self.create_publisher(
            WrenchStamped, "/mitt/diagnostics/rt_external_tcp_force_base", rt_qos
        )
        self._raw_publisher = self.create_publisher(
            WrenchStamped, "/mitt/diagnostics/rt_raw_force_torque_flange", rt_qos
        )
        self._corrected_publisher = self.create_publisher(
            WrenchStamped,
            str(self.get_parameter("corrected_topic").value),
            rt_qos,
        )
        self._rt_sample_publisher = self.create_publisher(
            RtMittSample,
            str(self.get_parameter("rt_sample_topic").value),
            rt_qos,
        )
        self._corrected_frame_id = str(
            self.get_parameter("corrected_frame_id").value
        )
        self._future: Any | None = None
        self._request_started_ns = 0
        self._last_wait_log_ns = 0
        self._window_started_ns = self.get_clock().now().nanoseconds
        self._window_samples = 0
        self.create_timer(self._period_ms / 1000.0, self._on_timer)
        self.get_logger().info(
            "RT force diagnostic started (read-only). No motion, compliance, "
            "collision-sensitivity, or safety-limit API is used."
        )

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        if self._future is not None:
            if self._future.done():
                future = self._future
                self._future = None
                try:
                    response = future.result()
                    if response is None:
                        raise RuntimeError("ReadDataRt returned no response")
                    stamp = now.to_msg()
                    self._external_publisher.publish(
                        make_wrench(
                            stamp,
                            "doosan_rt_base_unverified",
                            response.data.external_tcp_force,
                        )
                    )
                    self._raw_publisher.publish(
                        make_wrench(
                            stamp,
                            "doosan_rt_flange_unverified",
                            response.data.raw_force_torque,
                        )
                    )
                    rotation = rotation_from_zyz_degrees(
                        response.data.actual_tcp_position[3:6]
                    )
                    jacobian = tuple(
                        tuple(float(value) for value in row.data)
                        for row in response.data.jacobian_matrix
                    )
                    corrected = fuse_doosan_rt_wrench_to_tool(
                        response.data.external_tcp_force,
                        response.data.external_joint_torque,
                        jacobian,
                        rotation,
                    )
                    self._corrected_publisher.publish(
                        make_wrench(stamp, self._corrected_frame_id, corrected)
                    )
                    self._rt_sample_publisher.publish(
                        make_rt_sample(
                            stamp,
                            self._corrected_frame_id,
                            corrected,
                            response.data.actual_tcp_position,
                            response.data.actual_tcp_velocity,
                            response.data.robot_state,
                            response.data.singularity,
                        )
                    )
                except Exception as error:
                    self._log_throttled("warning", f"ReadDataRt failed: {error}")
                else:
                    self._window_samples += 1
                    self._report_rate(now.nanoseconds)
                return
            elapsed_ms = (now.nanoseconds - self._request_started_ns) / 1e6
            if elapsed_ms >= self._timeout_ms:
                self._future.cancel()
                self._future = None
                self._log_throttled("warning", "ReadDataRt request timeout")
            return

        if not self._client.service_is_ready() and not self._client.wait_for_service(
            timeout_sec=0.0
        ):
            self._log_throttled(
                "warning", "waiting for /dsr01/realtime/read_data_rt"
            )
            return
        self._future = self._client.call_async(ReadDataRt.Request())
        self._request_started_ns = now.nanoseconds

    def _report_rate(self, now_ns: int) -> None:
        elapsed = (now_ns - self._window_started_ns) / 1e9
        if elapsed < 2.0:
            return
        self.get_logger().info(
            f"RT diagnostic receive rate: {self._window_samples / elapsed:.1f} Hz"
        )
        self._window_started_ns = now_ns
        self._window_samples = 0

    def _log_throttled(self, level: str, message: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_wait_log_ns < 2_000_000_000:
            return
        getattr(self.get_logger(), level)(message)
        self._last_wait_log_ns = now_ns


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RtForceDiagnosticNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # A launch-level SIGINT can arrive again while ROS entities are
            # being destroyed. It is still a normal user-requested shutdown.
            pass

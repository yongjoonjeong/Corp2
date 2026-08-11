"""Read-only compliance preflight; never enables or releases compliance."""

from dataclasses import dataclass

from dsr_msgs2.srv import (
    GetRobotSpeedMode,
    ReadDataRt,
    ReleaseComplianceCtrl,
    TaskComplianceCtrl,
)
import rclpy
from rclpy.node import Node


@dataclass(frozen=True)
class PreflightResult:
    success: bool
    robot_state: int | None
    detail: str
    speed_mode: int | None = None


def evaluate_preflight(
    *,
    services_ready: bool,
    response: object | None,
    speed_response: object | None = None,
    require_normal_speed_mode: bool = False,
) -> PreflightResult:
    if not services_ready:
        return PreflightResult(False, None, "required service unavailable")
    if response is None:
        return PreflightResult(False, None, "ReadDataRt returned no response")
    try:
        robot_state = int(response.data.robot_state)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:
        return PreflightResult(False, None, f"invalid RT response: {error}")
    if robot_state != 1:
        return PreflightResult(
            False,
            robot_state,
            f"robot is not STATE_STANDBY(1): state={robot_state}",
        )
    speed_mode = None
    if speed_response is not None:
        try:
            if not bool(speed_response.success):  # type: ignore[attr-defined]
                return PreflightResult(
                    False,
                    robot_state,
                    "GetRobotSpeedMode returned success=false",
                )
            speed_mode = int(speed_response.speed_mode)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError) as error:
            return PreflightResult(
                False,
                robot_state,
                f"invalid speed-mode response: {error}",
            )
        if speed_mode not in (0, 1):
            return PreflightResult(
                False,
                robot_state,
                f"invalid robot speed mode: {speed_mode}",
                speed_mode,
            )
    if require_normal_speed_mode and speed_mode != 0:
        return PreflightResult(
            False,
            robot_state,
            "punch preflight requires NORMAL speed mode (0); "
            f"current speed mode={speed_mode}",
            speed_mode,
        )
    speed_detail = (
        "NORMAL(0)" if speed_mode == 0 else "REDUCED(1)" if speed_mode == 1 else "unknown"
    )
    return PreflightResult(
        True,
        robot_state,
        "services available and robot is STATE_STANDBY(1); "
        f"speed mode is {speed_detail}",
        speed_mode,
    )


class CompliancePreflightNode(Node):
    def __init__(self) -> None:
        super().__init__("compliance_preflight")
        self.declare_parameter("service_wait_timeout_sec", 2.0)
        self.declare_parameter("rt_response_timeout_sec", 0.5)
        self.declare_parameter("require_normal_speed_mode", False)
        self._activate = self.create_client(
            TaskComplianceCtrl, "/dsr01/force/task_compliance_ctrl"
        )
        self._release = self.create_client(
            ReleaseComplianceCtrl, "/dsr01/force/release_compliance_ctrl"
        )
        self._rt = self.create_client(ReadDataRt, "/dsr01/realtime/read_data_rt")
        self._speed_mode = self.create_client(
            GetRobotSpeedMode, "/dsr01/system/get_robot_speed_mode"
        )

    def run(self) -> PreflightResult:
        wait_timeout = float(
            self.get_parameter("service_wait_timeout_sec").value
        )
        response_timeout = float(
            self.get_parameter("rt_response_timeout_sec").value
        )
        clients = (self._activate, self._release, self._rt, self._speed_mode)
        ready = all(
            client.wait_for_service(timeout_sec=wait_timeout) for client in clients
        )
        if not ready:
            return evaluate_preflight(services_ready=False, response=None)

        # This executable only invokes read-only RT and speed-mode queries.
        future = self._rt.call_async(ReadDataRt.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=response_timeout)
        if not future.done():
            future.cancel()
            return PreflightResult(False, None, "ReadDataRt response timeout")
        try:
            response = future.result()
        except Exception as error:
            return PreflightResult(False, None, f"ReadDataRt failed: {error}")
        speed_future = self._speed_mode.call_async(GetRobotSpeedMode.Request())
        rclpy.spin_until_future_complete(
            self, speed_future, timeout_sec=response_timeout
        )
        if not speed_future.done():
            speed_future.cancel()
            return PreflightResult(
                False,
                int(response.data.robot_state),
                "GetRobotSpeedMode response timeout",
            )
        try:
            speed_response = speed_future.result()
        except Exception as error:
            return PreflightResult(
                False,
                int(response.data.robot_state),
                f"GetRobotSpeedMode failed: {error}",
            )
        return evaluate_preflight(
            services_ready=True,
            response=response,
            speed_response=speed_response,
            require_normal_speed_mode=bool(
                self.get_parameter("require_normal_speed_mode").value
            ),
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CompliancePreflightNode()
    result = PreflightResult(False, None, "preflight did not complete")
    try:
        node.get_logger().info(
            "READ-ONLY PREFLIGHT: task/release and safety-setting services "
            "will not be called"
        )
        result = node.run()
        log = node.get_logger().info if result.success else node.get_logger().error
        log(f"PREFLIGHT {'PASS' if result.success else 'FAIL'}: {result.detail}")
    except Exception as error:
        result = PreflightResult(False, None, f"unexpected error: {error}")
        node.get_logger().error(f"PREFLIGHT FAIL: {result.detail}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not result.success:
        raise SystemExit(1)

#!/usr/bin/env python3
"""Switch a Doosan robot to MANUAL safety mode for direct teaching.

This script does not activate backdrive by itself. Use the robot's supported
cockpit/hand-guide control according to the controller and site safety rules.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


def service_name(namespace: str, suffix: str) -> str:
    root = "/" + namespace.strip("/") if namespace.strip("/") else ""
    return f"{root}/{suffix.lstrip('/')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Enable M0609 manual teaching mode")
    parser.add_argument("--namespace", default="/dsr01")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    try:
        import rclpy
        from dsr_msgs2.srv import GetRobotMode, SetRobotMode, SetSafetyMode
    except ImportError:
        print(
            "[ERROR] Source ROS Humble and the Doosan ROS 2 workspace first.",
            file=sys.stderr,
        )
        return 1

    rclpy.init(args=[])
    node = rclpy.create_node("sandbag_enable_manual_teaching")

    def call(client: Any, request: Any, label: str) -> Any:
        if not client.wait_for_service(timeout_sec=args.timeout_s):
            raise RuntimeError(f"Service unavailable: {client.srv_name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout_s)
        response = future.result() if future.done() else None
        if response is None:
            raise RuntimeError(f"Service call failed: {label}")
        return response

    try:
        mode_client = node.create_client(
            SetRobotMode,
            service_name(args.namespace, "system/set_robot_mode"),
        )
        safety_client = node.create_client(
            SetSafetyMode,
            service_name(args.namespace, "system/set_safety_mode"),
        )
        verify_client = node.create_client(
            GetRobotMode,
            service_name(args.namespace, "system/get_robot_mode"),
        )

        mode_request = SetRobotMode.Request()
        mode_request.robot_mode = 0  # ROBOT_MODE_MANUAL
        mode_response = call(mode_client, mode_request, "set manual robot mode")
        if hasattr(mode_response, "success") and not mode_response.success:
            raise RuntimeError("Controller rejected ROBOT_MODE_MANUAL")

        safety_request = SetSafetyMode.Request()
        safety_request.safety_mode = 0  # SAFETY_MODE_MANUAL
        safety_request.safety_event = 1  # SAFETY_MODE_EVENT_MOVE
        safety_response = call(safety_client, safety_request, "set manual safety mode")
        if hasattr(safety_response, "success") and not safety_response.success:
            raise RuntimeError("Controller rejected SAFETY_MODE_MANUAL")

        verified = call(verify_client, GetRobotMode.Request(), "verify manual mode")
        if not getattr(verified, "success", True) or int(verified.robot_mode) != 0:
            raise RuntimeError(f"Manual mode verification failed: {verified.robot_mode}")
        print("[READY] robot_mode=MANUAL, safety_mode=MANUAL")
        print("Use only the controller-supported hand-guide control inside a safe work cell.")
        return 0
    except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

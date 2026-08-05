#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

UI_BASE = "http://127.0.0.1:5000"


class UiRobotBridge(Node):
    def __init__(self) -> None:
        super().__init__("ko_ui_robot_bridge")
        self.wake_pub = self.create_publisher(Bool, "/wakeword_detected", 10)
        self.weave_pub = self.create_publisher(String, "/robot_boxing/weave_command", 10)
        self.action_pub = self.create_publisher(String, "/robot_boxing/action_command", 10)
        self.create_subscription(String, "/robot_boxing/weave_state", self._on_state, 10)
        self._stop = threading.Event()
        self._last_event_id = 0
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._post_status(state="BRIDGE_READY", message="UI-ROS 브리지 연결 완료")
        self.get_logger().info("KO UI ↔ ROS 위빙 브리지 준비")

    def destroy_node(self):
        self._stop.set()
        self._poll_thread.join(timeout=1.0)
        return super().destroy_node()

    def _get_json(self, url: str) -> dict:
        with urlopen(url, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_status(self, **payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{UI_BASE}/api/robot/status_update",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=1.5):
                pass
        except Exception:
            pass

    def _on_state(self, msg: String) -> None:
        self._post_status(state=msg.data, message=f"로봇 상태: {msg.data}")

    def _publish_string(self, publisher, value: str) -> None:
        msg = String()
        msg.data = value
        publisher.publish(msg)

    def _handle_command(self, command: str, payload: dict) -> None:
        self.get_logger().info(f"UI 명령 수신: {command}")
        if command == "wakeword":
            msg = Bool()
            msg.data = True
            self.wake_pub.publish(msg)
            self._post_status(
                state="STOPPING_FOR_VOICE",
                message="호출어 인식 → 위빙 정지 및 음성 명령 대기",
            )
            return

        if command in {"prepare", "start"}:
            self._publish_string(self.weave_pub, "start")
            self._post_status(
                state="WEAVE_START_REQUESTED",
                message="위빙 시작 요청",
            )
            return

        if command == "training_start":
            transcript = str((payload.get("payload") or {}).get("text", "training_start"))
            self._publish_string(self.action_pub, transcript or "training_start")
            self._post_status(state="TRAINING_STOP_REQUESTED", message="STT 훈련 확정 → 위빙 정지/준비 자세 복귀")
            return

        if command in {"stop", "pause"}:
            self._publish_string(self.weave_pub, "stop")
            return
        if command == "resume":
            # 실제 훈련 중에는 위빙을 재개하지 않는다.
            self._post_status(state="TRAINING_READY", message="훈련 재개: 위빙은 정지 상태 유지")
            return
        if command == "home":
            self._publish_string(self.weave_pub, "home")
            return
        if command == "emergency_stop":
            self._publish_string(self.weave_pub, "stop")
            self._post_status(state="SOFT_STOP_REQUESTED", message="UI 비상정지 → 위빙 Soft Stop 요청")

    def _poll_loop(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            try:
                result = self._get_json(f"{UI_BASE}/api/robot/commands?after={self._last_event_id}")
                for event in result.get("events", []):
                    self._last_event_id = max(self._last_event_id, int(event.get("id", 0)))
                    payload = event.get("payload") or {}
                    self._handle_command(str(payload.get("command", "")), payload)
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(0.12)


def main() -> None:
    rclpy.init()
    node = UiRobotBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

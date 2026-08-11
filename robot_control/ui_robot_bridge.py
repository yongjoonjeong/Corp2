#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String


UI_BASE = os.environ.get("KO_UI_BASE_URL", "http://127.0.0.1:5000").rstrip("/")


class UiRobotBridge(Node):
    """Translate KO web commands into the ROS topics owned by the robot stack.

    Important ownership rule:
    - weaving/home motion -> robot_weaving_node
    - mitt positioning / StartHitTest / StopHitTest -> boxing_integration SessionBridge

    This bridge deliberately does not call /mitt/start_test or /mitt/stop_test itself.
    """

    def __init__(self) -> None:
        super().__init__("ko_ui_robot_bridge")
        self.wake_pub = self.create_publisher(Bool, "/wakeword_detected", 10)
        self.weave_pub = self.create_publisher(String, "/robot_boxing/weave_command", 10)
        self.action_pub = self.create_publisher(String, "/robot_boxing/action_command", 10)
        self.training_request_pub = self.create_publisher(
            String, "/robot_boxing/training_request", 10
        )
        self.session_command_pub = self.create_publisher(
            String, "/robot_boxing/session_command", 10
        )
        self.create_subscription(String, "/robot_boxing/weave_state", self._on_state, 10)

        self._stop = threading.Event()
        # Never replay motion commands that were queued before this bridge restart.
        self._last_event_id = self._latest_event_id()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._post_status(state="BRIDGE_READY", message="UI-ROS 브리지 연결 완료")
        self.get_logger().info(
            "KO UI ↔ ROS bridge ready; session control delegated to boxing_integration"
        )

    def destroy_node(self):
        self._stop.set()
        self._poll_thread.join(timeout=1.0)
        return super().destroy_node()

    def _get_json(self, url: str) -> dict:
        with urlopen(url, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _latest_event_id(self) -> int:
        try:
            result = self._get_json(f"{UI_BASE}/api/robot/commands?after=0")
            return max(
                [int(event.get("id", 0)) for event in result.get("events", [])]
                or [0]
            )
        except Exception:
            return 0

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

    def _publish_string(self, publisher, value: str) -> bool:
        msg = String()
        msg.data = value
        if not rclpy.ok():
            return False
        try:
            publisher.publish(msg)
            return True
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warning(
                    f"ROS publish 실패: {type(exc).__name__}: {exc}"
                )
            return False

    def _handle_command(self, command: str, payload: dict) -> None:
        self.get_logger().info(f"UI 명령 수신: {command}")

        if command == "wakeword":
            if not rclpy.ok():
                return
            msg = Bool()
            msg.data = True
            try:
                self.wake_pub.publish(msg)
            except Exception as exc:
                if rclpy.ok():
                    self.get_logger().warning(
                        f"wakeword publish 실패: {type(exc).__name__}: {exc}"
                    )
                return
            self._post_status(
                message="호출어 인식 → training_start 전까지 위빙 유지",
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
            training = payload.get("payload") or {}
            if not isinstance(training, dict) or not training:
                self._post_status(
                    state="TRAINING_REQUEST_REJECTED",
                    message="훈련 payload가 없습니다.",
                )
                return

            training_json = json.dumps(
                training, ensure_ascii=False, separators=(",", ":")
            )
            # SessionBridge receives the request immediately. The same payload is
            # embedded in action_command, so action_ready can recover it even if
            # DDS delivery order across the two topics is inverted.
            self._publish_string(self.training_request_pub, training_json)
            action_json = json.dumps(
                {"command": "training_start", **training},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._publish_string(self.action_pub, action_json)
            self._post_status(
                state="TRAINING_PREPARING",
                training=training,
                message="정면 MediaPipe 관절 검출 완료 → 위빙 정지 후 사용자 맞춤 미트 준비",
            )
            return

        if command == "training_go":
            # StartHitTest is already owned by SessionBridge. Do not overwrite
            # the authoritative robot state (normally WAITING_FOR_HIT) merely
            # because the browser timer started.
            self._post_status(
                ui_training_active=True,
                training=payload.get("payload") or {},
                message="UI 훈련 타이머 시작",
            )
            return

        if command == "training_end":
            self._publish_string(self.session_command_pub, "stop")
            self._post_status(
                state="TRAINING_STOP_REQUESTED",
                message="훈련 종료 → 힘 세션 종료 및 위빙 재시작 요청",
            )
            return

        if command == "pause":
            # Keep the mitt at the current personalized pose. SessionBridge
            # pauses StopHitTest/compliance without routing the robot back to weave-ready.
            self._publish_string(self.session_command_pub, "pause")
            self._post_status(
                state="PAUSE_REQUESTED",
                message="훈련 일시정지 → 힘/컴플라이언스 세션 정지 요청",
            )
            return

        if command == "resume":
            self._publish_string(self.session_command_pub, "resume")
            self._post_status(
                state="RESUME_REQUESTED",
                message="훈련 재개 → 힘/컴플라이언스 재활성화 요청",
            )
            return

        if command == "stop":
            self._publish_string(self.weave_pub, "stop")
            return

        if command == "home":
            self._publish_string(self.session_command_pub, "system_shutdown")
            return

        if command == "emergency_stop":
            self._publish_string(self.weave_pub, "stop")
            self._publish_string(self.session_command_pub, "emergency_stop")
            self._post_status(
                state="SOFT_STOP_REQUESTED",
                message="UI 비상정지 → 위빙/타격 세션 정지 요청",
            )
            return

    def _poll_loop(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            try:
                result = self._get_json(
                    f"{UI_BASE}/api/robot/commands?after={self._last_event_id}"
                )
                for event in result.get("events", []):
                    self._last_event_id = max(
                        self._last_event_id, int(event.get("id", 0))
                    )
                    payload = event.get("payload") or {}
                    self._handle_command(str(payload.get("command", "")), payload)
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
                pass
            except Exception as exc:
                # SIGINT/SIGTERM may invalidate the ROS context while the polling
                # thread is between HTTP receive and publish. Treat that as normal shutdown.
                if not rclpy.ok() or self._stop.is_set():
                    return
                self.get_logger().warning(
                    f"UI command 처리 실패: {type(exc).__name__}: {exc}"
                )
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

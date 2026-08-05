#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class KoUiBridge(Node):
    """Forward vision ROS 2 topics to the local KO UI HTTP API."""

    def __init__(self) -> None:
        super().__init__("ko_ui_vision_bridge")
        self.declare_parameter("ui_base_url", os.environ.get("KO_UI_BASE_URL", "http://127.0.0.1:5000"))
        self.declare_parameter("preview_max_fps", 8.0)
        self.ui_base_url = str(self.get_parameter("ui_base_url").value).rstrip("/")
        self.preview_max_fps = max(float(self.get_parameter("preview_max_fps").value), 0.5)
        self._last_preview_sent = 0.0
        self._request_lock = threading.Lock()
        self._last_ok = False

        event_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        evidence_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        preview_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(String, "/sandbag/form/score", self.on_score, event_qos)
        self.create_subscription(String, "/sandbag/form/status", self.on_status, event_qos)
        self.create_subscription(
            CompressedImage,
            "/sandbag/form/joint_evidence/compressed",
            self.on_evidence,
            evidence_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/sandbag/form/preview/compressed",
            self.on_preview,
            preview_qos,
        )
        self.create_timer(2.0, self.send_heartbeat)
        self.get_logger().info(f"KO UI bridge ready: {self.ui_base_url}")

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: float = 1.5) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ui_base_url}{endpoint}",
            data=raw,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self._request_lock, urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(64)
            if not self._last_ok:
                self.get_logger().info("KO UI connection established")
            self._last_ok = True
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if self._last_ok:
                self.get_logger().warning(f"KO UI connection lost: {error}")
            self._last_ok = False

    def on_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self._post("/api/vision/status_update", payload)

    def on_score(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid JSON received on /sandbag/form/score")
            return
        if not isinstance(payload, dict):
            return
        self._post("/api/vision/punch", payload)

    def _send_image(self, endpoint: str, message: CompressedImage) -> None:
        if not message.data:
            return
        payload = {
            "format": message.format or "jpeg",
            "frame_id": message.header.frame_id,
            "stamp_sec": int(message.header.stamp.sec),
            "stamp_nanosec": int(message.header.stamp.nanosec),
            "data_base64": base64.b64encode(bytes(message.data)).decode("ascii"),
        }
        self._post(endpoint, payload)

    def on_evidence(self, message: CompressedImage) -> None:
        self._send_image("/api/vision/evidence", message)

    def on_preview(self, message: CompressedImage) -> None:
        now = time.monotonic()
        if now - self._last_preview_sent < 1.0 / self.preview_max_fps:
            return
        self._last_preview_sent = now
        self._send_image("/api/vision/preview", message)

    def send_heartbeat(self) -> None:
        self._post(
            "/api/vision/heartbeat",
            {
                "node": self.get_name(),
                "topics": [
                    "/sandbag/form/score",
                    "/sandbag/form/joint_evidence/compressed",
                    "/sandbag/form/preview/compressed",
                    "/sandbag/form/status",
                ],
                "time": time.time(),
            },
            timeout=1.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KoUiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

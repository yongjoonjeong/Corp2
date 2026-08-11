#!/usr/bin/env python3
"""Forward the current sandbag_vision ROS 2 outputs to the local web UI.

This bridge intentionally contains no camera, pose, YOLO, or impact detection.
The realtime v3 node remains the only vision implementation.
"""
from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

class CurrentVisionUiBridge(Node):
    """Translate only the current v3 ROS topic contract into the UI HTTP API."""

    def __init__(self) -> None:
        super().__init__("sandbag_current_vision_ui_bridge")
        self.declare_parameter(
            "ui_base_url",
            os.environ.get("KO_UI_BASE_URL", "http://127.0.0.1:5000"),
        )
        self.ui_base_url = str(self.get_parameter("ui_base_url").value).rstrip("/")
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ui-http")
        self._busy: set[str] = set()
        self._busy_lock = threading.Lock()
        self._status: dict[str, Any] = {}
        self._fists: dict[str, dict[str, Any]] = {}
        self._last_fist_status_at = 0.0
        self._last_http_ok: bool | None = None
        self._impact_by_stamp_ns: dict[int, int] = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(String, "/sandbag/vision/status", self.on_status, sensor_qos)
        self.create_subscription(String, "/sandbag/fist_state", self.on_fist, sensor_qos)
        self.create_subscription(String, "/sandbag/impact_event", self.on_impact, event_qos)
        self.create_subscription(
            CompressedImage,
            "/sandbag/impact_feedback_image/compressed",
            self.on_evidence,
            event_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/sandbag/vision/preview/compressed",
            self.on_preview,
            sensor_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/sandbag/vision/front/compressed",
            self.on_front_preview,
            sensor_qos,
        )
        self.create_timer(2.0, self.send_heartbeat)
        self.get_logger().info(f"current v3 vision -> UI bridge ready: {self.ui_base_url}")

    @staticmethod
    def _decode_json(message: String) -> dict[str, Any] | None:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: float) -> None:
        request = urllib.request.Request(
            f"{self.ui_base_url}{endpoint}",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(64)
            if self._last_http_ok is not True:
                self.get_logger().info("UI connection established")
            self._last_http_ok = True
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if self._last_http_ok is not False:
                self.get_logger().warning(f"UI connection unavailable: {error}")
            self._last_http_ok = False

    def _done(self, key: str, _future: Future[None]) -> None:
        with self._busy_lock:
            self._busy.discard(key)

    def _submit(
        self,
        key: str,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float = 0.7,
        drop_if_busy: bool = True,
    ) -> None:
        if drop_if_busy:
            with self._busy_lock:
                if key in self._busy:
                    return
                self._busy.add(key)
        future = self._pool.submit(self._post, endpoint, payload, timeout)
        if drop_if_busy:
            future.add_done_callback(lambda completed, busy_key=key: self._done(busy_key, completed))

    def _live_status(self) -> dict[str, Any]:
        status = dict(self._status)
        target_locked = status.get("target_state") == "LOCKED"
        front_pose_detected = bool(status.get("front_pose_detected"))
        guard = status.get("guard", [0, 4])
        if not isinstance(guard, list) or len(guard) < 2:
            guard = [0, 4]
        return {
            **status,
            "detector_state": status.get("impact_state", "STARTING"),
            "target_locked": target_locked,
            # Compatibility alias: this now means a real, fresh FRONT landmark
            # result, not merely a YOLO/BoT-SORT target lock.
            "pose_detected": front_pose_detected,
            "centered": target_locked,
            "guard_count": int(guard[0]),
            "guard_goal": max(1, int(guard[1])),
            "fists": dict(self._fists),
        }

    def on_status(self, message: String) -> None:
        payload = self._decode_json(message)
        if payload is None:
            return
        self._status = payload
        self._submit("status", "/api/vision/status_update", self._live_status())

    def on_fist(self, message: String) -> None:
        payload = self._decode_json(message)
        if payload is None:
            return
        side = str(payload.get("side", ""))
        if side not in ("left", "right"):
            return
        self._fists[side] = payload
        now = time.monotonic()
        if now - self._last_fist_status_at >= 0.125:
            self._last_fist_status_at = now
            self._submit("status", "/api/vision/status_update", self._live_status())

    def on_impact(self, message: String) -> None:
        payload = self._decode_json(message)
        if payload is None:
            return
        side = str(payload.get("side", "right"))
        confidence = float(payload.get("confidence", 0.0))
        fist = self._fists.get(side, {})
        contact_pixel = [
            payload.get("contact_pixel_front_x"),
            payload.get("contact_pixel_front_y"),
        ]
        impact_id = int(payload.get("impact_id", 0))
        impact_stamp_ns = int(payload.get("impact_stamp_ns", 0) or 0)
        if impact_stamp_ns > 0:
            self._impact_by_stamp_ns[impact_stamp_ns] = impact_id
            if len(self._impact_by_stamp_ns) > 100:
                for key in sorted(self._impact_by_stamp_ns)[:-80]:
                    self._impact_by_stamp_ns.pop(key, None)
        ui_payload = {
            "punch_id": impact_id,
            "punch_type": "impact",
            "punch_side": side,
            "total_score": round(confidence * 100.0, 1),
            "passed": True,
            "violations": [],
            "impact_point": {
                "front_pixel": contact_pixel,
                "robot_base_mm": fist.get("position_base_mm"),
            },
            "quality": {
                "confidence": confidence,
                "camera_count": fist.get("camera_count", 0),
                "position_std_mm": fist.get("position_std_mm"),
                "measurement_age_ms": fist.get("measurement_age_ms"),
            },
            "raw_event": payload,
        }
        self._submit(
            f"impact-{ui_payload['punch_id']}",
            "/api/vision/punch",
            ui_payload,
            timeout=1.2,
            drop_if_busy=False,
        )

    @staticmethod
    def _image_payload(message: CompressedImage) -> dict[str, Any]:
        return {
            "format": message.format or "jpeg",
            "frame_id": message.header.frame_id,
            "stamp_sec": int(message.header.stamp.sec),
            "stamp_nanosec": int(message.header.stamp.nanosec),
            "data_base64": base64.b64encode(bytes(message.data)).decode("ascii"),
        }

    def on_evidence(self, message: CompressedImage) -> None:
        if message.data:
            payload = self._image_payload(message)
            stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            impact_id = self._impact_by_stamp_ns.get(stamp_ns)
            if impact_id is None and self._impact_by_stamp_ns:
                nearest_stamp = min(self._impact_by_stamp_ns, key=lambda value: abs(value - stamp_ns))
                if abs(nearest_stamp - stamp_ns) <= 5_000_000:
                    impact_id = self._impact_by_stamp_ns.get(nearest_stamp)
            if impact_id is not None:
                payload["impact_id"] = int(impact_id)
                payload["impact_stamp_ns"] = stamp_ns
            self._submit(
                "evidence",
                "/api/vision/evidence",
                payload,
                timeout=1.5,
                drop_if_busy=False,
            )

    def on_preview(self, message: CompressedImage) -> None:
        if message.data:
            self._submit(
                "preview",
                "/api/vision/preview",
                self._image_payload(message),
                timeout=0.8,
            )

    def on_front_preview(self, message: CompressedImage) -> None:
        if message.data:
            self._submit(
                "front-preview",
                "/api/vision/front",
                self._image_payload(message),
                timeout=0.8,
            )

    def send_heartbeat(self) -> None:
        self._submit(
            "heartbeat",
            "/api/vision/heartbeat",
            {
                "node": self.get_name(),
                "source": "sandbag_vision_realtime_v3",
                "topics": [
                    "/sandbag/vision/status",
                    "/sandbag/fist_state",
                    "/sandbag/impact_event",
                    "/sandbag/impact_feedback_image/compressed",
                    "/sandbag/vision/preview/compressed",
                    "/sandbag/vision/front/compressed",
                ],
                "time": time.time(),
            },
            timeout=0.7,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CurrentVisionUiBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from sandbag_vision.node import SandbagVisionNode
from sandbag_vision.types import FramePacket


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


def test_clean_front_preview_is_rate_limited_and_published_as_jpeg():
    node = object.__new__(SandbagVisionNode)
    node.publish_front_preview = True
    node.front_preview_publish_period_ns = 100_000_000
    node.front_preview_jpeg_quality = 82
    node._last_front_preview_publish_ns = 0
    node._last_front_preview_sequence = -1
    node.front_preview_publisher = RecordingPublisher()

    image = np.full((48, 64, 3), (20, 80, 180), dtype=np.uint8)
    packet = FramePacket("front", 1_234_567_890, 7, image)
    node._publish_front_preview({"front": packet}, 1_000_000_000)

    assert len(node.front_preview_publisher.messages) == 1
    message = node.front_preview_publisher.messages[0]
    assert message.format == "jpeg"
    assert message.header.frame_id == "front_realsense_color_optical_frame"
    assert message.header.stamp.sec == 1
    assert message.header.stamp.nanosec == 234_567_890
    assert bytes(message.data).startswith(b"\xff\xd8")
    assert bytes(message.data).endswith(b"\xff\xd9")

    node._publish_front_preview({"front": packet}, 1_200_000_000)
    assert len(node.front_preview_publisher.messages) == 1

import numpy as np

from sandbag_vision.capture import FrameRingBuffer
from sandbag_vision.ekf import ConstantVelocityEkf
from sandbag_vision.types import FramePacket


def test_ring_buffer_uses_nearest_capture_stamp() -> None:
    ring = FrameRingBuffer(4)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for sequence, stamp in enumerate((100, 200, 300), 1):
        ring.append(FramePacket("front", stamp, sequence, image))
    assert ring.nearest(240).stamp_ns == 200
    assert ring.nearest(290).stamp_ns == 300
    assert ring.nearest(1000, maximum_delta_ns=100) is None


def test_ekf_marks_stale_state_invalid() -> None:
    state_filter = ConstantVelocityEkf(1000.0, 30.0, 500.0, 8.0)
    assert state_filter.update(np.asarray((0.0, 0.0, 1000.0)), 1_000_000_000, 10.0)
    assert state_filter.update(np.asarray((10.0, 0.0, 1000.0)), 1_020_000_000, 10.0)
    fresh = state_filter.estimate(1_050_000_000, 100.0)
    stale = state_filter.estimate(1_200_000_000, 100.0)
    assert fresh is not None and fresh.valid
    assert stale is not None and not stale.valid
    assert fresh.velocity_mm_s[0] > 0.0

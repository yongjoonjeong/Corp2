import threading
from collections import deque

import numpy as np

from three_camera_punch_feedback_node import LatestUsbCamera, TimedFrame


def buffered_camera(stamps: list[float]) -> LatestUsbCamera:
    camera = LatestUsbCamera.__new__(LatestUsbCamera)
    camera._lock = threading.Lock()
    camera._frames = deque(
        [
            TimedFrame(np.full((2, 2, 3), index, dtype=np.uint8), stamp)
            for index, stamp in enumerate(stamps)
        ],
        maxlen=12,
    )
    return camera


def test_nearest_selects_frame_closest_to_realsense_stamp() -> None:
    camera = buffered_camera([10.000, 10.033, 10.067, 10.100])

    selected = camera.nearest(10.060)

    assert selected is not None
    assert selected.stamp_s == 10.067
    assert np.all(selected.image == 2)


def test_latest_remains_compatible_and_returns_an_image_copy() -> None:
    camera = buffered_camera([20.000, 20.033])

    selected = camera.latest()

    assert selected is not None
    assert selected.stamp_s == 20.033
    selected.image[:] = 99
    assert np.all(camera._frames[-1].image == 1)


def test_nearest_returns_none_before_first_webcam_frame() -> None:
    camera = buffered_camera([])

    assert camera.nearest(30.0) is None

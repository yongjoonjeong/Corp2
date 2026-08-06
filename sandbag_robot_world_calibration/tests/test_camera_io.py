from __future__ import annotations

import cv2
import numpy as np

from robot_calibration.camera_io import make_depth_colormap


def test_depth_colormap_shape_and_invalid_pixels() -> None:
    depth = np.array([[0, 200, 1000], [2000, 4000, 5000]], dtype=np.uint16)
    preview = make_depth_colormap(depth, min_mm=200, max_mm=4000)
    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert np.array_equal(preview[0, 0], np.zeros(3, dtype=np.uint8))


def test_depth_colormap_near_and_far_are_different() -> None:
    depth = np.array([[300, 3500]], dtype=np.uint16)
    preview = make_depth_colormap(depth, min_mm=200, max_mm=4000)
    assert not np.array_equal(preview[0, 0], preview[0, 1])

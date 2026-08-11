import numpy as np

from sandbag_vision.pose import (
    map_original_bbox_to_rotated,
    map_rotated_normalized_to_original,
    rotate_image,
)


def test_counterclockwise_pose_mapping_returns_raw_calibration_pixel() -> None:
    raw = np.zeros((3, 5, 3), dtype=np.uint8)
    raw[1, 4] = (10, 20, 30)
    upright = rotate_image(raw, "counterclockwise_90")
    assert upright.shape == (5, 3, 3)
    # Raw (x=4,y=1) becomes upright (x=1,y=0). Convert those normalized
    # coordinates back to the original crop and recover the raw point.
    x, y = map_rotated_normalized_to_original(
        1.0 / (upright.shape[1] - 1),
        0.0,
        raw.shape[1],
        raw.shape[0],
        "counterclockwise_90",
    )
    assert np.allclose((x, y), (4.0, 1.0))


def test_clockwise_pose_mapping_returns_raw_calibration_pixel() -> None:
    x, y = map_rotated_normalized_to_original(0.5, 0.25, 101, 201, "clockwise_90")
    assert np.allclose((x, y), (25.0, 100.0))


def test_raw_bounding_box_maps_into_counterclockwise_preview() -> None:
    mapped = map_original_bbox_to_rotated((100, 50, 300, 450), 640, 480, "counterclockwise_90")
    assert mapped == (50, 339, 450, 539)

import numpy as np

from sandbag_vision.target import (
    TargetCandidate,
    _bbox_to_raw_pixels,
    select_side_candidate,
)


def candidate(track_id: int, bbox: tuple[int, int, int, int], confidence: float = 0.8):
    x1, y1, x2, y2 = bbox
    return TargetCandidate(
        track_id=track_id,
        bbox_xyxy=bbox,
        confidence=confidence,
        anchor_pixel=((x1 + x2) * 0.5, y1 + 0.43 * (y2 - y1)),
    )


def test_front_projection_overrides_a_previous_side_camera_lock() -> None:
    previous = candidate(3, (20, 40, 180, 440), 0.95)
    front_aligned = candidate(9, (360, 60, 590, 450), 0.70)

    selected, aligned, distance = select_side_candidate(
        [previous, front_aligned],
        preferred_pixel=(470.0, 230.0),
        locked_id=3,
        last_bbox=previous.bbox_xyxy,
        maximum_alignment_distance_px=180.0,
    )

    assert selected == front_aligned
    assert aligned
    assert distance is not None and distance < 20.0


def test_side_camera_keeps_local_botsort_id_when_projection_is_too_far() -> None:
    locked = candidate(3, (20, 40, 180, 440))
    other = candidate(9, (360, 60, 590, 450))

    selected, aligned, distance = select_side_candidate(
        [locked, other],
        preferred_pixel=(1000.0, 1000.0),
        locked_id=3,
        last_bbox=locked.bbox_xyxy,
        maximum_alignment_distance_px=50.0,
    )

    assert selected == locked
    assert not aligned
    assert distance is None


def test_rotated_detector_box_maps_back_to_raw_calibration_pixels() -> None:
    # In a 640x480 raw frame rotated CCW, raw (x=400,y=100) becomes (x=100,y=239).
    mapped = _bbox_to_raw_pixels(
        np.asarray((90.0, 229.0, 110.0, 249.0)),
        raw_width=640,
        raw_height=480,
        rotation="counterclockwise_90",
    )

    assert np.allclose(mapped, (390, 90, 410, 110), atol=1)

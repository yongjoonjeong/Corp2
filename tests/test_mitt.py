import cv2
import numpy as np

from sandbag_vision.mitt import RedMittTracker


def frame(red_box: tuple[int, int, int, int], distractor: bool = False) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    x1, y1, x2, y2 = red_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)
    if distractor:
        cv2.rectangle(image, (30, 40), (250, 180), (0, 0, 255), -1)
    return image


def config() -> dict:
    return {
        "enabled": True,
        "saturation_min": 150,
        "value_min": 80,
        "minimum_area_px": 1000,
        "maximum_area_px": 20000,
        "padding_px": 0,
        "maximum_center_jump_px": 120,
        "smoothing_alpha": 1.0,
        "velocity_alpha": 0.0,
        "lost_hold_frames": 1,
    }


def test_red_mitt_tracker_follows_pad_and_rejects_distant_red_distractor() -> None:
    tracker = RedMittTracker(config(), [0.65, 0.65, 0.95, 0.95])
    first = tracker.update(1, frame((450, 360, 575, 417), distractor=True))
    second = tracker.update(2, frame((390, 330, 515, 387), distractor=True))
    assert first.state == "TRACKED" and second.state == "TRACKED"
    assert np.allclose(first.roi_normalized, (450 / 640, 360 / 480, 576 / 640, 418 / 480))
    assert np.allclose(second.roi_normalized, (390 / 640, 330 / 480, 516 / 640, 388 / 480))


def test_red_mitt_tracker_fails_closed_after_short_prediction_hold() -> None:
    tracker = RedMittTracker(config(), [0.65, 0.65, 0.95, 0.95])
    assert tracker.update(1, frame((450, 360, 575, 417))).state == "TRACKED"
    assert tracker.update(2, np.zeros((480, 640, 3), dtype=np.uint8)).state == "PREDICTED"
    lost = tracker.update(3, np.zeros((480, 640, 3), dtype=np.uint8))
    assert lost.state == "LOST"
    assert lost.roi_normalized is None


def test_red_mitt_tracker_reacquires_pad_moved_far_toward_center() -> None:
    tracker = RedMittTracker(config(), [0.65, 0.65, 0.95, 0.95])
    assert tracker.update(1, frame((450, 360, 575, 417))).state == "TRACKED"

    centered = frame((170, 230, 295, 287))
    assert tracker.update(2, centered).state == "PREDICTED"
    assert tracker.update(3, centered).state == "LOST"
    reacquired = tracker.update(4, centered)

    assert reacquired.state == "TRACKED"
    assert np.allclose(reacquired.roi_normalized, (170 / 640, 230 / 480, 296 / 640, 288 / 480))

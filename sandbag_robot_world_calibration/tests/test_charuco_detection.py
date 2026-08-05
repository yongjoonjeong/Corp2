from __future__ import annotations

import cv2

from robot_calibration.common import BoardSpec, board_objects, detect_charuco


def test_generated_board_is_detected() -> None:
    spec = BoardSpec(
        dictionary="DICT_4X4_50",
        squares_x=11,
        squares_y=8,
        square_length_mm=24.0,
        marker_length_mm=15.0,
        minimum_charuco_corners=20,
        legacy_pattern=False,
    )
    _, board = board_objects(spec)
    if hasattr(board, "generateImage"):
        gray = board.generateImage((1100, 800), marginSize=20)
    else:
        gray = board.draw((1100, 800), marginSize=20)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    observation, _ = detect_charuco(image, spec)
    assert observation is not None
    assert len(observation.ids) == 70

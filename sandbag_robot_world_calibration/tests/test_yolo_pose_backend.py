from types import SimpleNamespace

import numpy as np

from punch_feedback_3d_core import CameraModel, ThreeCameraCalibration
from yolo_pose_backend import (
    CAMERA_ORDER,
    CalibratedPoseSelector,
    SingleViewPoseSelector,
    YoloKeypoint,
    YoloPoseDetection,
    depth_mask_from_pose_box,
    parse_yolo_pose_result,
    resolve_yolo_device,
)


def _calibration() -> ThreeCameraCalibration:
    K = np.asarray(((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)))
    distortion = np.zeros((5, 1))
    return ThreeCameraCalibration(
        image_width=640,
        image_height=480,
        cameras={
            "front": CameraModel("front", K, distortion, np.eye(3), np.zeros((3, 1))),
            "left": CameraModel(
                "left", K, distortion, np.eye(3), np.asarray(((180.0,), (0.0,), (0.0,)))
            ),
            "right": CameraModel(
                "right", K, distortion, np.eye(3), np.asarray(((-180.0,), (0.0,), (0.0,)))
            ),
        },
    )


def _detection(
    calibration: ThreeCameraCalibration,
    camera: str,
    offset_x_mm: float,
) -> YoloPoseDetection:
    skeleton = {
        "nose": (offset_x_mm, -210.0, 1800.0),
        "left_shoulder": (offset_x_mm - 150.0, -80.0, 1800.0),
        "right_shoulder": (offset_x_mm + 150.0, -80.0, 1800.0),
        "left_elbow": (offset_x_mm - 230.0, 40.0, 1800.0),
        "right_elbow": (offset_x_mm + 230.0, 40.0, 1800.0),
        "left_wrist": (offset_x_mm - 100.0, -170.0, 1750.0),
        "right_wrist": (offset_x_mm + 100.0, -170.0, 1750.0),
        "left_hip": (offset_x_mm - 110.0, 250.0, 1800.0),
        "right_hip": (offset_x_mm + 110.0, 250.0, 1800.0),
    }
    landmarks = {}
    pixels = []
    for name, point in skeleton.items():
        pixel = calibration.cameras[camera].project(np.asarray(point))
        pixels.append(pixel)
        landmarks[name] = YoloKeypoint(
            x=float(pixel[0] / 639.0),
            y=float(pixel[1] / 479.0),
            confidence=0.95,
        )
    pixels_array = np.asarray(pixels)
    x1, y1 = np.min(pixels_array, axis=0) - 35.0
    x2, y2 = np.max(pixels_array, axis=0) + 35.0
    return YoloPoseDetection(
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        detection_confidence=0.90,
        landmarks=landmarks,
    )


def _scaled_bbox(
    detection: YoloPoseDetection,
    scale: float,
) -> YoloPoseDetection:
    x1, y1, x2, y2 = detection.bbox_xyxy
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = (x2 - x1) * 0.5 * scale
    half_height = (y2 - y1) * 0.5 * scale
    return YoloPoseDetection(
        bbox_xyxy=(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        detection_confidence=detection.detection_confidence,
        landmarks=detection.landmarks,
    )


def test_parse_yolo_pose_result_maps_required_coco_keypoints() -> None:
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    confidence = np.zeros((1, 17), dtype=np.float32)
    xy[0, 9] = (160.0, 120.0)
    confidence[0, 9] = 0.88
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(((100.0, 50.0, 300.0, 450.0),), dtype=np.float32),
            conf=np.asarray((0.91,), dtype=np.float32),
        ),
        keypoints=SimpleNamespace(xy=xy, conf=confidence),
    )

    detections = parse_yolo_pose_result(result, 640, 480)

    assert len(detections) == 1
    assert detections[0].detection_confidence == np.float32(0.91)
    assert np.isclose(detections[0].landmarks["left_wrist"].x, 160.0 / 639.0)
    assert np.isclose(detections[0].landmarks["left_wrist"].y, 120.0 / 479.0)
    assert np.isclose(detections[0].landmarks["left_wrist"].confidence, 0.88)


def test_selector_uses_calibration_and_keeps_the_boxer_lock() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    distractor = {
        camera: _detection(calibration, camera, 650.0) for camera in CAMERA_ORDER
    }
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
    )

    selected = selector.select(
        {
            camera: (distractor[camera], boxer[camera])
            for camera in CAMERA_ORDER
        },
        calibration,
        640,
        480,
    )

    assert selector.state == "LOCKED"
    assert all(selected[camera] is boxer[camera] for camera in CAMERA_ORDER)

    selected_again = selector.select(
        {
            camera: (distractor[camera], boxer[camera])
            for camera in CAMERA_ORDER
        },
        calibration,
        640,
        480,
    )
    assert all(selected_again[camera] is boxer[camera] for camera in CAMERA_ORDER)


def test_selector_default_top_k_bounds_cartesian_association_work() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    offsets = (-800.0, -450.0, 0.0, 450.0, 800.0)
    candidates = {
        camera: tuple(
            reversed(
                [
                    boxer[camera]
                    if offset == 0.0
                    else _detection(calibration, camera, offset)
                    for offset in offsets
                ]
            )
        )
        for camera in CAMERA_ORDER
    }
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
    )
    original_association_score = selector._association_score
    association_calls = []

    def counted_association_score(selected, calibration, width, height):
        association_calls.append(selected)
        return original_association_score(selected, calibration, width, height)

    selector._association_score = counted_association_score

    selected = selector.select(candidates, calibration, 640, 480)

    assert selector.candidate_top_k == 3
    assert len(association_calls) == 3**3
    assert all(selected[camera] is boxer[camera] for camera in CAMERA_ORDER)


def test_selector_top_k_ranking_is_independent_of_detector_order() -> None:
    calibration = _calibration()
    detections = tuple(
        _detection(calibration, "front", offset)
        for offset in (-800.0, -450.0, 0.0, 450.0, 800.0)
    )
    selector = CalibratedPoseSelector(keypoint_confidence=0.35)

    ranked_forward = selector._rank_and_prune_candidates(
        "front", detections, 640, 480
    )
    ranked_reverse = selector._rank_and_prune_candidates(
        "front", tuple(reversed(detections)), 640, 480
    )

    assert len(ranked_forward) == 3
    assert all(
        forward is reverse
        for forward, reverse in zip(ranked_forward, ranked_reverse)
    )


def test_locked_selector_prunes_center_and_area_outliers_before_product() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        maximum_temporal_center_jump=0.20,
        maximum_temporal_scale_log_change=0.50,
    )
    acquired = selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )
    assert all(acquired[camera] is boxer[camera] for camera in CAMERA_ORDER)

    far = {
        camera: _detection(calibration, camera, 650.0) for camera in CAMERA_ORDER
    }
    wrong_scale = {
        camera: _scaled_bbox(boxer[camera], 3.0) for camera in CAMERA_ORDER
    }
    original_association_score = selector._association_score
    association_calls = []

    def counted_association_score(selected, calibration, width, height):
        association_calls.append(selected)
        return original_association_score(selected, calibration, width, height)

    selector._association_score = counted_association_score

    selected = selector.select(
        {
            camera: (far[camera], wrong_scale[camera], boxer[camera])
            for camera in CAMERA_ORDER
        },
        calibration,
        640,
        480,
    )

    assert len(association_calls) == 1
    assert all(selected[camera] is boxer[camera] for camera in CAMERA_ORDER)


def test_selector_requires_one_detection_from_every_camera_by_default() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
    )

    selected = selector.select(
        {"front": (boxer["front"],), "left": (), "right": (boxer["right"],)},
        calibration,
        640,
        480,
    )

    assert all(selected[camera] is None for camera in CAMERA_ORDER)
    assert selector.state == "ACQUIRING"


def test_selector_requires_candidate_survive_pruning_in_every_camera() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        maximum_temporal_center_jump=0.20,
    )
    selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )

    selected = selector.select(
        {
            "front": (boxer["front"],),
            "left": (boxer["left"],),
            "right": (_detection(calibration, "right", 650.0),),
        },
        calibration,
        640,
        480,
    )

    assert all(selected[camera] is None for camera in CAMERA_ORDER)
    assert selector.state == "LOST"


def test_selector_still_allows_two_views_when_all_cameras_not_required() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        require_all_cameras=False,
    )

    selected = selector.select(
        {"front": (boxer["front"],), "left": (boxer["left"],), "right": ()},
        calibration,
        640,
        480,
    )

    assert selected["front"] is boxer["front"]
    assert selected["left"] is boxer["left"]
    assert selected["right"] is None
    assert selector.state == "LOCKED"


def test_selector_rejects_cross_view_person_mix_by_reprojection() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    background_left = _detection(calibration, "left", 650.0)
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
    )

    selected = selector.select(
        {
            "front": (boxer["front"],),
            "left": (background_left,),
            "right": (boxer["right"],),
        },
        calibration,
        640,
        480,
    )

    assert all(selected[camera] is None for camera in CAMERA_ORDER)
    assert selector.state == "ACQUIRING"


def test_locked_selector_rejects_consistent_background_triplet_then_recovers() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    background = {
        camera: _detection(calibration, camera, 650.0) for camera in CAMERA_ORDER
    }
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        lost_timeout_frames=3,
        maximum_temporal_center_jump=0.20,
    )

    acquired = selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )
    assert all(acquired[camera] is boxer[camera] for camera in CAMERA_ORDER)

    rejected = selector.select(
        {camera: (background[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )
    assert all(rejected[camera] is None for camera in CAMERA_ORDER)
    assert selector.state == "LOST"
    assert set(selector.previous_centers) == set(CAMERA_ORDER)

    recovered = selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )
    assert all(recovered[camera] is boxer[camera] for camera in CAMERA_ORDER)
    assert selector.state == "LOCKED"


def test_selector_never_mixes_background_side_view_into_locked_boxer() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    background_left = _detection(calibration, "left", 650.0)
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        lost_timeout_frames=3,
        maximum_temporal_center_jump=0.20,
    )
    selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )

    selected = selector.select(
        {
            "front": (boxer["front"],),
            "left": (background_left,),
            "right": (boxer["right"],),
        },
        calibration,
        640,
        480,
    )

    assert all(selected[camera] is None for camera in CAMERA_ORDER)
    assert selector.state == "LOST"


def test_selector_forgets_target_only_after_configured_lost_timeout() -> None:
    calibration = _calibration()
    boxer = {camera: _detection(calibration, camera, 0.0) for camera in CAMERA_ORDER}
    background = {
        camera: _detection(calibration, camera, 650.0) for camera in CAMERA_ORDER
    }
    selector = CalibratedPoseSelector(
        keypoint_confidence=0.35,
        maximum_reprojection_error_px=20.0,
        lost_timeout_frames=3,
        maximum_temporal_center_jump=0.20,
    )
    selector.select(
        {camera: (boxer[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )

    for expected_state in ("LOST", "LOST", "ACQUIRING"):
        rejected = selector.select(
            {camera: (background[camera],) for camera in CAMERA_ORDER},
            calibration,
            640,
            480,
        )
        assert all(rejected[camera] is None for camera in CAMERA_ORDER)
        assert selector.state == expected_state

    reacquired = selector.select(
        {camera: (background[camera],) for camera in CAMERA_ORDER},
        calibration,
        640,
        480,
    )
    assert all(
        reacquired[camera] is background[camera] for camera in CAMERA_ORDER
    )
    assert selector.state == "LOCKED"


def test_depth_mask_uses_pose_box_and_torso_depth_component() -> None:
    landmarks = {
        name: YoloKeypoint(0.5, 0.5, 0.9)
        for name in (
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        )
    }
    detection = YoloPoseDetection((2.0, 2.0, 8.0, 8.0), 0.9, landmarks)
    depth = np.full((10, 10), 2000, dtype=np.uint16)
    depth[2:8, 2:8] = 1000

    mask = depth_mask_from_pose_box(
        depth,
        detection,
        0.001,
        depth_band_mm=150.0,
        bbox_margin_ratio=0.0,
    )

    assert mask is not None
    assert mask[5, 5] == 1.0
    assert mask[0, 0] == 0.0


def test_single_view_selector_prefers_center_guard_and_keeps_lock() -> None:
    calibration = _calibration()
    boxer = _detection(calibration, "front", 0.0)
    distractor = _detection(calibration, "front", 650.0)
    selector = SingleViewPoseSelector(lost_timeout_frames=3)

    selected = selector.select((distractor, boxer), 640, 480)
    assert selected is boxer
    assert selector.state == "LOCKED"

    selected_again = selector.select((distractor, boxer), 640, 480)
    assert selected_again is boxer


def test_single_view_selector_resets_after_target_is_missing() -> None:
    calibration = _calibration()
    boxer = _detection(calibration, "front", 0.0)
    selector = SingleViewPoseSelector(lost_timeout_frames=2)
    assert selector.select((boxer,), 640, 480) is boxer

    assert selector.select((), 640, 480) is None
    assert selector.state == "LOST"
    assert selector.select((), 640, 480) is None
    assert selector.state == "ACQUIRING"


def test_yolo_detection_adapts_to_existing_2d_pose_sample() -> None:
    calibration = _calibration()
    detection = _detection(calibration, "front", 0.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    sample = detection.to_pose_sample(12.5, image)

    assert sample.stamp_s == 12.5
    assert set(sample.landmarks) == set(detection.landmarks)
    assert sample.landmarks["right_wrist"].visibility == 0.95
    assert sample.landmarks["right_wrist"].z == 0.0


def test_auto_device_falls_back_when_gpu_arch_is_not_compiled() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda _index: (12, 0),
            get_arch_list=lambda: ["sm_80", "sm_86", "sm_90"],
            get_device_name=lambda _index: "RTX 5070 Laptop GPU",
        )
    )

    device, reason = resolve_yolo_device(fake_torch, "auto")

    assert device == "cpu"
    assert reason is not None
    assert "sm_120" in reason


def test_auto_device_uses_gpu_when_arch_is_compiled() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda _index: (12, 0),
            get_arch_list=lambda: ["sm_90", "sm_120"],
            get_device_name=lambda _index: "Compatible GPU",
        )
    )

    assert resolve_yolo_device(fake_torch, "auto") == ("0", None)

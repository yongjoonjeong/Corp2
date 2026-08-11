from sandbag_vision.node import FRONT_READY_LANDMARKS, _front_pose_ready_status
from sandbag_vision.types import Landmark2D, PoseSample


def pose(stamp_ns: int, confidence: float = 0.8) -> PoseSample:
    landmarks = {
        name: Landmark2D((100.0, 100.0), confidence)
        for name in FRONT_READY_LANDMARKS
    }
    return PoseSample(
        camera="front",
        stamp_ns=stamp_ns,
        frame_sequence=1,
        image_size=(640, 480),
        roi_xyxy=(0, 0, 640, 480),
        landmarks=landmarks,
        inference_ms=10.0,
    )


def test_front_pose_ready_requires_fresh_complete_confident_mediapipe_joints() -> None:
    now_ns = 1_000_000_000
    ready, confidence, age_ms = _front_pose_ready_status(
        pose(now_ns - 50_000_000), now_ns, 0.5, 250.0
    )
    assert ready
    assert confidence == 0.8
    assert age_ms == 50.0

    stale, _, _ = _front_pose_ready_status(
        pose(now_ns - 300_000_000), now_ns, 0.5, 250.0
    )
    assert not stale

    low_confidence, _, _ = _front_pose_ready_status(
        pose(now_ns - 50_000_000, confidence=0.49), now_ns, 0.5, 250.0
    )
    assert not low_confidence


def test_front_pose_ready_rejects_missing_joint() -> None:
    sample = pose(1_000_000_000)
    landmarks = dict(sample.landmarks)
    landmarks.pop("left_wrist")
    incomplete = PoseSample(
        camera=sample.camera,
        stamp_ns=sample.stamp_ns,
        frame_sequence=sample.frame_sequence,
        image_size=sample.image_size,
        roi_xyxy=sample.roi_xyxy,
        landmarks=landmarks,
        inference_ms=sample.inference_ms,
    )
    ready, confidence, _ = _front_pose_ready_status(
        incomplete, 1_050_000_000, 0.5, 250.0
    )
    assert not ready
    assert confidence == 0.0

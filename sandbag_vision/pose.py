from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Mapping

import cv2
import numpy as np

from .types import FramePacket, Landmark2D, POSE_LANDMARKS, PoseSample


LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    mode = str(rotation).strip().lower()
    if mode in ("", "none", "0"):
        return image
    if mode in ("counterclockwise_90", "ccw90", "90_ccw"):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode in ("clockwise_90", "cw90", "90_cw"):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode in ("180", "rotate_180"):
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"unsupported camera rotation: {rotation}")


def map_rotated_normalized_to_original(
    x: float,
    y: float,
    original_width: int,
    original_height: int,
    rotation: str,
) -> tuple[float, float]:
    """Map a landmark from the upright inference image to the raw crop."""
    mode = str(rotation).strip().lower()
    width = max(int(original_width) - 1, 1)
    height = max(int(original_height) - 1, 1)
    if mode in ("", "none", "0"):
        return float(x) * width, float(y) * height
    if mode in ("counterclockwise_90", "ccw90", "90_ccw"):
        return (1.0 - float(y)) * width, float(x) * height
    if mode in ("clockwise_90", "cw90", "90_cw"):
        return float(y) * width, (1.0 - float(x)) * height
    if mode in ("180", "rotate_180"):
        return (1.0 - float(x)) * width, (1.0 - float(y)) * height
    raise ValueError(f"unsupported camera rotation: {rotation}")


def map_original_bbox_to_rotated(
    bbox: tuple[int, int, int, int],
    original_width: int,
    original_height: int,
    rotation: str,
) -> tuple[int, int, int, int]:
    """Map a raw calibration-space box into the rotated display image."""
    mode = str(rotation).strip().lower()
    x1, y1, x2, y2 = (float(value) for value in bbox)

    def transform(x: float, y: float) -> tuple[float, float]:
        if mode in ("", "none", "0"):
            return x, y
        if mode in ("counterclockwise_90", "ccw90", "90_ccw"):
            return y, float(original_width - 1) - x
        if mode in ("clockwise_90", "cw90", "90_cw"):
            return float(original_height - 1) - y, x
        if mode in ("180", "rotate_180"):
            return float(original_width - 1) - x, float(original_height - 1) - y
        raise ValueError(f"unsupported camera rotation: {rotation}")

    points = [transform(x, y) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    if mode in ("counterclockwise_90", "ccw90", "90_ccw", "clockwise_90", "cw90", "90_cw"):
        display_width, display_height = original_height, original_width
    else:
        display_width, display_height = original_width, original_height
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        int(np.clip(np.floor(min(xs)), 0, display_width - 1)),
        int(np.clip(np.floor(min(ys)), 0, display_height - 1)),
        int(np.clip(np.ceil(max(xs)), 1, display_width)),
        int(np.clip(np.ceil(max(ys)), 1, display_height)),
    )


_MEDIAPIPE_IMPORT_LOCK = threading.Lock()
_MEDIAPIPE_MODULE = None


def _import_mediapipe():
    import sys

    # MediaPipe imports TensorFlow only for optional documentation helpers.
    # Block it for this one-time import instead of changing process-wide sys.path:
    # target tracking imports user-site Ultralytics modules from another thread.
    global _MEDIAPIPE_MODULE
    with _MEDIAPIPE_IMPORT_LOCK:
        if _MEDIAPIPE_MODULE is not None:
            return _MEDIAPIPE_MODULE
        block_tensorflow = "tensorflow" not in sys.modules
        if block_tensorflow:
            sys.modules["tensorflow"] = None
        try:
            import mediapipe as mp
        finally:
            if block_tensorflow and sys.modules.get("tensorflow") is None:
                del sys.modules["tensorflow"]
        _MEDIAPIPE_MODULE = mp
        return mp


def expand_and_clip_roi(
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in roi)
    margin_x = (x2 - x1) * max(float(margin_ratio), 0.0)
    margin_y = (y2 - y1) * max(float(margin_ratio), 0.0)
    return (
        int(np.clip(np.floor(x1 - margin_x), 0, image_width - 2)),
        int(np.clip(np.floor(y1 - margin_y), 0, image_height - 2)),
        int(np.clip(np.ceil(x2 + margin_x), 1, image_width)),
        int(np.clip(np.ceil(y2 + margin_y), 1, image_height)),
    )


class LatestPoseWorker:
    """One independent MediaPipe graph with a replaceable one-frame queue."""

    def __init__(
        self,
        camera: str,
        model_path: str | Path,
        pose_config: Mapping[str, float],
        rotation: str = "none",
    ) -> None:
        self.camera = camera
        self.model_path = Path(model_path).expanduser().resolve()
        self.pose_config = dict(pose_config)
        self.rotation = str(rotation)
        self._condition = threading.Condition()
        self._pending: tuple[FramePacket, tuple[int, int, int, int]] | None = None
        self._latest: PoseSample | None = None
        self._latest_lock = threading.Lock()
        self._running = True
        self.error: str | None = None
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name=f"pose-{camera}", daemon=True)
        self._thread.start()

    def submit(self, packet: FramePacket, roi_xyxy: tuple[int, int, int, int]) -> None:
        with self._condition:
            if self._pending is not None:
                self.dropped += 1
            self._pending = (packet, roi_xyxy)
            self.submitted += 1
            self._condition.notify()

    def latest(self) -> PoseSample | None:
        with self._latest_lock:
            return self._latest

    def _create_landmarker(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")
        mp = _import_mediapipe()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(self.pose_config.get("minimum_detection_confidence", 0.35)),
            min_pose_presence_confidence=float(self.pose_config.get("minimum_presence_confidence", 0.35)),
            min_tracking_confidence=float(self.pose_config.get("minimum_tracking_confidence", 0.35)),
            output_segmentation_masks=False,
        )
        return mp, mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _run(self) -> None:
        try:
            mp, landmarker = self._create_landmarker()
            last_timestamp_ms = -1
            try:
                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._pending is not None or not self._running)
                        if not self._running:
                            return
                        packet, roi = self._pending
                        self._pending = None
                    x1, y1, x2, y2 = roi
                    raw_crop = packet.color_bgr[y1:y2, x1:x2]
                    if raw_crop.shape[0] < 32 or raw_crop.shape[1] < 32:
                        continue
                    upright_crop = rotate_image(raw_crop, self.rotation)
                    crop = np.ascontiguousarray(upright_crop[:, :, ::-1])
                    timestamp_ms = max(int(packet.stamp_ns // 1_000_000), last_timestamp_ms + 1)
                    last_timestamp_ms = timestamp_ms
                    started = time.perf_counter()
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=crop),
                        timestamp_ms,
                    )
                    inference_ms = (time.perf_counter() - started) * 1000.0
                    if not result.pose_landmarks:
                        sample = PoseSample(
                            camera=self.camera,
                            stamp_ns=packet.stamp_ns,
                            frame_sequence=packet.sequence,
                            image_size=(packet.color_bgr.shape[1], packet.color_bgr.shape[0]),
                            roi_xyxy=roi,
                            landmarks={},
                            inference_ms=inference_ms,
                        )
                    else:
                        raw = result.pose_landmarks[0]
                        roi_width = x2 - x1
                        roi_height = y2 - y1
                        landmarks: dict[str, Landmark2D] = {}
                        for name in POSE_LANDMARKS:
                            point = raw[LANDMARK_INDEX[name]]
                            confidence = min(
                                float(getattr(point, "visibility", 1.0)),
                                float(getattr(point, "presence", 1.0)),
                            )
                            local_x, local_y = map_rotated_normalized_to_original(
                                float(point.x),
                                float(point.y),
                                roi_width,
                                roi_height,
                                self.rotation,
                            )
                            landmarks[name] = Landmark2D(
                                pixel=(x1 + local_x, y1 + local_y),
                                confidence=float(np.clip(confidence, 0.0, 1.0)),
                            )
                        sample = PoseSample(
                            camera=self.camera,
                            stamp_ns=packet.stamp_ns,
                            frame_sequence=packet.sequence,
                            image_size=(packet.color_bgr.shape[1], packet.color_bgr.shape[0]),
                            roi_xyxy=roi,
                            landmarks=landmarks,
                            inference_ms=inference_ms,
                        )
                    with self._latest_lock:
                        self._latest = sample
                        self.completed += 1
            finally:
                landmarker.close()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)


def draw_pose(image: np.ndarray, sample: PoseSample | None, minimum_confidence: float = 0.30) -> np.ndarray:
    output = image.copy()
    if sample is None:
        return output
    connections = (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    )
    for first, second in connections:
        a = sample.landmarks.get(first)
        b = sample.landmarks.get(second)
        if a is None or b is None or min(a.confidence, b.confidence) < minimum_confidence:
            continue
        cv2.line(output, tuple(int(round(v)) for v in a.pixel), tuple(int(round(v)) for v in b.pixel), (0, 220, 0), 2, cv2.LINE_AA)
    for name, point in sample.landmarks.items():
        if name == "nose" or point.confidence < minimum_confidence:
            continue
        cv2.circle(output, tuple(int(round(v)) for v in point.pixel), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return output

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from punch_feedback_3d_core import (
    LANDMARK_NAMES_3D,
    LandmarkObservation2D,
    ThreeCameraCalibration,
    triangulate_landmark,
)
from punch_feedback_core import Landmark2D, PoseSample


CAMERA_ORDER = ("left", "front", "right")
YOLO_KEYPOINT_INDEX = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


@dataclass(frozen=True)
class YoloKeypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class YoloPoseDetection:
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    landmarks: Mapping[str, YoloKeypoint]

    def center_normalized(self, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.asarray(
            ((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height)),
            dtype=np.float64,
        )

    def observations(
        self,
        width: int,
        height: int,
        minimum_confidence: float,
    ) -> dict[str, LandmarkObservation2D]:
        observations: dict[str, LandmarkObservation2D] = {}
        for name, landmark in self.landmarks.items():
            if (
                landmark.confidence < minimum_confidence
                or not (-0.05 <= landmark.x <= 1.05)
                or not (-0.05 <= landmark.y <= 1.05)
            ):
                continue
            observations[name] = LandmarkObservation2D(
                pixel=(landmark.x * (width - 1), landmark.y * (height - 1)),
                confidence=landmark.confidence,
            )
        return observations

    def to_pose_sample(self, stamp_s: float, image: np.ndarray) -> PoseSample:
        return PoseSample(
            stamp_s=stamp_s,
            landmarks={
                name: Landmark2D(
                    x=float(self.landmarks[name].x),
                    y=float(self.landmarks[name].y),
                    visibility=float(self.landmarks[name].confidence),
                    z=0.0,
                )
                for name in LANDMARK_NAMES_3D
            },
            image=image.copy(),
        )


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def resolve_yolo_device(torch_module: Any, requested_device: str) -> tuple[str, str | None]:
    """Resolve auto safely, including GPU compute-capability compatibility."""
    requested = str(requested_device).strip()
    if requested.lower() != "auto":
        return requested, None
    try:
        if not torch_module.cuda.is_available():
            return "cpu", "CUDA is not available to PyTorch"
        major, minor = torch_module.cuda.get_device_capability(0)
        target_arches = {f"sm_{major}{minor}", f"compute_{major}{minor}"}
        compiled_arches = set(torch_module.cuda.get_arch_list())
        if compiled_arches and target_arches.isdisjoint(compiled_arches):
            gpu_name = torch_module.cuda.get_device_name(0)
            return (
                "cpu",
                f"{gpu_name} requires sm_{major}{minor}, but this PyTorch build "
                f"contains {', '.join(sorted(compiled_arches))}",
            )
    except Exception as error:
        return "cpu", f"CUDA compatibility check failed: {error}"
    return "0", None


def parse_yolo_pose_result(
    result: Any,
    width: int,
    height: int,
) -> tuple[YoloPoseDetection, ...]:
    """Convert one Ultralytics result without leaking tensors downstream."""
    if result is None or result.boxes is None or result.keypoints is None:
        return ()
    boxes = _as_numpy(getattr(result.boxes, "xyxy", None)).reshape(-1, 4)
    if boxes.size == 0:
        return ()
    box_confidence = _as_numpy(getattr(result.boxes, "conf", None)).reshape(-1)
    keypoint_xy = _as_numpy(getattr(result.keypoints, "xy", None))
    keypoint_confidence = _as_numpy(getattr(result.keypoints, "conf", None))
    if keypoint_xy.ndim != 3 or keypoint_xy.shape[0] != boxes.shape[0]:
        return ()
    if keypoint_confidence.shape[:2] != keypoint_xy.shape[:2]:
        keypoint_confidence = np.ones(keypoint_xy.shape[:2], dtype=np.float32)

    detections: list[YoloPoseDetection] = []
    safe_width = max(float(width - 1), 1.0)
    safe_height = max(float(height - 1), 1.0)
    for detection_index, box in enumerate(boxes):
        landmarks = {
            name: YoloKeypoint(
                x=float(keypoint_xy[detection_index, index, 0]) / safe_width,
                y=float(keypoint_xy[detection_index, index, 1]) / safe_height,
                confidence=float(keypoint_confidence[detection_index, index]),
            )
            for name, index in YOLO_KEYPOINT_INDEX.items()
        }
        confidence = (
            float(box_confidence[detection_index])
            if detection_index < len(box_confidence)
            else 1.0
        )
        detections.append(
            YoloPoseDetection(
                bbox_xyxy=tuple(float(value) for value in box),
                detection_confidence=confidence,
                landmarks=landmarks,
            )
        )
    return tuple(detections)


class YoloPoseBackend:
    """Run one YOLO11n-Pose batch for the synchronized three-camera frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        image_size: int = 640,
        detector_confidence: float = 0.25,
        maximum_detections: int = 5,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO11n-Pose dependencies are missing. Run ./setup_3d.sh first."
            ) from error
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO11n-Pose model is missing: {path}. Run ./setup_3d.sh first."
            )
        self.device, self.device_fallback_reason = resolve_yolo_device(torch, device)
        self.image_size = max(int(image_size), 256)
        self.detector_confidence = float(detector_confidence)
        self.maximum_detections = max(int(maximum_detections), 1)
        self.model_path = path.resolve()
        self.model = YOLO(str(self.model_path))

    def infer(
        self,
        frames: Mapping[str, np.ndarray],
    ) -> dict[str, tuple[YoloPoseDetection, ...]]:
        camera_names = tuple(frames)
        if not camera_names:
            return {}
        images = [frames[camera] for camera in camera_names]
        results = self.model.predict(
            source=images,
            imgsz=self.image_size,
            conf=self.detector_confidence,
            max_det=self.maximum_detections,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        if len(results) != len(camera_names):
            raise RuntimeError(
                f"YOLO batch returned {len(results)} results for "
                f"{len(camera_names)} camera frames"
            )
        return {
            camera: parse_yolo_pose_result(
                result,
                frames[camera].shape[1],
                frames[camera].shape[0],
            )
            for camera, result in zip(camera_names, results)
        }

    def close(self) -> None:
        self.model = None


def _guard_cost(detection: YoloPoseDetection) -> float:
    landmarks = detection.landmarks
    required = (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
    )
    if any(landmarks[name].confidence < 0.20 for name in required):
        return 1.0
    nose = np.asarray((landmarks["nose"].x, landmarks["nose"].y))
    left_shoulder = np.asarray(
        (landmarks["left_shoulder"].x, landmarks["left_shoulder"].y)
    )
    right_shoulder = np.asarray(
        (landmarks["right_shoulder"].x, landmarks["right_shoulder"].y)
    )
    shoulder_width = max(float(np.linalg.norm(left_shoulder - right_shoulder)), 1e-3)
    ratios = []
    for side in ("left", "right"):
        wrist = landmarks[f"{side}_wrist"]
        ratios.append(
            float(np.linalg.norm(np.asarray((wrist.x, wrist.y)) - nose))
            / shoulder_width
        )
    return float(np.clip(np.mean(ratios) / 2.0, 0.0, 1.5))


class SingleViewPoseSelector:
    """Keep one boxer selected in a webcam view without a heavy MOT model."""

    def __init__(self, lost_timeout_frames: int = 30) -> None:
        self.lost_timeout_frames = max(int(lost_timeout_frames), 1)
        self.previous_center: np.ndarray | None = None
        self.previous_area_ratio: float | None = None
        self.missing_frames = 0
        self.state = "ACQUIRING"

    def reset(self) -> None:
        self.previous_center = None
        self.previous_area_ratio = None
        self.missing_frames = 0
        self.state = "ACQUIRING"

    @staticmethod
    def _area_ratio(
        detection: YoloPoseDetection,
        width: int,
        height: int,
    ) -> float:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return max((x2 - x1) * (y2 - y1), 1.0) / max(width * height, 1)

    def select(
        self,
        candidates: Sequence[YoloPoseDetection],
        width: int,
        height: int,
    ) -> YoloPoseDetection | None:
        if not candidates:
            self._mark_missing()
            return None

        scored: list[tuple[float, YoloPoseDetection]] = []
        for detection in candidates:
            center = detection.center_normalized(width, height)
            area_ratio = self._area_ratio(detection, width, height)
            if self.previous_center is None:
                center_cost = float(
                    np.linalg.norm(center - np.asarray((0.5, 0.52)))
                )
                score = (
                    0.60 * center_cost
                    + 0.25 * _guard_cost(detection)
                    + 0.15 * (1.0 - detection.detection_confidence)
                )
            else:
                center_cost = float(np.linalg.norm(center - self.previous_center))
                scale_cost = abs(
                    float(
                        np.log(
                            max(area_ratio, 1e-6)
                            / max(self.previous_area_ratio or area_ratio, 1e-6)
                        )
                    )
                )
                score = (
                    0.75 * center_cost
                    + 0.15 * min(scale_cost, 2.0)
                    + 0.10 * (1.0 - detection.detection_confidence)
                )
            scored.append((score, detection))

        scored.sort(key=lambda item: item[0])
        _, selected = scored[0]
        center = selected.center_normalized(width, height)
        if (
            self.previous_center is not None
            and np.linalg.norm(center - self.previous_center) > 0.45
        ):
            self._mark_missing()
            return None
        self.previous_center = center
        self.previous_area_ratio = self._area_ratio(selected, width, height)
        self.missing_frames = 0
        self.state = "LOCKED"
        return selected

    def _mark_missing(self) -> None:
        self.missing_frames += 1
        self.state = "LOST" if self.previous_center is not None else "ACQUIRING"
        if self.missing_frames >= self.lost_timeout_frames:
            self.reset()


class CalibratedPoseSelector:
    """Lock one boxer using multi-view geometry and temporal continuity."""

    def __init__(
        self,
        *,
        keypoint_confidence: float,
        maximum_reprojection_error_px: float = 40.0,
        lost_timeout_frames: int = 30,
        require_all_cameras: bool = True,
        candidate_top_k: int = 3,
        maximum_temporal_center_jump: float = 0.25,
        maximum_temporal_scale_log_change: float = 0.80,
    ) -> None:
        self.keypoint_confidence = float(keypoint_confidence)
        self.maximum_reprojection_error_px = float(maximum_reprojection_error_px)
        self.lost_timeout_frames = max(int(lost_timeout_frames), 1)
        self.require_all_cameras = bool(require_all_cameras)
        self.candidate_top_k = max(int(candidate_top_k), 1)
        self.maximum_temporal_center_jump = max(
            float(maximum_temporal_center_jump), 0.0
        )
        self.maximum_temporal_scale_log_change = max(
            float(maximum_temporal_scale_log_change), 0.0
        )
        self.previous_centers: dict[str, np.ndarray] = {}
        self.previous_area_ratios: dict[str, float] = {}
        self.missing_frames = 0
        self.state = "ACQUIRING"
        self.last_score = float("inf")

    def reset(self) -> None:
        self.previous_centers.clear()
        self.previous_area_ratios.clear()
        self.missing_frames = 0
        self.state = "ACQUIRING"
        self.last_score = float("inf")

    @staticmethod
    def _area_ratio(
        detection: YoloPoseDetection,
        width: int,
        height: int,
    ) -> float:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return max((x2 - x1) * (y2 - y1), 1.0) / max(width * height, 1)

    @staticmethod
    def _candidate_tie_breaker(
        detection: YoloPoseDetection,
    ) -> tuple[float, ...]:
        """Make top-K independent of the detector's candidate ordering."""
        x1, y1, x2, y2 = detection.bbox_xyxy
        return (
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            -float(detection.detection_confidence),
        )

    def _rank_and_prune_candidates(
        self,
        camera: str,
        candidates: Sequence[YoloPoseDetection],
        width: int,
        height: int,
    ) -> tuple[YoloPoseDetection, ...]:
        """Apply cheap temporal gates before expensive multi-view geometry."""
        ranked: list[tuple[tuple[float, ...], YoloPoseDetection]] = []
        previous_center = self.previous_centers.get(camera)
        previous_area = self.previous_area_ratios.get(camera)

        for detection in candidates:
            center = detection.center_normalized(width, height)
            area = self._area_ratio(detection, width, height)
            confidence_cost = 1.0 - float(detection.detection_confidence)

            if previous_center is not None:
                center_jump = float(np.linalg.norm(center - previous_center))
                if center_jump > self.maximum_temporal_center_jump:
                    continue
                scale_change = 0.0
                if previous_area is not None:
                    scale_change = abs(
                        float(
                            np.log(
                                max(area, 1e-6)
                                / max(previous_area, 1e-6)
                            )
                        )
                    )
                    if scale_change > self.maximum_temporal_scale_log_change:
                        continue
                rank = (
                    0.75 * center_jump
                    + 0.15 * min(scale_change, 2.0)
                    + 0.10 * confidence_cost
                )
            else:
                center_cost = float(
                    np.linalg.norm(center - np.asarray((0.5, 0.52)))
                )
                guard_cost = _guard_cost(detection) if camera == "front" else 0.0
                rank = (
                    0.70 * center_cost
                    + 0.15 * guard_cost
                    + 0.15 * confidence_cost
                )

            sort_key = (rank,) + self._candidate_tie_breaker(detection)
            ranked.append((sort_key, detection))

        ranked.sort(key=lambda item: item[0])
        return tuple(
            detection for _, detection in ranked[: self.candidate_top_k]
        )

    def _association_score(
        self,
        selected: Mapping[str, YoloPoseDetection],
        calibration: ThreeCameraCalibration,
        width: int,
        height: int,
    ) -> float | None:
        temporal_costs = []
        for camera, detection in selected.items():
            previous_center = self.previous_centers.get(camera)
            current_center = detection.center_normalized(width, height)
            if previous_center is not None:
                center_jump = float(
                    np.linalg.norm(current_center - previous_center)
                )
                if center_jump > self.maximum_temporal_center_jump:
                    return None
                temporal_costs.append(center_jump)

            previous_area = self.previous_area_ratios.get(camera)
            if previous_area is not None:
                current_area = self._area_ratio(detection, width, height)
                scale_change = abs(
                    float(
                        np.log(
                            max(current_area, 1e-6)
                            / max(previous_area, 1e-6)
                        )
                    )
                )
                if scale_change > self.maximum_temporal_scale_log_change:
                    return None

        reprojection_errors: list[float] = []
        for name in LANDMARK_NAMES_3D:
            observations: dict[str, LandmarkObservation2D] = {}
            for camera, detection in selected.items():
                observations.update(
                    {
                        camera: observation
                        for joint_name, observation in detection.observations(
                            width, height, self.keypoint_confidence
                        ).items()
                        if joint_name == name
                    }
                )
            triangulated = triangulate_landmark(
                calibration,
                observations,
                self.maximum_reprojection_error_px,
            )
            if (
                triangulated is not None
                and len(triangulated.cameras) == len(selected)
            ):
                reprojection_errors.append(triangulated.reprojection_rms_px)
        if len(reprojection_errors) < 4:
            return None

        front = selected["front"]
        center_cost = float(
            np.linalg.norm(
                front.center_normalized(width, height) - np.asarray((0.5, 0.52))
            )
        )
        temporal_cost = float(np.mean(temporal_costs)) if temporal_costs else 0.0
        view_penalty = 0.30 * (3 - len(selected))
        reprojection_cost = float(np.mean(reprojection_errors)) / max(
            self.maximum_reprojection_error_px, 1.0
        )
        guard_cost = _guard_cost(front) if not self.previous_centers else 0.0
        confidence_cost = 1.0 - float(
            np.mean([item.detection_confidence for item in selected.values()])
        )
        return (
            0.48 * reprojection_cost
            + 0.24 * temporal_cost
            + 0.10 * center_cost
            + 0.08 * guard_cost
            + 0.10 * confidence_cost
            + view_penalty
        )

    def select(
        self,
        candidates: Mapping[str, Sequence[YoloPoseDetection]],
        calibration: ThreeCameraCalibration,
        width: int,
        height: int,
    ) -> dict[str, YoloPoseDetection | None]:
        empty = {camera: None for camera in CAMERA_ORDER}
        if not candidates.get("front"):
            self._mark_missing()
            return empty
        if self.require_all_cameras and any(
            not candidates.get(camera) for camera in CAMERA_ORDER
        ):
            self._mark_missing()
            return empty

        pruned_candidates = {
            camera: self._rank_and_prune_candidates(
                camera,
                candidates.get(camera, ()),
                width,
                height,
            )
            for camera in CAMERA_ORDER
        }
        if not pruned_candidates["front"]:
            self._mark_missing()
            return empty
        if self.require_all_cameras and any(
            not pruned_candidates[camera] for camera in CAMERA_ORDER
        ):
            self._mark_missing()
            return empty

        combinations: list[dict[str, YoloPoseDetection]] = []
        left_candidates = pruned_candidates["left"]
        right_candidates = pruned_candidates["right"]
        left_options: Sequence[YoloPoseDetection | None] = (
            left_candidates
            if self.require_all_cameras
            else left_candidates + (None,)
        )
        right_options: Sequence[YoloPoseDetection | None] = (
            right_candidates
            if self.require_all_cameras
            else right_candidates + (None,)
        )
        for front, left, right in product(
            pruned_candidates["front"], left_options, right_options
        ):
            selected = {"front": front}
            if left is not None:
                selected["left"] = left
            if right is not None:
                selected["right"] = right
            if len(selected) >= 2:
                combinations.append(selected)

        scored = []
        for selected in combinations:
            score = self._association_score(
                selected, calibration, width, height
            )
            if score is not None:
                scored.append((score, selected))
        if not scored:
            self._mark_missing()
            return empty
        scored.sort(key=lambda item: item[0])
        score, best = scored[0]
        updated_centers = dict(self.previous_centers)
        updated_centers.update({
            camera: detection.center_normalized(width, height)
            for camera, detection in best.items()
        })
        self.previous_centers = updated_centers
        updated_areas = dict(self.previous_area_ratios)
        updated_areas.update({
            camera: self._area_ratio(detection, width, height)
            for camera, detection in best.items()
        })
        self.previous_area_ratios = updated_areas
        self.missing_frames = 0
        self.state = "LOCKED"
        self.last_score = float(score)
        return {camera: best.get(camera) for camera in CAMERA_ORDER}

    def _mark_missing(self) -> None:
        self.missing_frames += 1
        self.state = "LOST" if self.previous_centers else "ACQUIRING"
        if self.missing_frames >= self.lost_timeout_frames:
            self.reset()


def depth_mask_from_pose_box(
    depth_raw: np.ndarray,
    detection: YoloPoseDetection | None,
    depth_scale_m: float,
    *,
    minimum_keypoint_confidence: float = 0.20,
    depth_band_mm: float = 350.0,
    bbox_margin_ratio: float = 0.05,
) -> np.ndarray | None:
    """Approximate a foreground person mask using YOLO ROI and torso depth."""
    if detection is None:
        return None
    depth = np.asarray(depth_raw)
    if depth.ndim != 2 or depth.size == 0:
        return None
    height, width = depth.shape
    x1, y1, x2, y2 = detection.bbox_xyxy
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    margin_x = box_width * max(float(bbox_margin_ratio), 0.0)
    margin_y = box_height * max(float(bbox_margin_ratio), 0.0)
    left = max(int(np.floor(x1 - margin_x)), 0)
    top = max(int(np.floor(y1 - margin_y)), 0)
    right = min(int(np.ceil(x2 + margin_x)), width)
    bottom = min(int(np.ceil(y2 + margin_y)), height)
    if left >= right or top >= bottom:
        return None

    seed_depths_mm: list[float] = []
    seed_pixels: list[tuple[int, int]] = []
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        landmark = detection.landmarks[name]
        if landmark.confidence < minimum_keypoint_confidence:
            continue
        x = int(np.clip(round(landmark.x * (width - 1)), 0, width - 1))
        y = int(np.clip(round(landmark.y * (height - 1)), 0, height - 1))
        patch = depth[max(y - 2, 0) : min(y + 3, height), max(x - 2, 0) : min(x + 3, width)]
        valid = patch[patch > 0]
        if valid.size:
            seed_depths_mm.append(float(np.median(valid)) * depth_scale_m * 1000.0)
            seed_pixels.append((x, y))
    if not seed_depths_mm:
        center_x = int(np.clip(round((x1 + x2) * 0.5), 0, width - 1))
        center_y = int(np.clip(round(y1 + box_height * 0.45), 0, height - 1))
        patch = depth[
            max(center_y - 4, 0) : min(center_y + 5, height),
            max(center_x - 4, 0) : min(center_x + 5, width),
        ]
        valid = patch[patch > 0]
        if not valid.size:
            return None
        seed_depths_mm.append(float(np.median(valid)) * depth_scale_m * 1000.0)
        seed_pixels.append((center_x, center_y))

    target_depth_mm = float(np.median(seed_depths_mm))
    roi_depth_mm = depth[top:bottom, left:right].astype(np.float32) * float(
        depth_scale_m * 1000.0
    )
    roi_mask = (
        (roi_depth_mm > 0.0)
        & (np.abs(roi_depth_mm - target_depth_mm) <= max(float(depth_band_mm), 1.0))
    ).astype(np.uint8)
    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi_mask, connectivity=8
    )
    if component_count <= 1:
        return None
    seed_labels = []
    for x, y in seed_pixels:
        if left <= x < right and top <= y < bottom:
            label = int(labels[y - top, x - left])
            if label > 0:
                seed_labels.append(label)
    selected_label = (
        max(set(seed_labels), key=seed_labels.count)
        if seed_labels
        else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    )
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = (labels == selected_label).astype(np.float32)
    return mask

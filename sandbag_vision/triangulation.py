from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .calibration import ThreeCameraCalibration
from .types import CAMERAS, Landmark2D, PoseSample, TriangulationResult


CAMERA_MASK = {"left": 1, "front": 2, "right": 4}


class PoseHistory:
    def __init__(self, maximum_samples: int = 30) -> None:
        self.samples: deque[PoseSample] = deque(maxlen=max(4, int(maximum_samples)))

    def add(self, sample: PoseSample) -> bool:
        if self.samples and sample.stamp_ns <= self.samples[-1].stamp_ns:
            return False
        self.samples.append(sample)
        return True

    def clear(self) -> None:
        self.samples.clear()

    @property
    def latest(self) -> PoseSample | None:
        return self.samples[-1] if self.samples else None

    def at(self, stamp_ns: int, maximum_gap_ns: int, maximum_skew_ns: int) -> PoseSample | None:
        if not self.samples:
            return None
        before = None
        after = None
        for sample in self.samples:
            if sample.stamp_ns <= stamp_ns:
                before = sample
            if sample.stamp_ns >= stamp_ns:
                after = sample
                break
        if before is not None and after is not None:
            if before.stamp_ns == after.stamp_ns:
                return before
            gap = after.stamp_ns - before.stamp_ns
            if gap <= maximum_gap_ns:
                ratio = (stamp_ns - before.stamp_ns) / gap
                landmarks: dict[str, Landmark2D] = {}
                for name in set(before.landmarks) & set(after.landmarks):
                    first = before.landmarks[name]
                    second = after.landmarks[name]
                    landmarks[name] = Landmark2D(
                        pixel=(
                            first.pixel[0] + ratio * (second.pixel[0] - first.pixel[0]),
                            first.pixel[1] + ratio * (second.pixel[1] - first.pixel[1]),
                        ),
                        confidence=min(first.confidence, second.confidence),
                    )
                return PoseSample(
                    camera=before.camera,
                    stamp_ns=int(stamp_ns),
                    frame_sequence=before.frame_sequence if ratio < 0.5 else after.frame_sequence,
                    image_size=before.image_size,
                    roi_xyxy=before.roi_xyxy if ratio < 0.5 else after.roi_xyxy,
                    landmarks=landmarks,
                    inference_ms=max(before.inference_ms, after.inference_ms),
                )
        nearest = min(self.samples, key=lambda sample: abs(sample.stamp_ns - int(stamp_ns)))
        return nearest if abs(nearest.stamp_ns - int(stamp_ns)) <= maximum_skew_ns else None

    def nearest(self, stamp_ns: int, maximum_skew_ns: int | None = None) -> PoseSample | None:
        if not self.samples:
            return None
        sample = min(self.samples, key=lambda item: abs(item.stamp_ns - int(stamp_ns)))
        if maximum_skew_ns is not None and abs(sample.stamp_ns - int(stamp_ns)) > maximum_skew_ns:
            return None
        return sample


def align_pose_histories(
    histories: Mapping[str, PoseHistory],
    now_ns: int,
    maximum_result_age_ns: int,
    maximum_gap_ns: int,
    maximum_skew_ns: int,
) -> tuple[int, dict[str, PoseSample]] | None:
    active = {
        camera: history.latest
        for camera, history in histories.items()
        if history.latest is not None and now_ns - history.latest.stamp_ns <= maximum_result_age_ns
    }
    if len(active) < 2:
        return None
    reference_ns = min(sample.stamp_ns for sample in active.values())
    aligned = {
        camera: sample
        for camera, history in histories.items()
        if (sample := history.at(reference_ns, maximum_gap_ns, maximum_skew_ns)) is not None
    }
    if len(aligned) < 2:
        return None
    return reference_ns, aligned


def _weighted_dlt(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, Landmark2D],
    cameras: Sequence[str],
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for name in cameras:
        observation = observations[name]
        x, y = calibration.cameras[name].undistort_normalized(observation.pixel)
        projection = calibration.cameras[name].projection_base_normalized
        weight = np.sqrt(max(float(observation.confidence), 1e-3))
        rows.append(weight * (x * projection[2] - projection[0]))
        rows.append(weight * (y * projection[2] - projection[1]))
    if len(rows) < 4:
        return None
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-9:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.all(np.isfinite(point)) else None


def _ray_angle_deg(calibration: ThreeCameraCalibration, observations: Mapping[str, Landmark2D], names: Sequence[str]) -> float:
    angles: list[float] = []
    for first, second in combinations(names, 2):
        _, ray_a = calibration.cameras[first].ray_base(observations[first].pixel)
        _, ray_b = calibration.cameras[second].ray_base(observations[second].pixel)
        cosine = float(np.clip(abs(np.dot(ray_a, ray_b)), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    return min(angles) if angles else 0.0


def _errors(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, Landmark2D],
    point_base_mm: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(np.linalg.norm(calibration.cameras[name].project_base(point_base_mm) - np.asarray(observation.pixel)))
        for name, observation in observations.items()
    }


def _robust_refine(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, Landmark2D],
    cameras: Sequence[str],
    initial: np.ndarray,
    huber_scale_px: float,
) -> np.ndarray:
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return initial

    def residual(point: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for name in cameras:
            projected = calibration.cameras[name].project_base(point)
            observed = np.asarray(observations[name].pixel, dtype=np.float64)
            weight = np.sqrt(max(observations[name].confidence, 1e-3))
            values.extend((weight * (projected - observed)).tolist())
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residual,
        np.asarray(initial, dtype=np.float64),
        loss="huber",
        f_scale=max(float(huber_scale_px), 1.0),
        max_nfev=30,
    )
    return result.x if result.success and np.all(np.isfinite(result.x)) else initial


def triangulate_robust(
    calibration: ThreeCameraCalibration,
    observations: Mapping[str, Landmark2D],
    maximum_reprojection_error_px: float,
    minimum_ray_angle_deg: float,
) -> TriangulationResult | None:
    available = tuple(
        name for name in CAMERAS if name in observations and observations[name].confidence > 0.0
    )
    if len(available) < 2:
        return None
    candidates: list[tuple[int, float, float, np.ndarray, tuple[str, ...]]] = []
    for pair in combinations(available, 2):
        angle = _ray_angle_deg(calibration, observations, pair)
        if angle < minimum_ray_angle_deg:
            continue
        point = _weighted_dlt(calibration, observations, pair)
        if point is None or any(calibration.cameras[name].base_to_camera(point)[2] <= 1.0 for name in pair):
            continue
        errors = _errors(calibration, observations, point)
        inliers = tuple(
            name
            for name in available
            if errors[name] <= maximum_reprojection_error_px
            and calibration.cameras[name].base_to_camera(point)[2] > 1.0
        )
        if len(inliers) < 2:
            continue
        rms = float(np.sqrt(np.mean([errors[name] ** 2 for name in inliers])))
        candidates.append((len(inliers), rms, -angle, point, inliers))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, _, initial, inliers = candidates[0]
    all_inlier_dlt = _weighted_dlt(calibration, observations, inliers)
    if all_inlier_dlt is not None:
        initial = all_inlier_dlt
    refined = _robust_refine(calibration, observations, inliers, initial, maximum_reprojection_error_px * 0.5)
    errors = _errors(calibration, {name: observations[name] for name in inliers}, refined)
    final_inliers = tuple(name for name in inliers if errors[name] <= maximum_reprojection_error_px)
    if len(final_inliers) < 2:
        return None
    if final_inliers != inliers:
        refined = _weighted_dlt(calibration, observations, final_inliers)
        if refined is None:
            return None
        errors = _errors(calibration, {name: observations[name] for name in final_inliers}, refined)
    rms = float(np.sqrt(np.mean([errors[name] ** 2 for name in final_inliers])))
    angle = _ray_angle_deg(calibration, observations, final_inliers)
    mean_depth = float(np.mean([calibration.cameras[name].base_to_camera(refined)[2] for name in final_inliers]))
    mean_focal = float(np.mean([calibration.cameras[name].camera_matrix[0, 0] for name in final_inliers]))
    geometry_scale = max(np.sin(np.radians(max(angle, 1.0))), 0.05)
    position_std = float(np.clip(max(rms, 0.35) * mean_depth / max(mean_focal, 1.0) / geometry_scale, 5.0, 300.0))
    visibility = float(np.mean([observations[name].confidence for name in final_inliers]))
    confidence = visibility * np.exp(-rms / max(maximum_reprojection_error_px, 1.0)) * np.clip(angle / 20.0, 0.25, 1.0)
    return TriangulationResult(
        point_base_mm=refined,
        cameras=final_inliers,
        camera_mask=sum(CAMERA_MASK[name] for name in final_inliers),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        reprojection_rms_px=rms,
        minimum_ray_angle_deg=angle,
        position_std_mm=position_std,
    )


def fuse_front_depth(
    result: TriangulationResult,
    depth_point_base_mm: np.ndarray | None,
    weight: float,
    maximum_disagreement_mm: float,
) -> TriangulationResult:
    if depth_point_base_mm is None:
        return result
    depth_point = np.asarray(depth_point_base_mm, dtype=np.float64).reshape(3)
    disagreement = float(np.linalg.norm(depth_point - result.point_base_mm))
    if not np.all(np.isfinite(depth_point)) or disagreement > maximum_disagreement_mm:
        return TriangulationResult(
            point_base_mm=result.point_base_mm,
            cameras=result.cameras,
            camera_mask=result.camera_mask,
            confidence=result.confidence,
            reprojection_rms_px=result.reprojection_rms_px,
            minimum_ray_angle_deg=result.minimum_ray_angle_deg,
            position_std_mm=result.position_std_mm,
            depth_used=False,
            depth_agreement_mm=disagreement,
        )
    blend = float(np.clip(weight, 0.0, 1.0))
    fused = (1.0 - blend) * result.point_base_mm + blend * depth_point
    return TriangulationResult(
        point_base_mm=fused,
        cameras=result.cameras,
        camera_mask=result.camera_mask,
        confidence=result.confidence,
        reprojection_rms_px=result.reprojection_rms_px,
        minimum_ray_angle_deg=result.minimum_ray_angle_deg,
        position_std_mm=result.position_std_mm,
        depth_used=True,
        depth_agreement_mm=disagreement,
    )


def median_depth_point_base(
    calibration: ThreeCameraCalibration,
    frame_depth_raw: np.ndarray | None,
    depth_scale_mm: float,
    pixel: tuple[float, float],
    radius: int = 3,
) -> np.ndarray | None:
    if frame_depth_raw is None:
        return None
    x = int(round(pixel[0]))
    y = int(round(pixel[1]))
    height, width = frame_depth_raw.shape[:2]
    patch = frame_depth_raw[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    values = patch[patch > 0]
    if not values.size:
        return None
    depth_mm = float(np.median(values)) * float(depth_scale_mm)
    if not 150.0 <= depth_mm <= 8000.0:
        return None
    return calibration.cameras["front"].depth_pixel_to_base(pixel, depth_mm)

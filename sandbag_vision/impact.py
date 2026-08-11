from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .types import ImpactEvent, Landmark2D, PoseSample, SIDES


def _angle_deg(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    first = a - vertex
    second = c - vertex
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    if norm <= 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / norm, -1.0, 1.0))))


@dataclass
class _ArmFeature:
    stamp_ns: int
    wrist: np.ndarray
    contact: np.ndarray
    velocity: np.ndarray
    speed: float
    reach: float
    elbow_angle: float


@dataclass
class _ContactCandidate:
    stamp_ns: int
    wrist_pixel: tuple[float, float]
    contact_pixel: tuple[float, float]
    score: float
    speed: float
    displacement: float
    extension_gain: float


class FrontImpactDetector2D:
    """Front-view visual contact candidate from outbound motion then deceleration."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.state = "WAIT_GUARD"
        self.guard_count = 0
        self._previous: dict[str, _ArmFeature] = {}
        self._baseline: dict[str, _ArmFeature] = {}
        self._start_confirm = {side: 0 for side in SIDES}
        self._active_side: str | None = None
        self._active_started_ns = 0
        self._peak_speed = 0.0
        self._previous_velocity: np.ndarray | None = None
        self._settle_count = 0
        self._best_contact: _ContactCandidate | None = None
        self._lost_frames = 0
        self._impact_until_ns = 0
        self._cooldown_until_ns = 0
        self._impact_id = 0
        self._mitt_roi_history: deque[
            tuple[int, tuple[float, float, float, float] | None]
        ] = deque(maxlen=120)

    def set_mitt_roi(
        self,
        stamp_ns: int,
        roi_normalized: tuple[float, float, float, float] | None,
    ) -> None:
        self._mitt_roi_history.append((int(stamp_ns), roi_normalized))

    def _mitt_roi_at(self, stamp_ns: int):
        if not self._mitt_roi_history:
            return self.config.get("mitt_roi_normalized")
        nearest_stamp, nearest_roi = min(
            self._mitt_roi_history,
            key=lambda item: abs(item[0] - int(stamp_ns)),
        )
        maximum_skew_ns = int(float(self.config.get("mitt_roi_max_time_skew_ms", 80.0)) * 1e6)
        return nearest_roi if abs(nearest_stamp - int(stamp_ns)) <= maximum_skew_ns else None

    @property
    def guard_progress(self) -> tuple[int, int]:
        return self.guard_count, int(self.config.get("guard_ready_frames", 4))

    def _valid(self, point: Landmark2D | None) -> bool:
        return point is not None and point.confidence >= float(self.config.get("minimum_landmark_confidence", 0.35))

    def _features(self, sample: PoseSample) -> dict[str, _ArmFeature] | None:
        landmarks = sample.landmarks
        shoulders = [landmarks.get("left_shoulder"), landmarks.get("right_shoulder")]
        if not all(self._valid(point) for point in shoulders):
            return None
        shoulder_points = [np.asarray(point.pixel, dtype=np.float64) for point in shoulders if point is not None]
        scale = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))
        if scale < 15.0:
            return None
        features: dict[str, _ArmFeature] = {}
        for side in SIDES:
            shoulder = landmarks.get(f"{side}_shoulder")
            elbow = landmarks.get(f"{side}_elbow")
            wrist = landmarks.get(f"{side}_wrist")
            if not all(self._valid(point) for point in (shoulder, elbow, wrist)):
                continue
            shoulder_xy = np.asarray(shoulder.pixel, dtype=np.float64)
            elbow_xy = np.asarray(elbow.pixel, dtype=np.float64)
            wrist_xy = np.asarray(wrist.pixel, dtype=np.float64)
            contact_xy = wrist_xy + float(self.config.get("fist_extension_forearm_ratio", 0.35)) * (
                wrist_xy - elbow_xy
            )
            reach = float(np.linalg.norm(wrist_xy - shoulder_xy) / scale)
            elbow_angle = _angle_deg(shoulder_xy, elbow_xy, wrist_xy)
            previous = self._previous.get(side)
            velocity = np.zeros(2, dtype=np.float64)
            planar_speed = 0.0
            extension_speed = 0.0
            if previous is not None and sample.stamp_ns > previous.stamp_ns:
                dt = (sample.stamp_ns - previous.stamp_ns) / 1e9
                if dt <= 0.20:
                    velocity = (wrist_xy - previous.wrist) / scale / dt
                    planar_speed = float(np.linalg.norm(velocity))
                    extension_speed = max(0.0, elbow_angle - previous.elbow_angle) / 45.0 / dt
            features[side] = _ArmFeature(
                stamp_ns=sample.stamp_ns,
                wrist=wrist_xy,
                contact=contact_xy,
                velocity=velocity,
                speed=max(planar_speed, extension_speed),
                reach=reach,
                elbow_angle=elbow_angle,
            )
        return features

    def _guard_ok(self, sample: PoseSample, features: Mapping[str, _ArmFeature]) -> bool:
        nose = sample.landmarks.get("nose")
        shoulders = [sample.landmarks.get("left_shoulder"), sample.landmarks.get("right_shoulder")]
        if not self._valid(nose) or not all(self._valid(point) for point in shoulders) or len(features) < 2:
            return False
        scale = np.linalg.norm(np.asarray(shoulders[0].pixel) - np.asarray(shoulders[1].pixel))
        maximum_ratio = float(self.config.get("guard_max_wrist_nose_ratio", 1.85))
        maximum_speed = float(self.config.get("guard_max_speed_body_s", 0.45))
        nose_xy = np.asarray(nose.pixel, dtype=np.float64)
        return all(
            np.linalg.norm(features[side].wrist - nose_xy) / max(scale, 1.0) <= maximum_ratio
            and features[side].speed <= maximum_speed
            for side in SIDES
        )

    def _reset_active(self, next_state: str) -> None:
        self.state = next_state
        self._active_side = None
        self._peak_speed = 0.0
        self._previous_velocity = None
        self._settle_count = 0
        self._best_contact = None
        self._start_confirm = {side: 0 for side in SIDES}

    def update(self, sample: PoseSample) -> ImpactEvent | None:
        if sample.camera != "front":
            return None
        features = self._features(sample)
        if features is None or not features:
            self._lost_frames += 1
            if self._lost_frames > int(self.config.get("max_lost_frames", 6)):
                self.guard_count = 0
                self._baseline.clear()
                self._reset_active("WAIT_GUARD")
            return None
        self._lost_frames = 0
        if self.state == "IMPACT":
            if sample.stamp_ns >= self._impact_until_ns:
                self.state = "COOLDOWN"
            self._previous.update(features)
            return None
        if self.state == "COOLDOWN":
            if sample.stamp_ns >= self._cooldown_until_ns:
                self.guard_count = 0
                self._baseline.clear()
                self._reset_active("WAIT_GUARD")
            self._previous.update(features)
            return None
        if self.state == "WAIT_GUARD":
            if self._guard_ok(sample, features):
                self.guard_count += 1
                self._baseline = dict(features)
                if self.guard_count >= int(self.config.get("guard_ready_frames", 4)):
                    self.state = "READY"
            else:
                self.guard_count = max(0, self.guard_count - 1)
            self._previous.update(features)
            return None
        if self.state == "READY":
            if self._guard_ok(sample, features):
                self._baseline = dict(features)
            for side, feature in features.items():
                baseline = self._baseline.get(side)
                if baseline is None:
                    continue
                displacement = float(np.linalg.norm(feature.wrist - baseline.wrist))
                shoulder_scale = max(
                    np.linalg.norm(
                        np.asarray(sample.landmarks["left_shoulder"].pixel)
                        - np.asarray(sample.landmarks["right_shoulder"].pixel)
                    ),
                    1.0,
                )
                displacement /= shoulder_scale
                extension_gain = feature.elbow_angle - baseline.elbow_angle
                starts = feature.speed >= float(self.config.get("start_speed_body_s", 0.90)) and (
                    displacement >= float(self.config.get("start_displacement_body", 0.10))
                    or extension_gain >= float(self.config.get("start_extension_gain_deg", 7.0))
                )
                self._start_confirm[side] = self._start_confirm[side] + 1 if starts else 0
                if self._start_confirm[side] >= int(self.config.get("start_confirm_frames", 2)):
                    self.state = "ACTIVE"
                    self._active_side = side
                    self._active_started_ns = sample.stamp_ns
                    self._peak_speed = feature.speed
                    self._previous_velocity = feature.velocity.copy()
                    self._best_contact = _ContactCandidate(
                        sample.stamp_ns,
                        tuple(feature.wrist),
                        tuple(feature.contact),
                        displacement + max(extension_gain, 0.0) / 90.0,
                        feature.speed,
                        displacement,
                        extension_gain,
                    )
                    break
            self._previous.update(features)
            return None
        side = self._active_side
        feature = features.get(side) if side is not None else None
        baseline = self._baseline.get(side) if side is not None else None
        if feature is None or baseline is None:
            self._lost_frames += 1
            if self._lost_frames > int(self.config.get("max_lost_frames", 6)):
                self.guard_count = 0
                self._baseline.clear()
                self._reset_active("WAIT_GUARD")
            self._previous.update(features)
            return None
        shoulder_scale = max(
            np.linalg.norm(
                np.asarray(sample.landmarks["left_shoulder"].pixel)
                - np.asarray(sample.landmarks["right_shoulder"].pixel)
            ),
            1.0,
        )
        displacement = float(np.linalg.norm(feature.wrist - baseline.wrist) / shoulder_scale)
        extension_gain = feature.elbow_angle - baseline.elbow_angle
        contact_score = displacement + 0.60 * max(feature.reach - baseline.reach, 0.0) + max(extension_gain, 0.0) / 90.0
        candidate = _ContactCandidate(
            sample.stamp_ns,
            tuple(feature.wrist),
            tuple(feature.contact),
            contact_score,
            feature.speed,
            displacement,
            extension_gain,
        )
        # Keep the earliest frame on a reach plateau. That frame is closer to
        # first contact than a later frame where the glove is already stopped.
        if self._best_contact is None or candidate.score > self._best_contact.score + 1e-6:
            self._best_contact = candidate
        self._peak_speed = max(self._peak_speed, feature.speed)
        reversal = False
        if self._previous_velocity is not None:
            norms = np.linalg.norm(self._previous_velocity) * np.linalg.norm(feature.velocity)
            if norms > 1e-8:
                cosine = float(np.dot(self._previous_velocity, feature.velocity) / norms)
                reversal = cosine <= float(self.config.get("reversal_cosine", 0.10))
        decelerated = feature.speed <= self._peak_speed * float(self.config.get("deceleration_ratio", 0.58))
        self._settle_count = self._settle_count + 1 if decelerated or reversal else 0
        self._previous_velocity = feature.velocity.copy()
        duration_s = (sample.stamp_ns - self._active_started_ns) / 1e9
        best = self._best_contact
        enough_contact = best is not None and best.displacement >= float(
            self.config.get("minimum_contact_displacement_body", 0.20)
        ) and (
            best.extension_gain >= float(self.config.get("minimum_contact_extension_gain_deg", 12.0))
            or best.displacement
            >= float(self.config.get("minimum_contact_displacement_without_extension_body", 0.35))
        )
        roi = self._mitt_roi_at(sample.stamp_ns)
        roi_valid = isinstance(roi, (list, tuple)) and len(roi) == 4
        inside_mitt = not bool(self.config.get("require_mitt_roi", False))
        if best is not None and roi_valid:
            width, height = sample.image_size
            x = best.contact_pixel[0] / max(width, 1)
            y = best.contact_pixel[1] / max(height, 1)
            inside_mitt = float(roi[0]) <= x <= float(roi[2]) and float(roi[1]) <= y <= float(roi[3])
        minimum_duration_reached = duration_s >= float(self.config.get("minimum_duration_s", 0.10))
        motion_settled = self._settle_count >= int(self.config.get("contact_settle_frames", 2))
        confirmed = (
            minimum_duration_reached
            and self._peak_speed >= float(self.config.get("minimum_peak_speed_body_s", 1.0))
            and enough_contact
            and inside_mitt
            and motion_settled
        )
        event = None
        if confirmed and best is not None:
            self._impact_id += 1
            confidence = float(
                np.clip(
                    0.35
                    + 0.20 * min(self._peak_speed / 3.0, 1.0)
                    + 0.25 * min(best.score / 0.65, 1.0)
                    + 0.20 * min(self._settle_count / 3.0, 1.0),
                    0.0,
                    1.0,
                )
            )
            event = ImpactEvent(
                impact_id=self._impact_id,
                stamp_ns=best.stamp_ns,
                detected_stamp_ns=sample.stamp_ns,
                side=str(side),
                source="vision_candidate",
                wrist_pixel_front=best.wrist_pixel,
                confidence=confidence,
                peak_speed_body_s=self._peak_speed,
                displacement_body=best.displacement,
                metadata={
                    "extension_gain_deg": round(best.extension_gain, 3),
                    "detection_delay_ms": round((sample.stamp_ns - best.stamp_ns) / 1e6, 3),
                    "contact_pixel_front_x": round(best.contact_pixel[0], 3),
                    "contact_pixel_front_y": round(best.contact_pixel[1], 3),
                    "mitt_roi_used": roi_valid,
                    "mitt_roi_x1": round(float(roi[0]), 6) if roi_valid else -1.0,
                    "mitt_roi_y1": round(float(roi[1]), 6) if roi_valid else -1.0,
                    "mitt_roi_x2": round(float(roi[2]), 6) if roi_valid else -1.0,
                    "mitt_roi_y2": round(float(roi[3]), 6) if roi_valid else -1.0,
                },
            )
            cooldown_ns = int(float(self.config.get("cooldown_s", 0.70)) * 1e9)
            impact_display_ns = int(float(self.config.get("impact_display_s", 0.22)) * 1e9)
            self._impact_until_ns = sample.stamp_ns + min(max(0, impact_display_ns), cooldown_ns)
            self._cooldown_until_ns = sample.stamp_ns + cooldown_ns
            self._reset_active("IMPACT")
        elif minimum_duration_reached and motion_settled:
            self._reset_active("READY")
        elif duration_s > float(self.config.get("maximum_duration_s", 1.40)):
            self._reset_active("READY")
        self._previous.update(features)
        return event

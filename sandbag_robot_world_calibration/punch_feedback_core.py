from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


LANDMARK_NAMES = (
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


@dataclass(frozen=True)
class Landmark2D:
    x: float
    y: float
    visibility: float
    # MediaPipe's normalized depth. Smaller values are closer to the camera.
    # Kept optional so recorded/test 2D samples remain compatible.
    z: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.asarray((self.x, self.y), dtype=np.float64)


@dataclass
class PoseSample:
    stamp_s: float
    landmarks: Mapping[str, Landmark2D]
    image: np.ndarray | None = None


@dataclass
class PunchEvent:
    side: str
    samples: list[PoseSample]
    max_speed: float


@dataclass
class ClassifiedPunch:
    punch_type: str
    side: str
    confidence: float
    key_sample: PoseSample
    motion_features: dict[str, float]
    classification_reason: str = ""


@dataclass(frozen=True)
class FeatureError:
    feature: str
    joint: str
    code: str
    value: float
    target: float
    tolerance: float
    weight: float
    error_ratio: float


@dataclass
class ScoreResult:
    total_score: float
    feedback_required: bool
    feature_errors: list[FeatureError]
    violations: list[FeatureError]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(ba, bc) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def shoulder_width(sample: PoseSample) -> float:
    left = sample.landmarks["left_shoulder"].xy
    right = sample.landmarks["right_shoulder"].xy
    return max(float(np.linalg.norm(left - right)), 1e-4)


def shoulder_center(sample: PoseSample) -> np.ndarray:
    left = sample.landmarks["left_shoulder"].xy
    right = sample.landmarks["right_shoulder"].xy
    return (left + right) * 0.5


def torso_relative_wrist_xy(sample: PoseSample, side: str) -> np.ndarray:
    wrist_name = side_landmark(side, "wrist")
    return (
        sample.landmarks[wrist_name].xy - shoulder_center(sample)
    ) / shoulder_width(sample)


def side_landmark(side: str, joint: str) -> str:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side}")
    if joint not in ("shoulder", "elbow", "wrist", "hip"):
        raise ValueError(f"Unsupported joint: {joint}")
    return f"{side}_{joint}"


def opposite_side(side: str) -> str:
    return "right" if side == "left" else "left"


def required_visibility(sample: PoseSample, side: str) -> float:
    other = opposite_side(side)
    names = (
        "nose",
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
        f"{other}_wrist",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    return min(sample.landmarks[name].visibility for name in names)


def normalized_wrist_speed(
    previous: PoseSample,
    current: PoseSample,
    side: str,
) -> float:
    dt = current.stamp_s - previous.stamp_s
    if dt <= 1e-5 or dt > 0.5:
        return 0.0
    displacement = np.linalg.norm(
        torso_relative_wrist_xy(current, side)
        - torso_relative_wrist_xy(previous, side)
    )
    return float(displacement / dt)


class PunchDetector:
    """Detect a punch with an explicit, latched READY state."""

    def __init__(self, settings: Mapping[str, Any]):
        self.start_speed = float(settings.get("start_speed", 1.2))
        self.guard_speed = float(settings.get("guard_speed", 0.35))
        self.end_speed = float(settings.get("end_speed", 0.45))
        self.ready_frames = int(settings.get("ready_frames", 8))
        self.end_frames = int(settings.get("end_frames", 5))
        self.min_duration_s = float(settings.get("min_duration_s", 0.12))
        self.max_duration_s = float(settings.get("max_duration_s", 1.5))
        self.cooldown_s = float(settings.get("cooldown_s", 0.5))
        self.min_visibility = float(settings.get("min_visibility", 0.55))
        self.speed_window = max(int(settings.get("speed_window", 3)), 1)
        self.start_displacement_ratio = float(
            settings.get("start_displacement_ratio", 0.12)
        )
        self.start_extension_deg = float(settings.get("start_extension_deg", 12.0))
        self.start_confirm_frames = max(
            int(settings.get("start_confirm_frames", 2)),
            1,
        )
        self.guard_max_wrist_face_ratio = float(
            settings.get("guard_max_wrist_face_ratio", 1.6)
        )
        self.max_lost_visibility_frames = int(
            settings.get("max_lost_visibility_frames", 8)
        )

        self.previous: PoseSample | None = None
        self.active_side: str | None = None
        self.samples: list[PoseSample] = []
        self.max_speed = 0.0
        self.stable_frames = 0
        self.slow_frames = 0
        self.cooldown_until = 0.0
        self.guard_baseline: PoseSample | None = None
        self.lost_visibility_frames = 0
        self._state = "WAIT_GUARD"
        self.speed_history = {
            side: deque(maxlen=self.speed_window) for side in ("left", "right")
        }
        self.start_confirm_count = {"left": 0, "right": 0}
        self._telemetry: dict[str, float] = {
            "left_speed": 0.0,
            "right_speed": 0.0,
            "left_displacement": 0.0,
            "right_displacement": 0.0,
            "left_elbow_delta": 0.0,
            "right_elbow_delta": 0.0,
            "left_start_conditions": 0.0,
            "right_start_conditions": 0.0,
        }

    @property
    def state(self) -> str:
        if self._state == "ACTIVE" and self.active_side is not None:
            return f"ACTIVE_{self.active_side.upper()}"
        return self._state

    @property
    def telemetry(self) -> dict[str, float]:
        values = dict(self._telemetry)
        values.update(
            {
                "guard_frames": float(self.stable_frames),
                "ready_frames": float(self.ready_frames),
                "left_start_confirm": float(self.start_confirm_count["left"]),
                "right_start_confirm": float(self.start_confirm_count["right"]),
                "start_confirm_frames": float(self.start_confirm_frames),
            }
        )
        return values

    def reset(self) -> None:
        self.previous = None
        self.active_side = None
        self.samples = []
        self.max_speed = 0.0
        self.stable_frames = 0
        self.slow_frames = 0
        self.guard_baseline = None
        self.lost_visibility_frames = 0
        self._state = "WAIT_GUARD"
        self.start_confirm_count = {"left": 0, "right": 0}
        for history in self.speed_history.values():
            history.clear()

    def _smoothed_speeds(
        self,
        previous: PoseSample,
        current: PoseSample,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for side in ("left", "right"):
            raw_speed = normalized_wrist_speed(previous, current, side)
            self.speed_history[side].append(raw_speed)
            result[side] = float(np.median(self.speed_history[side]))
        return result

    def _pose_visible(self, sample: PoseSample) -> bool:
        return all(
            required_visibility(sample, side) >= self.min_visibility
            for side in ("left", "right")
        )

    def _guard_pose_valid(self, sample: PoseSample) -> bool:
        if not self._pose_visible(sample):
            return False
        scale = shoulder_width(sample)
        nose = sample.landmarks["nose"].xy
        return all(
            np.linalg.norm(sample.landmarks[f"{side}_wrist"].xy - nose) / scale
            <= self.guard_max_wrist_face_ratio
            for side in ("left", "right")
        )

    def _set_guard_baseline(self, sample: PoseSample) -> None:
        self.guard_baseline = sample
        self.start_confirm_count = {"left": 0, "right": 0}

    def _update_telemetry(
        self,
        sample: PoseSample,
        speeds: Mapping[str, float],
    ) -> None:
        for side in ("left", "right"):
            self._telemetry[f"{side}_speed"] = float(speeds[side])
            displacement = 0.0
            elbow_delta = 0.0
            if self.guard_baseline is not None:
                displacement = float(
                    np.linalg.norm(
                        torso_relative_wrist_xy(sample, side)
                        - torso_relative_wrist_xy(self.guard_baseline, side)
                    )
                )
                elbow_delta = max(
                    0.0,
                    _elbow_angle(sample, side)
                    - _elbow_angle(self.guard_baseline, side),
                )

            speed_hit = speeds[side] >= self.start_speed
            displacement_hit = displacement >= self.start_displacement_ratio
            extension_hit = elbow_delta >= self.start_extension_deg
            condition_count = int(speed_hit) + int(displacement_hit) + int(extension_hit)

            self._telemetry[f"{side}_displacement"] = displacement
            self._telemetry[f"{side}_elbow_delta"] = elbow_delta
            self._telemetry[f"{side}_start_conditions"] = float(condition_count)

    def _ready_start_side(
        self,
        sample: PoseSample,
        speeds: Mapping[str, float],
    ) -> str | None:
        candidates: list[str] = []
        for side in ("left", "right"):
            speed_hit = speeds[side] >= self.start_speed
            displacement_hit = (
                self._telemetry[f"{side}_displacement"]
                >= self.start_displacement_ratio
            )
            extension_hit = (
                self._telemetry[f"{side}_elbow_delta"] >= self.start_extension_deg
            )
            visible = required_visibility(sample, side) >= self.min_visibility

            # Always require real speed plus one pose-change condition.
            if speed_hit and (displacement_hit or extension_hit) and visible:
                self.start_confirm_count[side] += 1
            else:
                self.start_confirm_count[side] = 0

            if self.start_confirm_count[side] >= self.start_confirm_frames:
                candidates.append(side)

        if not candidates:
            return None
        return max(candidates, key=lambda side: speeds[side])

    def _return_to_wait_guard(self) -> None:
        self._state = "WAIT_GUARD"
        self.active_side = None
        self.samples = []
        self.stable_frames = 0
        self.slow_frames = 0
        self.guard_baseline = None
        self.start_confirm_count = {"left": 0, "right": 0}

    def update(self, sample: PoseSample) -> PunchEvent | None:
        if self.previous is None:
            self.previous = sample
            return None

        speeds = self._smoothed_speeds(self.previous, sample)
        pose_visible = self._pose_visible(sample)
        self.lost_visibility_frames = (
            0 if pose_visible else self.lost_visibility_frames + 1
        )
        self._update_telemetry(sample, speeds)

        if self._state == "COOLDOWN":
            if sample.stamp_s < self.cooldown_until:
                self.previous = sample
                return None
            self._return_to_wait_guard()

        if self._state == "WAIT_GUARD":
            if self._guard_pose_valid(sample) and max(speeds.values()) <= self.guard_speed:
                self.stable_frames += 1
            else:
                self.stable_frames = max(0, self.stable_frames - 1)

            if self.stable_frames >= self.ready_frames:
                self._state = "READY"
                self._set_guard_baseline(sample)

            self.previous = sample
            return None

        if self._state == "READY":
            if self.lost_visibility_frames >= self.max_lost_visibility_frames:
                self._return_to_wait_guard()
                self.previous = sample
                return None

            # Refresh the reference while the user is still calmly guarding.
            if self._guard_pose_valid(sample) and max(speeds.values()) <= self.guard_speed:
                self._set_guard_baseline(sample)
                self._update_telemetry(sample, speeds)

            start_side = self._ready_start_side(sample, speeds)
            if start_side is not None:
                self._state = "ACTIVE"
                self.active_side = start_side
                baseline = self.guard_baseline or self.previous
                self.samples = [baseline, sample]
                self.max_speed = float(speeds[start_side])
                self.slow_frames = 0

            self.previous = sample
            return None

        if self._state != "ACTIVE" or self.active_side is None:
            self._return_to_wait_guard()
            self.previous = sample
            return None

        side = self.active_side
        speed = speeds[side]
        self.samples.append(sample)
        self.max_speed = max(self.max_speed, speed)
        self.slow_frames = self.slow_frames + 1 if speed <= self.end_speed else 0

        duration = sample.stamp_s - self.samples[0].stamp_s
        finished = (
            duration >= self.min_duration_s and self.slow_frames >= self.end_frames
        ) or duration >= self.max_duration_s

        self.previous = sample
        if not finished:
            return None

        event = PunchEvent(
            side=side,
            samples=list(self.samples),
            max_speed=self.max_speed,
        )
        self._state = "COOLDOWN"
        self.active_side = None
        self.samples = []
        self.max_speed = 0.0
        self.stable_frames = 0
        self.slow_frames = 0
        self.guard_baseline = None
        self.start_confirm_count = {"left": 0, "right": 0}
        self.cooldown_until = sample.stamp_s + self.cooldown_s
        return event


def _elbow_angle(sample: PoseSample, side: str) -> float:
    shoulder = sample.landmarks[side_landmark(side, "shoulder")].xy
    elbow = sample.landmarks[side_landmark(side, "elbow")].xy
    wrist = sample.landmarks[side_landmark(side, "wrist")].xy
    return calculate_angle(shoulder, elbow, wrist)


def _first_near_peak_index(values: np.ndarray, fraction: float) -> int:
    if values.size == 0:
        return 0
    peak = float(np.max(values))
    if peak <= 1e-9:
        return 0
    candidates = np.flatnonzero(values >= peak * fraction)
    return int(candidates[0]) if candidates.size else int(np.argmax(values))


def _trajectory_geometry(points: np.ndarray) -> tuple[float, float, float, float]:
    """Return path length, direct distance, linearity, and relative curvature."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 1.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    direct_vector = points[-1] - points[0]
    direct_distance = float(np.linalg.norm(direct_vector))
    if path_length <= 1e-9 or direct_distance <= 1e-9:
        return path_length, direct_distance, 0.0, 1.0

    linearity = float(np.clip(direct_distance / path_length, 0.0, 1.0))
    relative_points = points - points[0]
    cross_magnitudes = np.abs(
        direct_vector[0] * relative_points[:, 1]
        - direct_vector[1] * relative_points[:, 0]
    )
    max_perpendicular = float(np.max(cross_magnitudes) / direct_distance)
    curvature = max_perpendicular / direct_distance
    return path_length, direct_distance, linearity, curvature


def classify_punch(
    event: PunchEvent,
    settings: Mapping[str, Any],
) -> ClassifiedPunch:
    if not event.samples:
        raise ValueError("PunchEvent must contain at least one pose sample")

    side = event.side
    wrist_name = side_landmark(side, "wrist")
    start = event.samples[0]

    all_wrist_points = np.asarray(
        [sample.landmarks[wrist_name].xy for sample in event.samples],
        dtype=np.float64,
    )
    elbow_name = side_landmark(side, "elbow")
    shoulder_name = side_landmark(side, "shoulder")
    all_elbow_points = np.asarray(
        [sample.landmarks[elbow_name].xy for sample in event.samples],
        dtype=np.float64,
    )
    all_elbow_angles = np.asarray(
        [_elbow_angle(sample, side) for sample in event.samples],
        dtype=np.float64,
    )

    vertical_aspect = 1.0
    if start.image is not None and start.image.shape[1] > 0:
        vertical_aspect = float(start.image.shape[0] / start.image.shape[1])

    sample_scales = np.asarray(
        [shoulder_width(sample) for sample in event.samples],
        dtype=np.float64,
    )
    shoulder_centers = np.asarray(
        [shoulder_center(sample) for sample in event.samples],
        dtype=np.float64,
    )
    torso_relative_points = (all_wrist_points - shoulder_centers) / sample_scales[:, None]
    torso_relative_points[:, 1] *= vertical_aspect
    trajectory_deltas = torso_relative_points - torso_relative_points[0]
    all_planar_displacements = np.linalg.norm(trajectory_deltas, axis=1)

    # Compensate for torso motion by measuring the wrist depth relative to the
    # punching-side shoulder. MediaPipe z becomes smaller toward the camera.
    all_forward_depth_values = np.asarray(
        [
            (
                sample.landmarks[shoulder_name].z
                - sample.landmarks[wrist_name].z
            )
            / sample_scale
            for sample, sample_scale in zip(event.samples, sample_scales)
        ],
        dtype=np.float64,
    )
    all_forward_depth_gains = (
        all_forward_depth_values - all_forward_depth_values[0]
    )
    all_extension_gains = np.maximum(all_elbow_angles - all_elbow_angles[0], 0.0)

    # ACTIVE can include the hand returning to guard. Estimate the first impact
    # from the earliest near-peaks of planar reach, forward depth, and elbow
    # extension, then ignore every recovery frame after the latest of them.
    impact_peak_fraction = float(settings.get("impact_peak_fraction", 0.92))
    impact_peak_fraction = float(np.clip(impact_peak_fraction, 0.50, 1.0))
    impact_index = max(
        _first_near_peak_index(all_planar_displacements, impact_peak_fraction),
        _first_near_peak_index(
            np.maximum(all_forward_depth_gains, 0.0),
            impact_peak_fraction,
        ),
        _first_near_peak_index(all_extension_gains, impact_peak_fraction),
    )
    if len(event.samples) > 1:
        impact_index = max(1, impact_index)
    impact_index = min(impact_index, len(event.samples) - 1)
    outbound = slice(0, impact_index + 1)

    wrist_points = all_wrist_points[outbound]
    elbow_points = all_elbow_points[outbound]
    elbow_angles = all_elbow_angles[outbound]
    trajectory_points = torso_relative_points[outbound]
    trajectory_deltas = trajectory_points - trajectory_points[0]
    planar_displacement_values = np.linalg.norm(trajectory_deltas, axis=1)
    lateral_values = np.abs(trajectory_deltas[:, 0])
    vertical_values = np.abs(trajectory_deltas[:, 1])
    upward_values = -trajectory_deltas[:, 1]

    upward = float(np.max(upward_values))
    lateral = float(np.max(lateral_values))
    vertical_travel = float(np.max(vertical_values))
    planar_travel = float(np.max(planar_displacement_values))
    extension_gain = float(np.max(elbow_angles) - elbow_angles[0])
    forward_depth_values = all_forward_depth_values[outbound]
    forward_depth_gain_values = forward_depth_values - forward_depth_values[0]
    forward_depth_gain = float(np.max(forward_depth_gain_values))

    path_length, direct_travel, path_linearity, path_curvature = (
        _trajectory_geometry(trajectory_points)
    )

    start_outward_offset = float(trajectory_points[0, 0])
    outward_sign = 1.0 if start_outward_offset >= 0.0 else -1.0
    outward_positions = outward_sign * trajectory_points[:, 0]
    inward_values = np.maximum.accumulate(outward_positions) - outward_positions
    inward = float(np.max(inward_values))

    wrist_above_elbow_values = (
        (elbow_points[:, 1] - wrist_points[:, 1])
        * vertical_aspect
        / sample_scales[outbound]
    )
    direction_angle = float(
        np.degrees(np.arctan2(max(upward, 0.0), max(lateral, 1e-6)))
    )

    uppercut_min_up = float(settings.get("uppercut_min_upward_ratio", 0.30))
    uppercut_max_elbow = float(settings.get("uppercut_max_elbow_angle", 145.0))
    uppercut_vertical_dominance = float(
        settings.get("uppercut_vertical_dominance", 1.05)
    )
    uppercut_min_wrist_above = float(
        settings.get("uppercut_min_wrist_above_elbow_ratio", 0.10)
    )
    hook_min_lateral = float(settings.get("hook_min_lateral_ratio", 0.45))
    hook_max_elbow = float(settings.get("hook_max_elbow_angle", 145.0))
    hook_min_inward = float(settings.get("hook_min_inward_ratio", 0.15))
    hook_lateral_dominance = float(settings.get("hook_lateral_dominance", 1.0))
    hook_max_wrist_elbow_height = float(
        settings.get("hook_max_wrist_elbow_height_ratio", 0.35)
    )
    hook_fallback_lateral_scale = float(
        settings.get("hook_fallback_lateral_scale", 0.85)
    )
    hook_fallback_inward_scale = float(
        settings.get("hook_fallback_inward_scale", 0.75)
    )
    hook_fallback_lateral_dominance = float(
        settings.get("hook_fallback_lateral_dominance", 1.05)
    )
    hook_fallback_elbow_slack = float(
        settings.get("hook_fallback_elbow_slack_deg", 5.0)
    )
    hook_min_path_curvature = float(
        settings.get("hook_min_path_curvature_ratio", 0.18)
    )
    hook_max_path_linearity = float(
        settings.get("hook_max_path_linearity", 0.88)
    )
    hook_strong_arc_min_curvature = float(
        settings.get("hook_strong_arc_min_curvature_ratio", 0.25)
    )
    hook_strong_arc_max_linearity = float(
        settings.get("hook_strong_arc_max_linearity", 0.82)
    )
    hook_strong_arc_max_elbow = float(
        settings.get("hook_strong_arc_max_elbow_angle", 160.0)
    )
    straight_min_peak_elbow = float(
        settings.get("straight_min_peak_elbow_angle", 150.0)
    )
    straight_min_extension_gain = float(
        settings.get("straight_min_extension_gain_deg", 25.0)
    )
    straight_min_depth_gain = float(
        settings.get("straight_min_depth_gain_ratio", 0.18)
    )
    straight_depth_min_peak_elbow = float(
        settings.get("straight_depth_min_peak_elbow_angle", 125.0)
    )
    straight_depth_min_extension_gain = float(
        settings.get("straight_depth_min_extension_gain_deg", 8.0)
    )
    straight_max_planar_travel = float(
        settings.get("straight_max_planar_travel_ratio", 0.32)
    )
    straight_planar_min_peak_elbow = float(
        settings.get("straight_planar_min_peak_elbow_angle", 132.0)
    )
    straight_planar_min_extension_gain = float(
        settings.get("straight_planar_min_extension_gain_deg", 8.0)
    )
    straight_min_direct_travel = float(
        settings.get("straight_min_direct_travel_ratio", 0.12)
    )
    straight_min_path_linearity = float(
        settings.get("straight_min_path_linearity", 0.88)
    )
    straight_max_path_curvature = float(
        settings.get("straight_max_path_curvature_ratio", 0.18)
    )
    straight_path_min_peak_elbow = float(
        settings.get("straight_path_min_peak_elbow_angle", 123.0)
    )
    straight_path_min_extension_gain = float(
        settings.get("straight_path_min_extension_gain_deg", 6.0)
    )
    straight_priority_peak_elbow = float(
        settings.get("straight_priority_peak_elbow_angle", 150.0)
    )
    straight_priority_extension_gain = float(
        settings.get("straight_priority_extension_gain_deg", 25.0)
    )

    upper_index = int(np.argmax(upward_values))
    straight_index = int(np.argmax(elbow_angles))
    depth_index = int(np.argmax(forward_depth_gain_values))
    max_elbow_angle = float(elbow_angles[straight_index])
    impact_elbow_angle = float(elbow_angles[-1])

    hook_path_curved = (
        path_curvature >= hook_min_path_curvature
        and path_linearity <= hook_max_path_linearity
    )
    hook_arc_evidence = inward >= hook_min_inward or hook_path_curved
    hook_direction_pass = (
        lateral >= hook_min_lateral
        and lateral >= vertical_travel * hook_lateral_dominance
    )

    hook_detected = (
        hook_direction_pass
        and hook_arc_evidence
        and impact_elbow_angle <= hook_max_elbow
        and abs(wrist_above_elbow_values[-1])
        <= hook_max_wrist_elbow_height
    )
    hook_priority_detected = (
        hook_direction_pass
        and path_curvature >= hook_strong_arc_min_curvature
        and path_linearity <= hook_strong_arc_max_linearity
        and impact_elbow_angle <= hook_strong_arc_max_elbow
    )
    uppercut_detected = (
        upward >= uppercut_min_up
        and upward >= lateral * uppercut_vertical_dominance
        and elbow_angles[upper_index] <= uppercut_max_elbow
        and wrist_above_elbow_values[upper_index] >= uppercut_min_wrist_above
    )
    straight_2d_detected = (
        max_elbow_angle >= straight_min_peak_elbow
        and extension_gain >= straight_min_extension_gain
    )
    straight_depth_detected = (
        forward_depth_gain >= straight_min_depth_gain
        and max_elbow_angle >= straight_depth_min_peak_elbow
        and extension_gain >= straight_depth_min_extension_gain
    )
    # A punch toward a frontal camera has little x/y travel. Require some arm
    # extension as well so pose jitter or a stationary guard cannot qualify.
    straight_low_planar_detected = (
        planar_travel <= straight_max_planar_travel
        and max_elbow_angle >= straight_planar_min_peak_elbow
        and extension_gain >= straight_planar_min_extension_gain
    )
    straight_path_linear = (
        path_linearity >= straight_min_path_linearity
        and path_curvature <= straight_max_path_curvature
    )
    straight_linear_path_detected = (
        direct_travel >= straight_min_direct_travel
        and straight_path_linear
        and max_elbow_angle >= straight_path_min_peak_elbow
        and extension_gain >= straight_path_min_extension_gain
    )
    straight_detected = (
        straight_2d_detected
        or straight_depth_detected
        or straight_low_planar_detected
        or straight_linear_path_detected
    )
    strong_2d_straight = (
        max_elbow_angle >= straight_priority_peak_elbow
        and extension_gain >= straight_priority_extension_gain
        and straight_path_linear
    )
    straight_priority_detected = (
        straight_depth_detected
        or straight_low_planar_detected
        or straight_linear_path_detected
        or strong_2d_straight
    )

    hook_fallback = (
        not straight_detected
        and lateral >= hook_min_lateral * hook_fallback_lateral_scale
        and inward >= hook_min_inward * hook_fallback_inward_scale
        and lateral >= vertical_travel * hook_fallback_lateral_dominance
        and impact_elbow_angle
        <= hook_max_elbow + hook_fallback_elbow_slack
    )
    uppercut_fallback = (
        not straight_detected
        and upward >= uppercut_min_up * 0.70
        and upward > lateral
        and wrist_above_elbow_values[upper_index] > 0.0
    )

    # A clearly curved outbound path overrides elbow extension. This avoids
    # labelling a wide hook as straight just because its impact frame happens
    # to look extended in 2D.
    if hook_priority_detected:
        punch_type = "hook"
        key_index = impact_index
        classification_reason = (
            "hook_outside_to_inside"
            if inward >= hook_min_inward
            else "hook_curved_outbound_path"
        )
        confidence = 0.65 + min(
            0.30,
            max(
                path_curvature - hook_strong_arc_min_curvature,
                hook_strong_arc_max_linearity - path_linearity,
            ),
        )
    elif straight_priority_detected:
        punch_type = "straight"
        key_index = impact_index
        if straight_depth_detected:
            classification_reason = "straight_forward_depth_and_extension"
        elif straight_low_planar_detected:
            classification_reason = "straight_low_planar_motion_and_extension"
        elif straight_linear_path_detected:
            classification_reason = "straight_linear_outbound_path"
        else:
            classification_reason = "straight_strong_elbow_extension"
        depth_margin = max(forward_depth_gain - straight_min_depth_gain, 0.0)
        planar_margin = max(straight_max_planar_travel - planar_travel, 0.0)
        extension_margin = max(
            extension_gain - straight_min_extension_gain,
            0.0,
        ) / 60.0
        confidence = 0.65 + min(
            0.30,
            max(depth_margin, planar_margin, extension_margin),
        )
    # A bent-arm lateral arc can still qualify without the strong-curve override.
    elif hook_detected:
        punch_type = "hook"
        key_index = impact_index
        classification_reason = (
            "hook_outside_to_inside"
            if inward >= hook_min_inward
            else "hook_curved_outbound_path"
        )
        confidence = 0.60 + min(
            0.35,
            max(lateral - hook_min_lateral, inward - hook_min_inward),
        )
    elif uppercut_detected:
        punch_type = "uppercut"
        key_index = upper_index
        classification_reason = "uppercut_vertical_dominant"
        confidence = 0.60 + min(0.35, upward - uppercut_min_up)
    elif straight_detected:
        punch_type = "straight"
        key_index = impact_index
        if straight_depth_detected:
            classification_reason = "straight_forward_depth_and_extension"
        elif straight_low_planar_detected:
            classification_reason = "straight_low_planar_motion_and_extension"
        elif straight_linear_path_detected:
            classification_reason = "straight_linear_outbound_path"
        else:
            classification_reason = "straight_elbow_extension"
        confidence = 0.60 + min(
            0.35,
            max(extension_gain - straight_min_extension_gain, 0.0) / 60.0,
        )
    elif hook_fallback:
        punch_type = "hook"
        key_index = impact_index
        classification_reason = "hook_bent_arm_lateral_fallback"
        confidence = 0.52
    elif uppercut_fallback:
        punch_type = "uppercut"
        key_index = upper_index
        classification_reason = "uppercut_bent_arm_vertical_fallback"
        confidence = 0.52
    else:
        punch_type = "straight"
        key_index = impact_index
        classification_reason = "straight_low_directional_evidence_fallback"
        confidence = 0.40

    motion_features = {
        "upward_ratio": upward,
        "lateral_ratio": lateral,
        "vertical_travel_ratio": vertical_travel,
        "planar_travel_ratio": planar_travel,
        "direct_travel_ratio": direct_travel,
        "path_length_ratio": path_length,
        "path_linearity": path_linearity,
        "path_curvature_ratio": path_curvature,
        "inward_ratio": inward,
        "direction_angle_deg": direction_angle,
        "wrist_above_elbow_ratio": float(wrist_above_elbow_values[key_index]),
        "extension_gain_deg": extension_gain,
        "max_elbow_angle_deg": max_elbow_angle,
        "impact_elbow_angle_deg": impact_elbow_angle,
        "forward_depth_gain_ratio": forward_depth_gain,
        "peak_forward_depth_ratio": float(forward_depth_values[depth_index]),
        "straight_2d_extension_pass": float(straight_2d_detected),
        "straight_depth_pass": float(straight_depth_detected),
        "straight_low_planar_pass": float(straight_low_planar_detected),
        "straight_linear_path_pass": float(straight_linear_path_detected),
        "straight_priority_pass": float(straight_priority_detected),
        "straight_extension_pass": float(straight_detected),
        "hook_path_curve_pass": float(hook_path_curved),
        "hook_priority_pass": float(hook_priority_detected),
        "hook_strict_pass": float(hook_detected),
        "hook_fallback_pass": float(hook_fallback),
        "impact_sample_index": float(impact_index),
        "event_sample_count": float(len(event.samples)),
        "recovery_frames_ignored": float(len(event.samples) - impact_index - 1),
        "max_speed_body_widths_s": float(event.max_speed),
    }
    return ClassifiedPunch(
        punch_type=punch_type,
        side=side,
        confidence=float(np.clip(confidence, 0.0, 0.95)),
        key_sample=event.samples[key_index],
        motion_features=motion_features,
        classification_reason=classification_reason,
    )


def _point_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    line = b - a
    length = float(np.linalg.norm(line))
    if length <= 1e-9:
        return 0.0
    return float(abs(np.cross(line, point - a)) / length)


def extract_form_features(sample: PoseSample, side: str) -> dict[str, float]:
    other = opposite_side(side)
    scale = shoulder_width(sample)
    strike_shoulder = sample.landmarks[f"{side}_shoulder"].xy
    strike_elbow = sample.landmarks[f"{side}_elbow"].xy
    strike_wrist = sample.landmarks[f"{side}_wrist"].xy
    guard_wrist = sample.landmarks[f"{other}_wrist"].xy
    nose = sample.landmarks["nose"].xy

    shoulder_center = (
        sample.landmarks["left_shoulder"].xy
        + sample.landmarks["right_shoulder"].xy
    ) * 0.5
    hip_center = (
        sample.landmarks["left_hip"].xy + sample.landmarks["right_hip"].xy
    ) * 0.5
    torso_vector = shoulder_center - hip_center
    torso_lean = float(
        np.degrees(
            np.arctan2(
                abs(float(torso_vector[0])),
                max(abs(float(torso_vector[1])), 1e-6),
            )
        )
    )

    return {
        "strike_elbow_angle_deg": calculate_angle(
            strike_shoulder,
            strike_elbow,
            strike_wrist,
        ),
        "elbow_flare_ratio": _point_line_distance(
            strike_elbow,
            strike_shoulder,
            strike_wrist,
        )
        / scale,
        "guard_to_face_ratio": float(np.linalg.norm(guard_wrist - nose) / scale),
        "torso_lean_deg": torso_lean,
        "strike_wrist_height_ratio": float(
            (strike_wrist[1] - strike_shoulder[1]) / scale
        ),
        "wrist_elbow_height_ratio": float(
            (strike_wrist[1] - strike_elbow[1]) / scale
        ),
    }


def score_punch(
    classified: ClassifiedPunch,
    profiles: Mapping[str, Any],
    feedback_settings: Mapping[str, Any],
) -> ScoreResult:
    try:
        profile = profiles[classified.punch_type]
    except KeyError as error:
        raise ValueError(
            f"Missing reference profile: {classified.punch_type}"
        ) from error

    values = extract_form_features(classified.key_sample, classified.side)
    errors: list[FeatureError] = []
    weighted_penalty = 0.0
    total_weight = 0.0

    for feature, reference in profile["features"].items():
        if feature not in values:
            raise ValueError(f"Unsupported form feature in YAML: {feature}")
        target = float(reference["target"])
        tolerance = max(float(reference["tolerance"]), 1e-6)
        weight = max(float(reference["weight"]), 0.0)
        value = float(values[feature])
        error_ratio = abs(value - target) / tolerance

        errors.append(
            FeatureError(
                feature=feature,
                joint=str(reference["joint"]),
                code=str(reference["code"]),
                value=value,
                target=target,
                tolerance=tolerance,
                weight=weight,
                error_ratio=error_ratio,
            )
        )
        weighted_penalty += weight * min(error_ratio, 1.0)
        total_weight += weight

    if total_weight <= 0.0:
        raise ValueError(f"Reference profile has zero total weight: {classified.punch_type}")

    total_score = 100.0 * (1.0 - weighted_penalty / total_weight)
    joint_threshold = float(feedback_settings.get("joint_error_threshold", 1.0))
    score_threshold = float(feedback_settings.get("score_threshold", 75.0))
    violations = [error for error in errors if error.error_ratio >= joint_threshold]
    feedback_required = total_score < score_threshold or bool(violations)

    if feedback_required and not violations:
        violations = sorted(errors, key=lambda error: error.error_ratio, reverse=True)[:2]

    return ScoreResult(
        total_score=float(np.clip(total_score, 0.0, 100.0)),
        feedback_required=feedback_required,
        feature_errors=errors,
        violations=violations,
    )


def joint_landmark_names(joint: str, strike_side: str) -> Sequence[str]:
    guard_side = opposite_side(strike_side)
    mapping = {
        "strike_elbow": (f"{strike_side}_elbow",),
        "strike_wrist": (f"{strike_side}_wrist",),
        "guard_wrist": (f"{guard_side}_wrist",),
        "torso": (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ),
    }
    return mapping.get(joint, ())

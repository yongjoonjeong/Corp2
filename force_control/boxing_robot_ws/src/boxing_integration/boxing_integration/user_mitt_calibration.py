"""User-specific mitt calibration from Vision-predicted mitt targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable


PUNCH_ROLES = {"jab", "straight"}
HANDS = {"left", "right"}


def hand_for_punch_role(dominant_hand: str, punch_role: str) -> str:
    dominant = str(dominant_hand).lower()
    role = str(punch_role).lower()
    if dominant not in HANDS:
        raise ValueError("dominant hand must be left or right")
    if role not in PUNCH_ROLES:
        raise ValueError("punch role must be jab or straight")
    if role == "straight":
        return dominant
    return "left" if dominant == "right" else "right"


def reach_calibration_hand(dominant_hand: str) -> str:
    """Use the non-dominant, jab-side arm for the stationary reach check."""

    dominant = str(dominant_hand).lower()
    if dominant not in HANDS:
        raise ValueError("dominant hand must be left or right")
    return "left" if dominant == "right" else "right"


def apply_tool_xy_correction(
    base_pose_mm_deg: Iterable[float], correction_x_mm: float, correction_y_mm: float
) -> tuple[float, float, float, float, float, float]:
    pose = tuple(float(value) for value in base_pose_mm_deg)
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise ValueError("base TCP pose must contain six finite values")
    correction_x = float(correction_x_mm)
    correction_y = float(correction_y_mm)
    if not math.isfinite(correction_x) or not math.isfinite(correction_y):
        raise ValueError("TCP correction must be finite")
    alpha, beta, gamma = (math.radians(value) for value in pose[3:])
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rotation = (
        (ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb),
        (sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb),
        (-sb * cg, sb * sg, cb),
    )
    delta = tuple(
        rotation[row][0] * correction_x + rotation[row][1] * correction_y
        for row in range(3)
    )
    return (
        pose[0] + delta[0],
        pose[1] + delta[1],
        pose[2] + delta[2],
        pose[3],
        pose[4],
        pose[5],
    )


def apply_tool_z_correction(
    base_pose_mm_deg: Iterable[float], correction_z_mm: float
) -> tuple[float, float, float, float, float, float]:
    """Translate a BASE TCP pose along its Tool +Z (mitt-face normal)."""

    pose = tuple(float(value) for value in base_pose_mm_deg)
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise ValueError("base TCP pose must contain six finite values")
    correction = float(correction_z_mm)
    if not math.isfinite(correction):
        raise ValueError("Tool-Z correction must be finite")
    rotation = _rotation_base_from_tcp(pose)
    return (
        pose[0] + rotation[0][2] * correction,
        pose[1] + rotation[1][2] * correction,
        pose[2] + rotation[2][2] * correction,
        pose[3],
        pose[4],
        pose[5],
    )


def normal_force_delta_n(
    corrected_wrench: Iterable[float], baseline_normal_force_n: float
) -> float:
    wrench = tuple(float(value) for value in corrected_wrench)
    baseline = float(baseline_normal_force_n)
    if len(wrench) != 6 or not all(math.isfinite(value) for value in wrench):
        raise ValueError("corrected wrench must contain six finite values")
    if not math.isfinite(baseline):
        raise ValueError("normal-force baseline must be finite")
    return abs(wrench[2] - baseline)


def tool_z_offset_between(
    base_pose_mm_deg: Iterable[float], target_pose_mm_deg: Iterable[float]
) -> float:
    base = tuple(float(value) for value in base_pose_mm_deg)
    target = tuple(float(value) for value in target_pose_mm_deg)
    if any(
        len(pose) != 6 or not all(math.isfinite(value) for value in pose)
        for pose in (base, target)
    ):
        raise ValueError("TCP poses must contain six finite values")
    rotation = _rotation_base_from_tcp(base)
    delta = tuple(target[index] - base[index] for index in range(3))
    return sum(rotation[row][2] * delta[row] for row in range(3))


@dataclass(frozen=True)
class VisionTarget:
    tcp_pose_mm_deg: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CalibrationResult:
    correction_x_mm: float
    correction_y_mm: float
    raw_center_x_mm: float
    raw_center_y_mm: float
    sample_count: int
    accepted_sample_count: int
    dispersion_mm: float
    correction_limited: bool


def _rotation_base_from_tcp(
    pose: tuple[float, ...],
) -> tuple[tuple[float, float, float], ...]:
    alpha, beta, gamma = (math.radians(value) for value in pose[3:])
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return (
        (ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb),
        (sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb),
        (-sb * cg, sb * sg, cb),
    )


def predict_vision_target_pose(
    base_pose_mm_deg: Iterable[float],
    current_pose_mm_deg: Iterable[float],
    fist_position_base_mm: Iterable[float],
    fist_velocity_base_mm_s: Iterable[float],
    *,
    maximum_offset_mm: float = 50.0,
    minimum_time_to_plane_ms: float = 40.0,
    maximum_time_to_plane_ms: float = 450.0,
    minimum_normal_speed_mm_s: float = 100.0,
) -> tuple[tuple[float, float, float, float, float, float], float] | None:
    """Return the full bounded Vision target; no LATE/reachability policy is used."""

    base = tuple(float(value) for value in base_pose_mm_deg)
    current = tuple(float(value) for value in current_pose_mm_deg)
    position = tuple(float(value) for value in fist_position_base_mm)
    velocity = tuple(float(value) for value in fist_velocity_base_mm_s)
    if len(base) != 6 or len(current) != 6 or len(position) != 3 or len(velocity) != 3:
        raise ValueError("Vision target inputs have invalid dimensions")
    if not all(math.isfinite(value) for values in (base, current, position, velocity) for value in values):
        raise ValueError("Vision target inputs must be finite")
    limit = float(maximum_offset_mm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum Vision target offset must be positive")

    rotation = _rotation_base_from_tcp(base)
    normal = tuple(rotation[row][2] for row in range(3))
    normal_speed = sum(axis * speed for axis, speed in zip(normal, velocity))
    if abs(normal_speed) < float(minimum_normal_speed_mm_s):
        return None
    distance = sum(
        axis * (plane_axis - fist_axis)
        for axis, plane_axis, fist_axis in zip(normal, current[:3], position)
    )
    time_s = distance / normal_speed
    time_ms = time_s * 1000.0
    if time_s <= 0.0 or not minimum_time_to_plane_ms <= time_ms <= maximum_time_to_plane_ms:
        return None
    intersection = tuple(
        fist_axis + speed * time_s for fist_axis, speed in zip(position, velocity)
    )
    offset_base = tuple(
        target_axis - base_axis for target_axis, base_axis in zip(intersection, base[:3])
    )
    offset_tcp = tuple(
        sum(rotation[row][column] * offset_base[row] for row in range(3))
        for column in range(3)
    )
    bounded_x = max(-limit, min(limit, offset_tcp[0]))
    bounded_y = max(-limit, min(limit, offset_tcp[1]))
    return apply_tool_xy_correction(base, bounded_x, bounded_y), time_ms


def calculate_vision_target_calibration(
    samples: Iterable[VisionTarget],
    base_pose_mm_deg: Iterable[float],
    *,
    required_sample_count: int = 10,
    maximum_correction_mm: float = 50.0,
) -> CalibrationResult:
    base = tuple(float(value) for value in base_pose_mm_deg)
    if len(base) != 6 or not all(math.isfinite(value) for value in base):
        raise ValueError("base TCP pose must contain six finite values")
    rotation = _rotation_base_from_tcp(base)
    points: list[tuple[float, float]] = []
    for sample in samples:
        pose = tuple(float(value) for value in sample.tcp_pose_mm_deg)
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError("Vision target poses must contain six finite values")
        delta_base = tuple(pose[index] - base[index] for index in range(3))
        offset_tcp = tuple(
            sum(rotation[row][column] * delta_base[row] for row in range(3))
            for column in range(3)
        )
        points.append((offset_tcp[0], offset_tcp[1]))
    if required_sample_count <= 0:
        raise ValueError("required sample count must be positive")
    if len(points) < required_sample_count:
        raise ValueError(
            f"at least {required_sample_count} Vision target samples are required"
        )
    if not all(
        math.isfinite(value)
        for point in points
        for value in point
    ):
        raise ValueError("Vision target samples must be finite")
    limit = float(maximum_correction_mm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum correction must be finite and positive")

    center_x = statistics.median(point[0] for point in points)
    center_y = statistics.median(point[1] for point in points)
    deviations = [
        math.hypot(point[0] - center_x, point[1] - center_y)
        for point in points
    ]
    median_deviation = statistics.median(deviations)
    mad = statistics.median(
        abs(deviation - median_deviation) for deviation in deviations
    )
    threshold = median_deviation + 3.0 * 1.4826 * mad
    accepted = [
        point
        for point, deviation in zip(points, deviations)
        if deviation <= threshold + 1e-9
    ]
    if len(accepted) * 2 <= len(points):
        raise ValueError("a majority of Vision target samples must agree")

    raw_x = statistics.median(point[0] for point in accepted)
    raw_y = statistics.median(point[1] for point in accepted)
    correction_x = max(-limit, min(limit, raw_x))
    correction_y = max(-limit, min(limit, raw_y))
    dispersion = math.sqrt(
        sum(
            (point[0] - raw_x) ** 2 + (point[1] - raw_y) ** 2
            for point in accepted
        )
        / len(accepted)
    )
    return CalibrationResult(
        correction_x_mm=correction_x,
        correction_y_mm=correction_y,
        raw_center_x_mm=raw_x,
        raw_center_y_mm=raw_y,
        sample_count=len(points),
        accepted_sample_count=len(accepted),
        dispersion_mm=dispersion,
        correction_limited=(
            not math.isclose(raw_x, correction_x, abs_tol=1e-9)
            or not math.isclose(raw_y, correction_y, abs_tol=1e-9)
        ),
    )

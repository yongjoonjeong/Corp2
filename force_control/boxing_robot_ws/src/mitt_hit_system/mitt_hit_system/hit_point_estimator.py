"""Estimate an impact point from wrench samples expressed in the mitt frame."""

from dataclasses import dataclass
import math
from typing import Iterable

from mitt_hit_system.common.enums import HitDirection


@dataclass(frozen=True)
class ImpactSample:
    """Core wrench components after conversion to the mitt coordinate frame.

    ``normal_force_n`` is the mitt Z-axis force. ``moment_x_nm`` and
    ``moment_y_nm`` are moments about the mitt X and Y axes.
    """

    normal_force_n: float
    moment_x_nm: float
    moment_y_nm: float


@dataclass(frozen=True)
class ImpactPointConfig:
    normal_sign: int = 1
    sign_x: float = 1.0
    sign_y: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    cross_xy: float = 0.0
    cross_yx: float = 0.0
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    minimum_force_for_position_n: float = 10.0
    use_force_weighted_average: bool = True
    perfect_radius_mm: float = 20.0
    valid_radius_mm: float = 100.0
    direction_deadband_mm: float = 10.0

    def __post_init__(self) -> None:
        if self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be -1 or 1")
        values = (
            self.sign_x,
            self.sign_y,
            self.scale_x,
            self.scale_y,
            self.cross_xy,
            self.cross_yx,
            self.offset_x_mm,
            self.offset_y_mm,
            self.minimum_force_for_position_n,
            self.perfect_radius_mm,
            self.valid_radius_mm,
            self.direction_deadband_mm,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("impact point configuration values must be finite")
        if self.minimum_force_for_position_n <= 0.0:
            raise ValueError("minimum_force_for_position_n must be positive")
        if self.perfect_radius_mm < 0.0:
            raise ValueError("perfect_radius_mm must be non-negative")
        if self.valid_radius_mm <= self.perfect_radius_mm:
            raise ValueError("valid_radius_mm must exceed perfect_radius_mm")
        if not 0.0 <= self.direction_deadband_mm <= self.perfect_radius_mm:
            raise ValueError("direction_deadband_mm must be within perfect radius")


@dataclass(frozen=True)
class EstimatedImpactPoint:
    x_mm: float
    y_mm: float
    center_error_mm: float
    direction: HitDirection
    used_sample_count: int


class HitPointEstimator:
    """Apply x=-My/Fz and y=Mx/Fz with configurable calibration."""

    def __init__(self, config: ImpactPointConfig | None = None) -> None:
        self.config = config or ImpactPointConfig()

    def estimate(self, samples: Iterable[ImpactSample]) -> EstimatedImpactPoint:
        weighted_x_sum = 0.0
        weighted_y_sum = 0.0
        weight_sum = 0.0
        used_sample_count = 0

        for sample in samples:
            values = (
                sample.normal_force_n,
                sample.moment_x_nm,
                sample.moment_y_nm,
            )
            if not all(math.isfinite(value) for value in values):
                continue

            normal_force = sample.normal_force_n * self.config.normal_sign
            if abs(normal_force) < self.config.minimum_force_for_position_n:
                continue

            raw_x_mm = (-sample.moment_y_nm / normal_force) * 1000.0
            raw_y_mm = (sample.moment_x_nm / normal_force) * 1000.0
            weight = (
                abs(normal_force)
                if self.config.use_force_weighted_average
                else 1.0
            )
            weighted_x_sum += raw_x_mm * weight
            weighted_y_sum += raw_y_mm * weight
            weight_sum += weight
            used_sample_count += 1

        if used_sample_count == 0 or weight_sum <= 0.0:
            raise ValueError("no samples have sufficient normal force")

        raw_x_mm = weighted_x_sum / weight_sum
        raw_y_mm = weighted_y_sum / weight_sum
        mixed_x_mm = (
            self.config.scale_x * raw_x_mm
            + self.config.cross_xy * raw_y_mm
        )
        mixed_y_mm = (
            self.config.scale_y * raw_y_mm
            + self.config.cross_yx * raw_x_mm
        )
        x_mm = self.config.sign_x * mixed_x_mm + self.config.offset_x_mm
        y_mm = self.config.sign_y * mixed_y_mm + self.config.offset_y_mm
        center_error_mm = math.hypot(x_mm, y_mm)

        return EstimatedImpactPoint(
            x_mm=x_mm,
            y_mm=y_mm,
            center_error_mm=center_error_mm,
            direction=self.classify_direction(x_mm, y_mm),
            used_sample_count=used_sample_count,
        )

    def classify_direction(self, x_mm: float, y_mm: float) -> HitDirection:
        if not math.isfinite(x_mm) or not math.isfinite(y_mm):
            return HitDirection.INVALID

        error = math.hypot(x_mm, y_mm)
        if error <= self.config.perfect_radius_mm:
            return HitDirection.CENTER
        if error >= self.config.valid_radius_mm:
            return HitDirection.MISS

        deadband = self.config.direction_deadband_mm
        horizontal = 0 if abs(x_mm) <= deadband else (1 if x_mm > 0.0 else -1)
        vertical = 0 if abs(y_mm) <= deadband else (1 if y_mm > 0.0 else -1)

        directions = {
            (-1, 0): HitDirection.LEFT,
            (1, 0): HitDirection.RIGHT,
            (0, 1): HitDirection.UP,
            (0, -1): HitDirection.DOWN,
            (-1, 1): HitDirection.UP_LEFT,
            (1, 1): HitDirection.UP_RIGHT,
            (-1, -1): HitDirection.DOWN_LEFT,
            (1, -1): HitDirection.DOWN_RIGHT,
        }
        return directions.get((horizontal, vertical), HitDirection.CENTER)

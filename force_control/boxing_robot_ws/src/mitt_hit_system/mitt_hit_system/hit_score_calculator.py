"""ROS-independent conversion from center error to an accuracy score."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math


@dataclass(frozen=True)
class ScoringConfig:
    perfect_radius_mm: float = 20.0
    valid_radius_mm: float = 100.0
    minimum_score: float = 0.0
    maximum_score: float = 10.0
    round_digits: int = 1

    def __post_init__(self) -> None:
        values = (
            self.perfect_radius_mm,
            self.valid_radius_mm,
            self.minimum_score,
            self.maximum_score,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("scoring values must be finite")
        if self.perfect_radius_mm < 0.0:
            raise ValueError("perfect_radius_mm must be non-negative")
        if self.valid_radius_mm <= self.perfect_radius_mm:
            raise ValueError("valid_radius_mm must exceed perfect_radius_mm")
        if self.maximum_score <= self.minimum_score:
            raise ValueError("maximum_score must exceed minimum_score")
        if self.round_digits < 0:
            raise ValueError("round_digits must be non-negative")


class HitScoreCalculator:
    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

    def calculate(self, center_error_mm: float) -> float:
        """Return the accuracy score for a non-negative center error."""
        if not math.isfinite(center_error_mm) or center_error_mm < 0.0:
            raise ValueError("center_error_mm must be finite and non-negative")

        config = self.config
        if center_error_mm <= config.perfect_radius_mm:
            score = config.maximum_score
        elif center_error_mm >= config.valid_radius_mm:
            score = config.minimum_score
        else:
            usable_width = config.valid_radius_mm - config.perfect_radius_mm
            ratio = (center_error_mm - config.perfect_radius_mm) / usable_width
            score = config.maximum_score - ratio * (
                config.maximum_score - config.minimum_score
            )

        clamped = min(config.maximum_score, max(config.minimum_score, score))
        quantum = Decimal("1").scaleb(-config.round_digits)
        return float(
            Decimal(str(clamped)).quantize(quantum, rounding=ROUND_HALF_UP)
        )

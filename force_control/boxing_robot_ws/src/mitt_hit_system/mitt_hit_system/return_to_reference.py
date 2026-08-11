"""High-rate return-to-reference settling decision without ROS dependencies."""

from dataclasses import dataclass
from enum import Enum
import math


class ReturnStatus(str, Enum):
    WAITING = "WAITING"
    SETTLED = "SETTLED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class ReturnObservation:
    timestamp_ns: int
    tcp_position_mm: tuple[float, float, float]
    tcp_velocity_mm_s: tuple[float, float, float]
    displacement_mm: float
    translation_speed_mm_s: float
    normal_force_n: float
    robot_state: int


@dataclass(frozen=True)
class ReturnToReferenceConfig:
    position_tolerance_mm: float
    velocity_tolerance_mm_s: float
    force_tolerance_n: float
    settle_time_ms: float
    timeout_ms: float

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_mm,
            self.velocity_tolerance_mm_s,
            self.force_tolerance_n,
            self.settle_time_ms,
            self.timeout_ms,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("return-to-reference values must be finite and positive")
        if self.timeout_ms <= self.settle_time_ms:
            raise ValueError("return timeout must exceed settle time")


class ReturnToReferenceMonitor:
    def __init__(self, config: ReturnToReferenceConfig) -> None:
        self.config = config
        self._started_ns: int | None = None
        self._stable_started_ns: int | None = None

    def start(self, timestamp_ns: int) -> None:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        self._started_ns = timestamp_ns
        self._stable_started_ns = None

    def process(
        self,
        timestamp_ns: int,
        *,
        displacement_mm: float,
        translation_speed_mm_s: float,
        normal_force_n: float,
    ) -> ReturnStatus:
        if self._started_ns is None:
            raise RuntimeError("return monitor has not started")
        values = (displacement_mm, translation_speed_mm_s, normal_force_n)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("return observations must be finite and non-negative")
        if timestamp_ns < self._started_ns:
            raise ValueError("return timestamp precedes start")
        elapsed_ms = (timestamp_ns - self._started_ns) / 1e6
        if elapsed_ms > self.config.timeout_ms:
            return ReturnStatus.TIMEOUT

        stable = (
            displacement_mm <= self.config.position_tolerance_mm
            and translation_speed_mm_s <= self.config.velocity_tolerance_mm_s
            and normal_force_n <= self.config.force_tolerance_n
        )
        if not stable:
            self._stable_started_ns = None
            return ReturnStatus.WAITING
        if self._stable_started_ns is None:
            self._stable_started_ns = timestamp_ns
            return ReturnStatus.WAITING
        stable_ms = (timestamp_ns - self._stable_started_ns) / 1e6
        if stable_ms >= self.config.settle_time_ms:
            return ReturnStatus.SETTLED
        return ReturnStatus.WAITING

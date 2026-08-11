"""State machine that segments a normal-force stream into individual hits."""

from dataclasses import dataclass
from enum import Enum
import math


class HitDetectorState(str, Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    RISING = "RISING"
    IMPACT = "IMPACT"
    FALLING = "FALLING"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class HitDetectionConfig:
    start_force_n: float = 10.0
    end_force_n: float = 5.0
    stable_force_n: float = 3.0
    minimum_hit_duration_ms: float = 10.0
    maximum_hit_duration_ms: float = 300.0
    end_stable_time_ms: float = 20.0
    debounce_ms: float = 150.0

    def __post_init__(self) -> None:
        values = (
            self.start_force_n,
            self.end_force_n,
            self.stable_force_n,
            self.minimum_hit_duration_ms,
            self.maximum_hit_duration_ms,
            self.end_stable_time_ms,
            self.debounce_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("hit detection values must be finite")
        if not 0.0 <= self.stable_force_n <= self.end_force_n < self.start_force_n:
            raise ValueError("force thresholds must satisfy stable <= end < start")
        if self.minimum_hit_duration_ms < 0.0:
            raise ValueError("minimum_hit_duration_ms must be non-negative")
        if self.maximum_hit_duration_ms <= self.minimum_hit_duration_ms:
            raise ValueError("maximum_hit_duration_ms must exceed minimum")
        if self.end_stable_time_ms < 0.0 or self.debounce_ms < 0.0:
            raise ValueError("time thresholds must be non-negative")


@dataclass(frozen=True)
class DetectedHit:
    valid: bool
    reason: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    contact_duration_ms: float
    peak_normal_force_n: float


class HitDetector:
    def __init__(self, config: HitDetectionConfig | None = None) -> None:
        self.config = config or HitDetectionConfig()
        self.state = HitDetectorState.IDLE
        self._last_timestamp_ns: int | None = None
        self._start_timestamp_ns = 0
        self._falling_timestamp_ns = 0
        self._complete_timestamp_ns = 0
        self._peak_normal_force_n = 0.0

    def start(self) -> None:
        self._clear_event()
        self._last_timestamp_ns = None
        self.state = HitDetectorState.WAITING

    def stop(self) -> None:
        self._clear_event()
        self._last_timestamp_ns = None
        self.state = HitDetectorState.IDLE

    def process(
        self,
        timestamp_ns: int,
        normal_force_n: float,
        *,
        safety_stop: bool = False,
        system_blocked: bool = False,
    ) -> DetectedHit | None:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError("timestamps must be monotonic")
        if not math.isfinite(normal_force_n):
            raise ValueError("normal_force_n must be finite")
        self._last_timestamp_ns = timestamp_ns
        force = abs(normal_force_n)

        if self.state is HitDetectorState.IDLE:
            return None
        if self.state is HitDetectorState.COMPLETE:
            debounce_ns = self._milliseconds_to_ns(self.config.debounce_ms)
            if (
                timestamp_ns - self._complete_timestamp_ns >= debounce_ns
                and force <= self.config.stable_force_n
            ):
                self._clear_event()
                self.state = HitDetectorState.WAITING
            return None

        active = self.state in {
            HitDetectorState.RISING,
            HitDetectorState.IMPACT,
            HitDetectorState.FALLING,
        }
        if active and safety_stop:
            return self._finish(timestamp_ns, timestamp_ns, False, "SAFETY_STOP")
        if active and system_blocked:
            return self._finish(timestamp_ns, timestamp_ns, False, "SYSTEM_BLOCKED")
        if safety_stop or system_blocked:
            return None

        if self.state is HitDetectorState.WAITING:
            if force >= self.config.start_force_n:
                self._start_timestamp_ns = timestamp_ns
                self._peak_normal_force_n = force
                self.state = HitDetectorState.RISING
            return None

        self._peak_normal_force_n = max(self._peak_normal_force_n, force)
        elapsed_ms = self._nanoseconds_to_ms(timestamp_ns - self._start_timestamp_ns)
        if elapsed_ms > self.config.maximum_hit_duration_ms:
            return self._finish(timestamp_ns, timestamp_ns, False, "CONTACT_TOO_LONG")

        if self.state is HitDetectorState.RISING:
            if force <= self.config.end_force_n:
                self._falling_timestamp_ns = timestamp_ns
                self.state = HitDetectorState.FALLING
            elif elapsed_ms >= self.config.minimum_hit_duration_ms:
                self.state = HitDetectorState.IMPACT
            return None

        if self.state is HitDetectorState.IMPACT:
            if force <= self.config.end_force_n:
                self._falling_timestamp_ns = timestamp_ns
                self.state = HitDetectorState.FALLING
            return None

        if self.state is HitDetectorState.FALLING:
            if force > self.config.end_force_n:
                self.state = HitDetectorState.IMPACT
                return None
            if force > self.config.stable_force_n:
                self._falling_timestamp_ns = timestamp_ns
                return None

            stable_ms = self._nanoseconds_to_ms(
                timestamp_ns - self._falling_timestamp_ns
            )
            if stable_ms < self.config.end_stable_time_ms:
                return None

            contact_ms = self._nanoseconds_to_ms(
                self._falling_timestamp_ns - self._start_timestamp_ns
            )
            valid = contact_ms >= self.config.minimum_hit_duration_ms
            reason = "" if valid else "CONTACT_TOO_SHORT"
            return self._finish(
                timestamp_ns,
                self._falling_timestamp_ns,
                valid,
                reason,
            )
        return None

    def _finish(
        self,
        complete_timestamp_ns: int,
        contact_end_timestamp_ns: int,
        valid: bool,
        reason: str,
    ) -> DetectedHit:
        contact_duration_ms = self._nanoseconds_to_ms(
            contact_end_timestamp_ns - self._start_timestamp_ns
        )
        event = DetectedHit(
            valid=valid,
            reason=reason,
            start_timestamp_ns=self._start_timestamp_ns,
            end_timestamp_ns=contact_end_timestamp_ns,
            contact_duration_ms=contact_duration_ms,
            peak_normal_force_n=self._peak_normal_force_n,
        )
        self._complete_timestamp_ns = complete_timestamp_ns
        self.state = HitDetectorState.COMPLETE
        return event

    def _clear_event(self) -> None:
        self._start_timestamp_ns = 0
        self._falling_timestamp_ns = 0
        self._complete_timestamp_ns = 0
        self._peak_normal_force_n = 0.0

    @staticmethod
    def _milliseconds_to_ns(milliseconds: float) -> int:
        return int(milliseconds * 1_000_000.0)

    @staticmethod
    def _nanoseconds_to_ms(nanoseconds: int) -> float:
        return nanoseconds / 1_000_000.0

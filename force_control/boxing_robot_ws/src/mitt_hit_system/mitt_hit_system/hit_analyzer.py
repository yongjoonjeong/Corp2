"""ROS-independent preview analysis for short contacts on the real mitt."""

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

from mitt_hit_system.common.enums import HitDirection
from mitt_hit_system.force_calibrator import (
    ForceCalibrationError,
    ForceCalibrator,
    WrenchZeroCalibration,
)
from mitt_hit_system.hit_detector import (
    DetectedHit,
    HitDetectionConfig,
    HitDetector,
    HitDetectorState,
)
from mitt_hit_system.hit_point_estimator import (
    HitPointEstimator,
    ImpactPointConfig,
    ImpactSample,
)


Wrench6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class HitAnalyzerConfig:
    calibration_duration_ms: float = 3000.0
    minimum_calibration_samples: int = 50
    maximum_zero_force_stddev_n: float = 0.5
    maximum_zero_moment_stddev_nm: float = 0.05
    normal_sign: int = -1
    sign_x: float = -1.0
    sign_y: float = -1.0
    start_force_n: float = 10.0
    end_force_n: float = 5.0
    stable_force_n: float = 3.0
    minimum_hit_duration_ms: float = 10.0
    maximum_hit_duration_ms: float = 300.0
    end_stable_time_ms: float = 20.0
    debounce_ms: float = 150.0
    minimum_position_force_n: float = 8.0
    minimum_position_samples: int = 2
    perfect_radius_mm: float = 20.0
    direction_deadband_mm: float = 10.0
    mitt_width_mm: float = 190.0
    mitt_height_mm: float = 150.0
    preview_force_warning_n: float = 30.0

    def __post_init__(self) -> None:
        numeric = (
            self.calibration_duration_ms,
            self.maximum_zero_force_stddev_n,
            self.maximum_zero_moment_stddev_nm,
            self.sign_x,
            self.sign_y,
            self.minimum_position_force_n,
            self.perfect_radius_mm,
            self.direction_deadband_mm,
            self.mitt_width_mm,
            self.mitt_height_mm,
            self.preview_force_warning_n,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("hit analyzer configuration values must be finite")
        if self.calibration_duration_ms <= 0.0:
            raise ValueError("calibration_duration_ms must be positive")
        if self.minimum_calibration_samples < 10:
            raise ValueError("minimum_calibration_samples must be at least 10")
        if self.minimum_position_samples <= 0:
            raise ValueError("minimum_position_samples must be positive")
        if self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be -1 or 1")
        if self.minimum_position_force_n <= 0.0:
            raise ValueError("minimum_position_force_n must be positive")
        if self.mitt_width_mm <= 0.0 or self.mitt_height_mm <= 0.0:
            raise ValueError("mitt dimensions must be positive")


@dataclass(frozen=True)
class HitAnalyzerResult:
    hit_id: int
    valid: bool
    reason: str
    direction: HitDirection
    x_mm: float
    y_mm: float
    center_error_mm: float
    peak_force_n: float
    peak_normal_force_n: float
    impulse_ns: float
    contact_duration_ms: float
    force_warning: bool
    sample_count: int
    contact_samples: tuple[tuple[int, Wrench6], ...]


class HitAnalyzerProcessor:
    """Segment short contacts and calculate preview-only hit metrics."""

    def __init__(self, config: HitAnalyzerConfig | None = None) -> None:
        self.config = config or HitAnalyzerConfig()
        self._calibrator = ForceCalibrator(
            maximum_force_stddev_n=self.config.maximum_zero_force_stddev_n,
            maximum_moment_stddev_nm=self.config.maximum_zero_moment_stddev_nm,
            minimum_samples=self.config.minimum_calibration_samples,
        )
        self._detector = HitDetector(
            HitDetectionConfig(
                start_force_n=self.config.start_force_n,
                end_force_n=self.config.end_force_n,
                stable_force_n=self.config.stable_force_n,
                minimum_hit_duration_ms=self.config.minimum_hit_duration_ms,
                maximum_hit_duration_ms=self.config.maximum_hit_duration_ms,
                end_stable_time_ms=self.config.end_stable_time_ms,
                debounce_ms=self.config.debounce_ms,
            )
        )
        self._estimator = HitPointEstimator(
            ImpactPointConfig(
                normal_sign=self.config.normal_sign,
                sign_x=self.config.sign_x,
                sign_y=self.config.sign_y,
                minimum_force_for_position_n=self.config.minimum_position_force_n,
                perfect_radius_mm=self.config.perfect_radius_mm,
                valid_radius_mm=math.hypot(
                    self.config.mitt_width_mm / 2.0,
                    self.config.mitt_height_mm / 2.0,
                ) + 1.0,
                direction_deadband_mm=self.config.direction_deadband_mm,
            )
        )
        self._zero_samples: deque[Wrench6] = deque(maxlen=2000)
        self._zero_started_ns: int | None = None
        self._zero: WrenchZeroCalibration | None = None
        self._capture: list[tuple[int, Wrench6]] = []
        self._next_hit_id = 1

    @property
    def calibrated(self) -> bool:
        return self._zero is not None

    @property
    def calibration_sample_count(self) -> int:
        return len(self._zero_samples)

    @property
    def zero_calibration(self) -> WrenchZeroCalibration | None:
        return self._zero

    def begin_zero_recalibration(self) -> None:
        """Forget the old bias and collect a fresh stable no-contact baseline."""
        self._zero = None
        self._zero_samples.clear()
        self._zero_started_ns = None
        self._capture = []
        self._detector.start()

    def start_test(self) -> None:
        """Reset event state and visible numbering for a new test session."""
        self._next_hit_id = 1
        self._capture = []
        self._detector.start()

    def correct_wrench(self, wrench: Sequence[float]) -> Wrench6:
        if self._zero is None:
            raise RuntimeError("wrench zero calibration is not complete")
        return ForceCalibrator.apply(self._normalize_wrench(wrench), self._zero)

    def compressive_normal_force(self, wrench: Sequence[float]) -> float:
        corrected = self.correct_wrench(wrench)
        return max(corrected[2] * self.config.normal_sign, 0.0)

    def process(
        self, timestamp_ns: int, wrench: Sequence[float]
    ) -> HitAnalyzerResult | None:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        values = self._normalize_wrench(wrench)
        if self._zero is None:
            self._process_zero(timestamp_ns, values)
            return None

        corrected = self.correct_wrench(values)
        normal_force = corrected[2] * self.config.normal_sign
        # Only compression into the mitt is a contact. Rebound/pull force must
        # not reach HitDetector, which intentionally uses an absolute value.
        compressive_force = max(normal_force, 0.0)
        state_before = self._detector.state
        event = self._detector.process(timestamp_ns, compressive_force)
        state_after = self._detector.state

        if (
            state_before is HitDetectorState.WAITING
            and state_after is HitDetectorState.RISING
        ):
            self._capture = [(timestamp_ns, corrected)]
        elif self._capture:
            self._capture.append((timestamp_ns, corrected))

        if event is None:
            return None

        capture = tuple(self._capture)
        self._capture = []
        result = self._analyze_event(event, capture)
        self._next_hit_id += 1
        return result

    def _process_zero(self, timestamp_ns: int, wrench: Wrench6) -> None:
        if self._zero_started_ns is None:
            self._zero_started_ns = timestamp_ns
        self._zero_samples.append(wrench)
        elapsed_ms = (timestamp_ns - self._zero_started_ns) / 1_000_000.0
        if (
            elapsed_ms < self.config.calibration_duration_ms
            or len(self._zero_samples) < self.config.minimum_calibration_samples
        ):
            return
        try:
            self._zero = self._calibrator.calculate(
                self._zero_samples, pose_name="HIT_ANALYZER_IDLE"
            )
        except ForceCalibrationError:
            # Start a fresh window so a touched/unstable startup cannot contaminate zero.
            self._zero_samples.clear()
            self._zero_samples.append(wrench)
            self._zero_started_ns = timestamp_ns
            return
        self._detector.start()

    def _analyze_event(
        self, event: DetectedHit, capture: tuple[tuple[int, Wrench6], ...]
    ) -> HitAnalyzerResult:
        contact = tuple(
            sample
            for sample in capture
            if event.start_timestamp_ns <= sample[0] <= event.end_timestamp_ns
        )
        if not contact:
            contact = capture
        wrenches = [sample[1] for sample in contact]
        normal_forces = [
            max(wrench[2] * self.config.normal_sign, 0.0) for wrench in wrenches
        ]
        peak_normal = max(
            event.peak_normal_force_n,
            max(normal_forces, default=0.0),
        )
        peak_force = max(
            (math.sqrt(sum(value * value for value in wrench[:3])) for wrench in wrenches),
            default=peak_normal,
        )
        impulse = self._integrate_impulse(contact)
        valid = bool(event.valid)
        reason = str(event.reason)
        direction = HitDirection.INVALID
        x_mm = y_mm = center_error_mm = 0.0

        position_samples = [
            ImpactSample(wrench[2], wrench[3], wrench[4])
            for wrench in wrenches
            if wrench[2] * self.config.normal_sign
            >= self.config.minimum_position_force_n
        ]
        if valid and len(position_samples) < self.config.minimum_position_samples:
            valid = False
            reason = "INSUFFICIENT_POSITION_SAMPLES"
        if valid:
            try:
                estimated = self._estimator.estimate(position_samples)
            except ValueError:
                valid = False
                reason = "POSITION_ESTIMATION_FAILED"
            else:
                x_mm = estimated.x_mm
                y_mm = estimated.y_mm
                center_error_mm = estimated.center_error_mm
                within_mitt = (
                    abs(x_mm) <= self.config.mitt_width_mm / 2.0
                    and abs(y_mm) <= self.config.mitt_height_mm / 2.0
                )
                if within_mitt:
                    direction = estimated.direction
                else:
                    valid = False
                    reason = "OUTSIDE_MITT"
                    direction = HitDirection.MISS

        return HitAnalyzerResult(
            hit_id=self._next_hit_id,
            valid=valid,
            reason=reason,
            direction=direction,
            x_mm=x_mm,
            y_mm=y_mm,
            center_error_mm=center_error_mm,
            peak_force_n=peak_force,
            peak_normal_force_n=peak_normal,
            impulse_ns=impulse,
            contact_duration_ms=event.contact_duration_ms,
            force_warning=peak_normal >= self.config.preview_force_warning_n,
            sample_count=len(contact),
            contact_samples=contact,
        )

    def _integrate_impulse(
        self, samples: tuple[tuple[int, Wrench6], ...]
    ) -> float:
        impulse = 0.0
        for (time_a, wrench_a), (time_b, wrench_b) in zip(samples, samples[1:]):
            force_a = max(wrench_a[2] * self.config.normal_sign, 0.0)
            force_b = max(wrench_b[2] * self.config.normal_sign, 0.0)
            impulse += 0.5 * (force_a + force_b) * (time_b - time_a) / 1e9
        return impulse

    @staticmethod
    def _normalize_wrench(wrench: Sequence[float]) -> Wrench6:
        if len(wrench) != 6:
            raise ValueError("wrench must contain six values")
        values = tuple(float(value) for value in wrench)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wrench values must be finite")
        return values  # type: ignore[return-value]

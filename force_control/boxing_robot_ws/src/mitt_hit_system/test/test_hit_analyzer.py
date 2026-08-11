import pytest

from mitt_hit_system.common.enums import HitDirection
from mitt_hit_system.hit_analyzer import (
    HitAnalyzerConfig,
    HitAnalyzerProcessor,
)


MS = 1_000_000
BASELINE = (-4.0, -1.0, -0.2, -0.2, 0.8, 0.1)


def add_delta(delta: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(value + change for value, change in zip(BASELINE, delta))


def make_processor() -> HitAnalyzerProcessor:
    return HitAnalyzerProcessor(
        HitAnalyzerConfig(
            calibration_duration_ms=90.0,
            minimum_calibration_samples=10,
            minimum_position_samples=2,
        )
    )


def calibrate(processor: HitAnalyzerProcessor) -> None:
    for index in range(10):
        assert processor.process(index * 10 * MS, BASELINE) is None
    assert processor.calibrated


def run_valid_event(
    processor: HitAnalyzerProcessor,
    moment_x_nm: float = 0.0,
    moment_y_nm: float = 0.0,
):
    sequence = (
        (200, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (210, (0.0, 0.0, -12.0, moment_x_nm * 0.6, moment_y_nm * 0.6, 0.0)),
        (220, (0.0, 0.0, -20.0, moment_x_nm, moment_y_nm, 0.0)),
        (230, (0.0, 0.0, -16.0, moment_x_nm * 0.8, moment_y_nm * 0.8, 0.0)),
        (240, (0.0, 0.0, -2.0, 0.0, 0.0, 0.0)),
        (260, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    result = None
    for timestamp_ms, delta in sequence:
        result = processor.process(timestamp_ms * MS, add_delta(delta)) or result
    return result


def test_short_center_event_reports_force_duration_and_impulse() -> None:
    processor = make_processor()
    calibrate(processor)

    result = run_valid_event(processor)

    assert result is not None
    assert result.valid
    assert result.direction is HitDirection.CENTER
    assert result.peak_normal_force_n == pytest.approx(20.0)
    assert result.contact_duration_ms == pytest.approx(30.0)
    assert result.impulse_ns > 0.0
    assert result.sample_count == 4
    assert len(result.contact_samples) == 4


def test_front_facing_right_event_uses_verified_axis_signs() -> None:
    processor = make_processor()
    calibrate(processor)

    result = run_valid_event(processor, moment_y_nm=1.0)

    assert result is not None and result.valid
    assert result.direction is HitDirection.RIGHT
    assert result.x_mm == pytest.approx(50.0)
    assert result.y_mm == pytest.approx(0.0)


def test_long_static_push_is_not_a_valid_hit() -> None:
    processor = make_processor()
    calibrate(processor)

    processor.process(200 * MS, add_delta((0, 0, -15, 0, 0, 0)))
    result = processor.process(501 * MS, add_delta((0, 0, -15, 0, 0, 0)))

    assert result is not None
    assert not result.valid
    assert result.reason == "CONTACT_TOO_LONG"


def test_contact_shorter_than_minimum_is_rejected() -> None:
    processor = make_processor()
    calibrate(processor)

    processor.process(200 * MS, add_delta((0, 0, -15, 0, 0, 0)))
    processor.process(205 * MS, add_delta((0, 0, 0, 0, 0, 0)))
    result = processor.process(225 * MS, add_delta((0, 0, 0, 0, 0, 0)))

    assert result is not None
    assert not result.valid
    assert result.reason == "CONTACT_TOO_SHORT"


def test_preview_warning_does_not_mark_safety_stop() -> None:
    processor = make_processor()
    calibrate(processor)

    sequence = (
        (200, (0, 0, -31, 0, 0, 0)),
        (220, (0, 0, -35, 0, 0, 0)),
        (240, (0, 0, 0, 0, 0, 0)),
        (260, (0, 0, 0, 0, 0, 0)),
    )
    result = None
    for timestamp_ms, delta in sequence:
        result = processor.process(timestamp_ms * MS, add_delta(delta)) or result

    assert result is not None
    assert result.force_warning


def test_rebound_force_in_opposite_direction_does_not_create_event() -> None:
    processor = make_processor()
    calibrate(processor)

    result = None
    for timestamp_ms in range(200, 700, 20):
        # Positive TOOL Fz is away from the mitt, not compressive contact.
        result = processor.process(
            timestamp_ms * MS, add_delta((0, 0, 20, 0, 0, 0))
        ) or result

    assert result is None


def test_return_force_uses_same_zero_calibration_as_hit_detection() -> None:
    processor = make_processor()
    calibrate(processor)

    assert processor.compressive_normal_force(BASELINE) == pytest.approx(0.0)
    assert processor.compressive_normal_force(
        add_delta((0, 0, -12, 0, 0, 0))
    ) == pytest.approx(12.0)


def test_post_activation_recalibration_replaces_residual_wrench_baseline() -> None:
    processor = make_processor()
    calibrate(processor)
    new_baseline = (2.0, -3.0, 1.2, 0.4, -0.3, 0.2)

    processor.begin_zero_recalibration()

    assert not processor.calibrated
    assert processor.zero_calibration is None
    for index in range(10):
        assert processor.process((300 + index * 10) * MS, new_baseline) is None

    assert processor.calibrated
    assert processor.zero_calibration is not None
    assert processor.zero_calibration.offset == pytest.approx(new_baseline)
    assert processor.compressive_normal_force(new_baseline) == pytest.approx(0.0)
    assert processor.compressive_normal_force(
        tuple(value + delta for value, delta in zip(new_baseline, (0, 0, -12, 0, 0, 0)))
    ) == pytest.approx(12.0)


def test_low_force_commissioning_threshold_detects_a_gentle_tap() -> None:
    processor = HitAnalyzerProcessor(
        HitAnalyzerConfig(
            calibration_duration_ms=90.0,
            minimum_calibration_samples=10,
            start_force_n=2.0,
            end_force_n=1.0,
            stable_force_n=0.5,
            minimum_position_force_n=1.5,
            minimum_position_samples=2,
        )
    )
    calibrate(processor)
    sequence = (
        (210, (0.0, 0.0, -2.2, 0.0, 0.0, 0.0)),
        (220, (0.0, 0.0, -3.0, 0.0, 0.0, 0.0)),
        (230, (0.0, 0.0, -2.0, 0.0, 0.0, 0.0)),
        (240, (0.0, 0.0, -0.4, 0.0, 0.0, 0.0)),
        (260, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )

    result = None
    for timestamp_ms, delta in sequence:
        result = processor.process(timestamp_ms * MS, add_delta(delta)) or result

    assert result is not None and result.valid
    assert result.direction is HitDirection.CENTER
    assert result.peak_normal_force_n == pytest.approx(3.0)

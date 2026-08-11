import pytest

from mitt_hit_system.common.enums import HitDirection
from mitt_hit_system.hit_point_estimator import (
    HitPointEstimator,
    ImpactPointConfig,
    ImpactSample,
)


def test_center_hit() -> None:
    result = HitPointEstimator().estimate([ImpactSample(100.0, 0.0, 0.0)])

    assert result.x_mm == pytest.approx(0.0)
    assert result.y_mm == pytest.approx(0.0)
    assert result.direction is HitDirection.CENTER


def test_right_50_mm_from_force_and_moment() -> None:
    result = HitPointEstimator().estimate([ImpactSample(100.0, 0.0, -5.0)])

    assert result.x_mm == pytest.approx(50.0)
    assert result.y_mm == pytest.approx(0.0)
    assert result.center_error_mm == pytest.approx(50.0)
    assert result.direction is HitDirection.RIGHT


def test_up_50_mm_from_force_and_moment() -> None:
    result = HitPointEstimator().estimate([ImpactSample(100.0, 5.0, 0.0)])

    assert result.x_mm == pytest.approx(0.0)
    assert result.y_mm == pytest.approx(50.0)
    assert result.direction is HitDirection.UP


def test_force_weighted_average() -> None:
    samples = [
        ImpactSample(50.0, 0.0, -1.0),   # x=20 mm, weight=50
        ImpactSample(150.0, 0.0, -9.0),  # x=60 mm, weight=150
    ]
    result = HitPointEstimator().estimate(samples)

    assert result.x_mm == pytest.approx(50.0)
    assert result.used_sample_count == 2


def test_sign_calibration_and_diagonal_direction() -> None:
    estimator = HitPointEstimator(ImpactPointConfig(sign_x=-1.0))
    result = estimator.estimate([ImpactSample(100.0, 4.0, -4.0)])

    assert result.x_mm == pytest.approx(-40.0)
    assert result.y_mm == pytest.approx(40.0)
    assert result.direction is HitDirection.UP_LEFT


def test_outside_valid_radius_is_miss() -> None:
    result = HitPointEstimator().estimate([ImpactSample(100.0, 0.0, -12.0)])

    assert result.center_error_mm == pytest.approx(120.0)
    assert result.direction is HitDirection.MISS


def test_samples_below_position_force_are_rejected() -> None:
    with pytest.raises(ValueError):
        HitPointEstimator().estimate([ImpactSample(5.0, 1.0, 1.0)])

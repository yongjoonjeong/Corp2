import pytest

from mitt_hit_system.hit_score_calculator import HitScoreCalculator, ScoringConfig


def test_score_boundaries_and_midpoint() -> None:
    calculator = HitScoreCalculator()

    assert calculator.calculate(0.0) == 10.0
    assert calculator.calculate(20.0) == 10.0
    assert calculator.calculate(60.0) == 5.0
    assert calculator.calculate(100.0) == 0.0
    assert calculator.calculate(120.0) == 0.0


def test_half_scores_use_conventional_rounding() -> None:
    calculator = HitScoreCalculator()

    assert calculator.calculate(50.0) == 6.3
    assert calculator.calculate(90.0) == 1.3


def test_negative_error_is_rejected() -> None:
    with pytest.raises(ValueError):
        HitScoreCalculator().calculate(-0.1)


def test_invalid_scoring_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ScoringConfig(perfect_radius_mm=100.0, valid_radius_mm=100.0)

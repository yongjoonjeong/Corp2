from pathlib import Path

import pytest

from mitt_hit_system.force_calibrator import (
    ForceCalibrationError,
    ForceCalibrator,
)


def test_calibration_mean_and_offset_application() -> None:
    calibrator = ForceCalibrator(minimum_samples=3)
    samples = [
        (1.0, -2.0, 4.0, 0.01, -0.02, 0.03),
        (1.2, -2.2, 4.2, 0.02, -0.01, 0.02),
        (0.8, -1.8, 3.8, 0.00, -0.03, 0.04),
    ]

    calibration = calibrator.calculate(
        samples, pose_name="CENTER", created_at="2026-08-04T00:00:00+09:00"
    )
    corrected = calibrator.apply(calibration.offset, calibration)

    assert calibration.offset == pytest.approx((1.0, -2.0, 4.0, 0.01, -0.02, 0.03))
    assert corrected == pytest.approx((0.0,) * 6)
    assert calibration.sample_count == 3


def test_unstable_force_samples_are_rejected() -> None:
    calibrator = ForceCalibrator(
        minimum_samples=3,
        maximum_force_stddev_n=0.1,
    )

    with pytest.raises(
        ForceCalibrationError,
        match=r"force samples are not stable: stddev \[Fx=.*limit=0.100000 N",
    ):
        calibrator.calculate([(0.0,) * 6, (1.0, 0, 0, 0, 0, 0), (-1.0, 0, 0, 0, 0, 0)])


def test_unstable_moment_error_reports_each_axis_stddev() -> None:
    calibrator = ForceCalibrator(
        minimum_samples=3,
        maximum_moment_stddev_nm=0.01,
    )

    with pytest.raises(
        ForceCalibrationError,
        match=r"moment samples are not stable: stddev \[Mx=.*My=.*Mz=.*limit=0.010000 Nm",
    ):
        calibrator.calculate(
            [(0.0,) * 6, (0, 0, 0, 0.1, 0, 0), (0, 0, 0, -0.1, 0, 0)]
        )


def test_calibration_yaml_round_trip(tmp_path: Path) -> None:
    calibrator = ForceCalibrator(minimum_samples=3)
    calibration = calibrator.calculate([(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)] * 3)
    path = tmp_path / "calibration" / "wrench_zero.yaml"

    calibrator.save(calibration, path)
    loaded = calibrator.load(path)

    assert loaded == calibration
    assert path.is_file()

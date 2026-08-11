import pytest

from mitt_hit_system.return_to_reference import (
    ReturnStatus,
    ReturnToReferenceConfig,
    ReturnToReferenceMonitor,
)


MS = 1_000_000


def monitor():
    return ReturnToReferenceMonitor(
        ReturnToReferenceConfig(
            position_tolerance_mm=1.0,
            velocity_tolerance_mm_s=2.0,
            force_tolerance_n=3.0,
            settle_time_ms=100.0,
            timeout_ms=1000.0,
        )
    )


def test_return_requires_position_velocity_and_force_to_stay_stable():
    value = monitor()
    value.start(0)

    assert value.process(10 * MS, displacement_mm=0.5, translation_speed_mm_s=3.0, normal_force_n=0.0) is ReturnStatus.WAITING
    assert value.process(20 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=2.0) is ReturnStatus.WAITING
    assert value.process(119 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=2.0) is ReturnStatus.WAITING
    assert value.process(120 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=2.0) is ReturnStatus.SETTLED


def test_unstable_sample_resets_settle_window():
    value = monitor()
    value.start(0)
    value.process(10 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=1.0)
    value.process(80 * MS, displacement_mm=2.0, translation_speed_mm_s=1.0, normal_force_n=1.0)
    value.process(90 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=1.0)

    assert value.process(150 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=1.0) is ReturnStatus.WAITING
    assert value.process(190 * MS, displacement_mm=0.5, translation_speed_mm_s=1.0, normal_force_n=1.0) is ReturnStatus.SETTLED


def test_return_timeout_is_reported():
    value = monitor()
    value.start(0)
    assert value.process(1001 * MS, displacement_mm=5.0, translation_speed_mm_s=0.0, normal_force_n=0.0) is ReturnStatus.TIMEOUT


def test_invalid_return_config_is_rejected():
    with pytest.raises(ValueError):
        ReturnToReferenceConfig(0.0, 1.0, 1.0, 100.0, 1000.0)

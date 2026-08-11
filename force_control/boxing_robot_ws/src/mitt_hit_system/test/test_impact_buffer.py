import pytest

from mitt_hit_system.impact_buffer import BufferedWrenchSample, ImpactBuffer


def sample(index: int) -> BufferedWrenchSample:
    wrench = (0.0, 0.0, float(index), 0.0, 0.0, 0.0)
    return BufferedWrenchSample(index * 10_000_000, wrench, wrench)


def test_pre_contact_and_post_contact_samples_are_captured() -> None:
    buffer = ImpactBuffer(
        sample_period_ms=10.0,
        pre_buffer_ms=20.0,
        post_buffer_ms=20.0,
    )
    for index in range(3):
        assert buffer.append(sample(index)) is None

    buffer.start_capture(7)
    buffer.append(sample(3))
    assert buffer.mark_contact_complete() is None
    assert buffer.append(sample(4)) is None
    captured = buffer.append(sample(5))

    assert captured is not None
    assert captured.hit_id == 7
    assert [item.raw_wrench[2] for item in captured.samples] == [0, 1, 2, 3, 4, 5]
    assert not buffer.capturing


def test_second_capture_cannot_start_during_active_capture() -> None:
    buffer = ImpactBuffer(sample_period_ms=10.0)
    buffer.append(sample(0))
    buffer.start_capture(1)

    with pytest.raises(RuntimeError):
        buffer.start_capture(2)


def test_buffer_size_configuration_is_bounded() -> None:
    with pytest.raises(ValueError):
        ImpactBuffer(
            sample_period_ms=1.0,
            pre_buffer_ms=100.0,
            post_buffer_ms=100.0,
            maximum_samples=100,
        )

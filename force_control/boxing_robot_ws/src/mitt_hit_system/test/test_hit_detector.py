from mitt_hit_system.hit_detector import (
    HitDetectionConfig,
    HitDetector,
    HitDetectorState,
)


MS = 1_000_000


def test_force_below_start_threshold_does_not_create_hit() -> None:
    detector = HitDetector()
    detector.start()

    for timestamp_ms, force_n in [(0, 0.0), (10, 9.9), (20, 5.0), (30, 0.0)]:
        assert detector.process(timestamp_ms * MS, force_n) is None

    assert detector.state is HitDetectorState.WAITING


def test_one_force_pulse_creates_one_valid_hit() -> None:
    detector = HitDetector()
    detector.start()
    event = None

    sequence = [(0, 0.0), (10, 12.0), (20, 40.0), (30, 20.0),
                (40, 4.0), (50, 2.0), (60, 0.0)]
    for timestamp_ms, force_n in sequence:
        event = detector.process(timestamp_ms * MS, force_n) or event

    assert event is not None
    assert event.valid
    assert event.reason == ""
    assert event.contact_duration_ms == 30.0
    assert event.peak_normal_force_n == 40.0
    assert detector.state is HitDetectorState.COMPLETE


def test_short_contact_is_invalid() -> None:
    detector = HitDetector()
    detector.start()

    detector.process(0, 15.0)
    detector.process(5 * MS, 0.0)
    event = detector.process(25 * MS, 0.0)

    assert event is not None
    assert not event.valid
    assert event.reason == "CONTACT_TOO_SHORT"


def test_long_push_is_invalid() -> None:
    detector = HitDetector()
    detector.start()

    detector.process(0, 15.0)
    event = detector.process(301 * MS, 15.0)

    assert event is not None
    assert not event.valid
    assert event.reason == "CONTACT_TOO_LONG"


def test_debounce_prevents_duplicate_hit() -> None:
    detector = HitDetector(HitDetectionConfig(debounce_ms=150.0))
    detector.start()

    sequence = [(0, 12.0), (10, 30.0), (20, 0.0), (40, 0.0)]
    events = [
        event
        for timestamp_ms, force_n in sequence
        if (event := detector.process(timestamp_ms * MS, force_n)) is not None
    ]
    assert len(events) == 1

    assert detector.process(50 * MS, 30.0) is None
    assert detector.state is HitDetectorState.COMPLETE
    assert detector.process(190 * MS, 0.0) is None
    assert detector.state is HitDetectorState.WAITING


def test_safety_stop_invalidates_active_hit() -> None:
    detector = HitDetector()
    detector.start()
    detector.process(0, 15.0)

    event = detector.process(10 * MS, 50.0, safety_stop=True)

    assert event is not None
    assert not event.valid
    assert event.reason == "SAFETY_STOP"

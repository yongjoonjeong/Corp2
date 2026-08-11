from collections import deque
import threading
import time
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from boxing_interfaces.msg import RtMittSample
from mitt_hit_system.common.enums import HitDirection, SystemState
from mitt_hit_system.hit_analyzer import HitAnalyzerResult
from mitt_hit_system.hit_analyzer_node import HitAnalyzerNode
from mitt_hit_system.rebound_motion import ReboundPhase
from mitt_hit_system.return_to_reference import ReturnObservation


class Publisher:
    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message


def test_current_pose_rezero_service_restarts_idle_zero_collection():
    calls = []
    node = SimpleNamespace(
        _intentional_motion_lock=threading.Lock(),
        _intentional_motion_active=False,
        _post_move_zero_active=False,
        _manager=SimpleNamespace(
            begin_calibration=lambda: SimpleNamespace(
                success=True, message="recalibrating wrench zero at current pose"
            )
        ),
        _processor=SimpleNamespace(
            begin_zero_recalibration=lambda: calls.append("rezero")
        ),
        _ready_logged=True,
        _robot_ready=True,
        _publish_state=lambda: calls.append("publish"),
    )
    request = SimpleNamespace(data=True)
    response = SimpleNamespace(success=False, message="")

    result = HitAnalyzerNode._on_recalibrate_zero(node, request, response)

    assert result.success
    assert calls == ["rezero", "publish"]
    assert not node._ready_logged
    assert not node._robot_ready


def test_published_power_score_remains_zero():
    publisher = Publisher()
    node = SimpleNamespace(
        _publisher=publisher,
        _score_calculator=SimpleNamespace(calculate=lambda error: 8.0),
    )
    source = SimpleNamespace(stamp=Time())
    result = HitAnalyzerResult(
        hit_id=1,
        valid=True,
        reason="VALID",
        direction=HitDirection.CENTER,
        x_mm=0.0,
        y_mm=0.0,
        center_error_mm=0.0,
        peak_force_n=20.0,
        peak_normal_force_n=20.0,
        impulse_ns=0.2,
        contact_duration_ms=20.0,
        force_warning=False,
        sample_count=2,
        contact_samples=(),
    )

    message = HitAnalyzerNode._publish(node, source, result)

    assert message.power_score == 0.0
    assert message.total_score == message.accuracy_score == 8.0


def test_return_record_contains_tuning_extrema_and_clears_buffer():
    calls = []
    logger = SimpleNamespace(
        log_return=lambda hit_id, result, samples: (
            calls.append((hit_id, result, tuple(samples))) or (True, "")
        )
    )
    samples = deque(
        [
            ReturnObservation(
                100,
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 0.0),
                0.3,
                1.0,
                2.0,
                1,
            ),
            ReturnObservation(
                1_000_100,
                (1.1, 2.0, 3.0),
                (0.1, 0.0, 0.0),
                0.5,
                3.0,
                4.0,
                1,
            ),
        ]
    )
    node = SimpleNamespace(
        _return_hit_id=2,
        _return_samples=samples,
        _record_logger=logger,
    )

    HitAnalyzerNode._record_return(node, "SETTLED", "stable")

    assert calls[0][0] == 2
    summary = calls[0][1]
    assert summary["outcome"] == "SETTLED"
    assert summary["duration_ms"] == 1.0
    assert summary["max_displacement_mm"] == 0.5
    assert summary["max_translation_speed_mm_s"] == 3.0
    assert summary["max_normal_force_n"] == 4.0
    assert not samples
    assert node._return_hit_id is None


def test_shutdown_session_is_idempotent_and_releases_before_teardown():
    calls = []
    manager = SimpleNamespace(
        accepting_hits=True,
        compliance=SimpleNamespace(active=True),
        state=SimpleNamespace(name="WAITING_FOR_HIT"),
        last_release_message="release not requested",
    )

    def shutdown():
        calls.append("release")
        manager.accepting_hits = False
        manager.compliance.active = False
        manager.last_release_message = "released"

    manager.shutdown = shutdown
    node = SimpleNamespace(
        _shutdown_session_complete=False,
        _manager=manager,
        _rebound_motion=SimpleNamespace(moving=False),
        _save_compliance_session_summary=lambda outcome, detail: calls.append(
            ("summary", outcome, detail)
        ),
        _record_logger=SimpleNamespace(
            end_session=lambda: calls.append("end_session")
        ),
        _restore_collision_sensitivity=lambda reason: calls.append(
            ("restore_collision_sensitivity", reason)
        ),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: calls.append(("log", message))
        ),
    )

    HitAnalyzerNode.shutdown_session(node)
    HitAnalyzerNode.shutdown_session(node)

    assert calls[0] == "release"
    assert calls.count("release") == 1
    assert calls.count("end_session") == 1


def test_absolute_wrench_watchdog_keeps_force_and_torque_limits_independent():
    parameters = {
        "maximum_total_force_n": 10.0,
        "maximum_total_torque_nm": 2.0,
    }
    node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=parameters[name])
    )

    assert HitAnalyzerNode._absolute_wrench_limit_fault(
        node, (6.0, 8.0, 0.0, 0.0, 0.0, 2.0)
    ) is None
    force_fault = HitAnalyzerNode._absolute_wrench_limit_fault(
        node, (6.1, 8.0, 0.0, 0.0, 0.0, 0.0)
    )
    torque_fault = HitAnalyzerNode._absolute_wrench_limit_fault(
        node, (0.0, 0.0, 0.0, 0.0, 1.3, 1.6)
    )

    assert force_fault is not None and "force exceeded" in force_fault
    assert torque_fault is not None and "torque exceeded" in torque_fault


def test_immediate_rebound_starts_once_on_contact_threshold():
    calls = []
    rebound = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        moving=False,
        start_retreat=lambda force, reference: (
            calls.append((force, reference)) or (True, "retreat commanded")
        ),
    )
    node = SimpleNamespace(
        _rebound_motion=rebound,
        _compliance_reference_pose_mm_deg=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
        _reset_rt_watchdog_after_motion_service=lambda: calls.append("reset"),
        _fault_session=lambda *args: calls.append(("fault", args)),
        get_logger=lambda: SimpleNamespace(info=lambda message: calls.append(message)),
    )

    assert HitAnalyzerNode._start_rebound_on_contact(node, 12.0, 0.2)
    assert calls[0] == (12.0, node._compliance_reference_pose_mm_deg)
    assert "reset" in calls

    rebound.moving = True
    before = tuple(calls)
    assert HitAnalyzerNode._start_rebound_on_contact(node, 30.0, 1.0)
    assert tuple(calls) == before


def test_combination_session_suppresses_commanded_rebound():
    calls = []
    node = SimpleNamespace(
        _session_rebound_enabled=False,
        _rebound_motion=SimpleNamespace(
            config=SimpleNamespace(enabled=True),
            moving=False,
            start_retreat=lambda *args: calls.append(args),
        ),
    )

    assert HitAnalyzerNode._start_rebound_on_contact(node, 30.0, 1.0)
    assert calls == []


def test_combination_session_skips_return_to_reference_state():
    node = SimpleNamespace(
        _session_rebound_enabled=False,
        _manager=SimpleNamespace(
            compliance=SimpleNamespace(config=SimpleNamespace(enabled=True))
        ),
    )

    assert not HitAnalyzerNode._requires_return_after_hit(node)


def test_commanded_return_uses_fixed_compliance_stiffness():
    calls = []
    node = SimpleNamespace(
        _rebound_motion=SimpleNamespace(
            start_return=lambda: (calls.append("move") or (True, "accepted"))
        ),
        _reset_rt_watchdog_after_motion_service=lambda: calls.append("reset"),
        _fault_session=lambda *args: calls.append(("fault", args)),
        get_logger=lambda: SimpleNamespace(info=lambda message: calls.append(message)),
    )

    assert HitAnalyzerNode._start_commanded_return(node, 50.0)
    assert "reset" in calls
    assert any("fixed compliance stiffness retained" in item for item in calls)


def test_rehit_requires_actual_return_phase_and_five_mm_return_progress():
    rebound = SimpleNamespace(
        config=SimpleNamespace(
            enabled=True,
            rehit_enabled=True,
            rehit_arm_return_progress_mm=5.0,
        ),
        moving=True,
        phase=ReboundPhase.RETREATING,
        commanded_distance_mm=50.0,
    )
    node = SimpleNamespace(
        _rebound_motion=rebound,
        _manager=SimpleNamespace(target_hit_count=0, hit_count=1),
        _processor=SimpleNamespace(
            compressive_normal_force=lambda wrench: float(wrench[0])
        ),
        get_parameter=lambda name: SimpleNamespace(value=10.0),
    )

    assert not HitAnalyzerNode._should_interrupt_return_for_contact(
        node, (20.0,), 44.0
    )
    rebound.phase = ReboundPhase.RETURNING
    assert not HitAnalyzerNode._should_interrupt_return_for_contact(
        node, (20.0,), 49.0
    )
    assert HitAnalyzerNode._should_interrupt_return_for_contact(
        node, (20.0,), 45.0
    )


def test_activation_wrench_watchdog_uses_separate_tight_limits():
    parameters = {
        "maximum_total_force_n": 80.0,
        "maximum_total_torque_nm": 8.0,
        "maximum_activation_total_force_n": 10.0,
        "maximum_activation_total_torque_nm": 3.0,
    }
    node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=parameters[name])
    )

    activation_fault = HitAnalyzerNode._absolute_wrench_limit_fault(
        node, (11.0, 0.0, 0.0, 0.0, 0.0, 0.0), activation=True
    )
    session_fault = HitAnalyzerNode._absolute_wrench_limit_fault(
        node, (11.0, 0.0, 0.0, 0.0, 0.0, 0.0), activation=False
    )

    assert activation_fault is not None
    assert "activation total force" in activation_fault
    assert session_fault is None


def test_compliance_stabilization_timeout_faults_instead_of_waiting_forever():
    faults = []
    parameters = {
        "compliance_stabilization_timeout_ms": 1.0,
    }
    node = SimpleNamespace(
        _compliance_stabilization_started_monotonic_ns=0,
        _fault_session=lambda detail, displacement: faults.append(
            (detail, displacement)
        ),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    HitAnalyzerNode._process_compliance_stabilization(
        node,
        0,
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        1,
    )

    assert faults == [("compliance stabilization timeout", 0.0)]


def test_stable_compliance_recaptures_tcp_and_enables_hits():
    calls = []
    parameters = {
        "compliance_stabilization_timeout_ms": 5000.0,
        "compliance_stabilization_velocity_tolerance_mm_s": 0.2,
        "compliance_stabilization_angular_velocity_tolerance_deg_s": 0.1,
    }
    calibration = SimpleNamespace(
        created_at="2026-08-06T00:00:00+09:00",
        sample_count=100,
        offset=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
        stddev=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01),
    )

    class Processor:
        zero_calibration = None
        calibration_sample_count = 0

        def process(self, timestamp_ns, wrench):
            del timestamp_ns, wrench
            self.zero_calibration = calibration
            self.calibration_sample_count = 1

    compliance = SimpleNamespace(
        STATE_STANDBY=1,
        maximum_observed_displacement_mm=0.42,
        maximum_observed_angular_displacement_deg=0.03,
        recapture_reference_tcp_position=lambda pose: (
            calls.append(("reference", pose)) or tuple(pose[:3])
        ),
    )
    manager = SimpleNamespace(
        compliance=compliance,
        compliance_stabilized=lambda: (
            calls.append("stabilized")
            or SimpleNamespace(success=True, message="ready")
        ),
    )
    node = SimpleNamespace(
        _compliance_stabilization_started_monotonic_ns=time.monotonic_ns(),
        _processor=Processor(),
        _manager=manager,
        _record_logger=SimpleNamespace(
            log_compliance_baseline=lambda baseline: (
                calls.append(("baseline", baseline)) or (True, "")
            )
        ),
        _robot_ready=False,
        _session_maximum_total_force_n=4.0,
        _session_maximum_total_torque_nm=0.4,
        _fault_session=lambda detail, displacement: calls.append(
            ("fault", detail, displacement)
        ),
        _log_throttled=lambda *args: None,
        _publish_state=lambda: calls.append("published"),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: SimpleNamespace(info=lambda message: calls.append(message)),
    )

    pose = (10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
    HitAnalyzerNode._process_compliance_stabilization(
        node,
        123,
        calibration.offset,
        pose,
        (0.0,) * 6,
        1,
    )

    assert ("reference", pose) in calls
    assert "stabilized" in calls
    assert "published" in calls
    baseline_call = next(
        call
        for call in calls
        if isinstance(call, tuple) and call[0] == "baseline"
    )
    assert baseline_call[1]["activation_maximum_tcp_displacement_mm"] == 0.42
    assert baseline_call[1][
        "activation_maximum_tcp_angular_displacement_deg"
    ] == 0.03
    assert baseline_call[1]["activation_maximum_total_force_n"] == 4.0
    assert baseline_call[1]["activation_maximum_total_torque_nm"] == 0.4
    assert node._robot_ready
    assert node._compliance_stabilization_started_monotonic_ns is None
    assert not any(isinstance(call, tuple) and call[0] == "fault" for call in calls)


def test_error_state_does_not_restart_zero_calibration():
    calls = []
    message = RtMittSample()
    message.frame_id = "mitt_tool_corrected"
    message.corrected_wrench = [0.0] * 6
    message.tcp_pose_mm_deg = [0.0] * 6
    message.tcp_velocity_mm_deg_s = [0.0] * 6
    message.robot_state = 1
    manager = SimpleNamespace(
        session_active=False,
        state=SystemState.ERROR,
        compliance=SimpleNamespace(config=SimpleNamespace(enabled=True)),
    )
    node = SimpleNamespace(
        _expected_frame_id="mitt_tool_corrected",
        _manager=manager,
        _processor=SimpleNamespace(process=lambda *args: calls.append(args)),
        _last_rt_sample_monotonic_ns=0,
        _log_throttled=lambda *args: None,
        _valid_intentional_motion_pose=lambda sample: tuple(sample.tcp_pose_mm_deg),
        _intentional_motion_lock=threading.Lock(),
        _intentional_reference_pending=False,
        _intentional_motion_active=False,
        _intentional_motion_deadline=0.0,
        _post_move_zero_active=False,
        _on_rt_sample_core=lambda sample: HitAnalyzerNode._on_rt_sample_core(node, sample),
    )

    HitAnalyzerNode._on_rt_sample(node, message)

    assert calls == []


def test_bounded_motion_service_restarts_rt_sample_deadline(monkeypatch):
    node = SimpleNamespace(_last_rt_sample_monotonic_ns=1)
    monkeypatch.setattr(time, "monotonic_ns", lambda: 987_654_321)

    HitAnalyzerNode._reset_rt_watchdog_after_motion_service(node)

    assert node._last_rt_sample_monotonic_ns == 987_654_321


def test_rt_watchdog_is_disabled_during_intentional_stop():
    faults = []
    node = SimpleNamespace(
        _manager=SimpleNamespace(
            session_active=True,
            state=SystemState.STOPPING,
            compliance=SimpleNamespace(config=SimpleNamespace(enabled=True)),
        ),
        _last_rt_sample_monotonic_ns=0,
        _fault_session=lambda *args: faults.append(args),
        _intentional_motion_lock=threading.Lock(),
        _intentional_motion_active=False,
        _intentional_motion_deadline=0.0,
    )

    HitAnalyzerNode._on_rt_watchdog_timer(node)

    assert faults == []


def test_compliance_summary_separates_post_reference_motion_from_activation():
    calls = []
    node = SimpleNamespace(
        _record_logger=SimpleNamespace(
            session_id="session",
            log_compliance_summary=lambda summary: (
                calls.append(summary) or (True, "")
            ),
        ),
        _manager=SimpleNamespace(
            hit_count=5,
            compliance=SimpleNamespace(
                config=SimpleNamespace(enabled=True),
                maximum_observed_displacement_mm=7.5,
                maximum_observed_angular_displacement_deg=0.4,
            ),
        ),
        _session_maximum_tcp_displacement_mm=8.0,
        _session_maximum_tcp_angular_displacement_deg=0.5,
        _session_maximum_total_force_n=40.0,
        _session_maximum_total_torque_nm=3.0,
        get_logger=lambda: SimpleNamespace(error=lambda message: None),
    )

    HitAnalyzerNode._save_compliance_session_summary(node, "STOPPED", "done")

    assert calls[0]["maximum_tcp_displacement_mm"] == 8.0
    assert calls[0]["maximum_post_reference_tcp_displacement_mm"] == 7.5
    assert calls[0][
        "maximum_post_reference_tcp_angular_displacement_deg"
    ] == 0.4

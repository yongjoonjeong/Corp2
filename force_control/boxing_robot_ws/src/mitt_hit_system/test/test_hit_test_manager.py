from types import SimpleNamespace

from mitt_hit_system.common.enums import SystemState
from mitt_hit_system.hit_test_manager import HitTestManager


class MockCompliance:
    def __init__(self, *, ready=True, standby=True, enable_success=True):
        self.ready = ready
        self.standby = standby
        self.enable_success = enable_success
        self.active = False
        self.enable_calls = 0
        self.release_calls = 0
        self.config = SimpleNamespace(enabled=True)
        self.release_failed_locked = False

    def services_ready(self):
        return self.ready

    def robot_is_standby(self):
        return self.standby

    def enable(self):
        self.enable_calls += 1
        self.active = self.enable_success
        return self.enable_success, "enabled" if self.enable_success else "failed"

    def release(self):
        self.release_calls += 1
        self.active = False
        return True, "released"


def manager(compliance=None, *, continuous=False):
    sessions = []
    value = HitTestManager(
        compliance or MockCompliance(),
        lambda: sessions.append("new") or "id",
        allow_continuous_session=continuous,
    )
    return value, sessions


def test_start_before_zero_is_rejected_without_session():
    value, sessions = manager()
    assert not value.start(1).success
    assert sessions == []


def test_idle_system_can_rezero_at_current_pose_before_start():
    value, _ = manager()
    value.calibration_ready()

    result = value.begin_calibration()
    assert result.success
    assert value.state is SystemState.CALIBRATING

    value.calibration_ready()
    assert value.state is SystemState.READY
    assert value.start(1).success
    assert not value.begin_calibration().success


def test_zero_target_and_duplicate_start_are_rejected():
    value, sessions = manager()
    value.calibration_ready()
    assert not value.start(0).success
    assert value.start(2).success
    assert not value.start(2).success
    assert sessions == ["new"]


def test_zero_target_runs_continuously_until_explicit_stop_when_enabled():
    compliance = MockCompliance()
    value, sessions = manager(compliance, continuous=True)
    value.calibration_ready()

    assert value.start(0).success
    assert value.session_active
    assert value.compliance_stabilized().success
    assert value.begin_analysis()
    assert value.complete_hit(require_return=True) is None
    assert value.complete_return() is None
    assert value.hit_count == 1
    assert value.state is SystemState.WAITING_FOR_HIT
    assert value.accepting_hits
    assert compliance.release_calls == 0

    assert value.stop().success
    assert compliance.release_calls == 1
    assert sessions == ["new"]


def test_unavailable_service_and_false_response_prevent_hit_acceptance():
    unavailable, sessions = manager(MockCompliance(ready=False))
    unavailable.calibration_ready()
    assert not unavailable.start(1).success
    assert not unavailable.accepting_hits
    assert sessions == []

    failed_compliance = MockCompliance(enable_success=False)
    failed, sessions = manager(failed_compliance)
    failed.calibration_ready()
    assert not failed.start(1).success
    assert not failed.accepting_hits
    assert failed_compliance.release_calls == 1
    assert sessions == []


def test_stop_releases_exactly_once():
    compliance = MockCompliance()
    value, _ = manager(compliance)
    value.calibration_ready()
    assert value.start(3).success
    assert value.stop().success
    assert compliance.release_calls == 1
    assert value.state is SystemState.READY


def test_target_count_auto_stop_releases_and_completes():
    compliance = MockCompliance()
    value, _ = manager(compliance)
    value.calibration_ready()
    assert value.start(1).success
    assert value.compliance_stabilized().success
    assert value.begin_analysis()
    result = value.complete_hit()
    assert result is not None and result.success
    assert compliance.release_calls == 1
    assert value.state is SystemState.TEST_COMPLETE
    assert not value.accepting_hits


def test_compliant_hit_waits_for_return_before_next_hit_or_release():
    compliance = MockCompliance()
    value, _ = manager(compliance)
    value.calibration_ready()
    assert value.start(2).success
    assert value.state is SystemState.STABILIZING_COMPLIANCE
    assert not value.accepting_hits
    assert not value.begin_analysis()
    assert value.compliance_stabilized().success
    assert value.begin_analysis()

    assert value.complete_hit(require_return=True) is None
    assert compliance.release_calls == 0
    assert value.state is SystemState.RETURNING_TO_REFERENCE
    assert value.session_active
    assert not value.accepting_hits
    assert value.complete_return() is None
    assert compliance.release_calls == 0
    assert value.state is SystemState.WAITING_FOR_HIT
    assert value.accepting_hits

    assert value.begin_analysis()
    value.complete_hit(require_return=True)
    stopped = value.complete_return()
    assert stopped is not None and stopped.success
    assert compliance.release_calls == 1
    assert value.state is SystemState.TEST_COMPLETE


def test_continuous_session_can_accept_new_hit_during_return():
    compliance = MockCompliance()
    value, _ = manager(compliance, continuous=True)
    value.calibration_ready()
    assert value.start(0).success
    assert value.compliance_stabilized().success
    assert value.begin_analysis()
    assert value.complete_hit(require_return=True) is None

    result = value.interrupt_return_for_hit()

    assert result.success
    assert value.state is SystemState.WAITING_FOR_HIT
    assert value.accepting_hits
    assert compliance.release_calls == 0


def test_shutdown_is_the_only_normal_release_before_target_completion():
    compliance = MockCompliance()
    value, _ = manager(compliance)
    value.calibration_ready()
    assert value.start(3).success
    assert compliance.release_calls == 0

    assert value.compliance_stabilized().success
    assert value.begin_analysis()
    assert value.complete_hit(require_return=True) is None
    assert value.complete_return() is None
    assert compliance.release_calls == 0

    value.shutdown()

    assert compliance.release_calls == 1
    assert not compliance.active


def test_rt_or_robot_state_fault_releases_active_compliance():
    compliance = MockCompliance()
    value, _ = manager(compliance)
    value.calibration_ready()
    assert value.start(2).success
    value.fault("RT timeout")
    assert compliance.release_calls == 1
    assert value.state is SystemState.ERROR


def test_stabilization_transition_is_valid_only_once_after_start():
    value, _ = manager()
    value.calibration_ready()

    assert not value.compliance_stabilized().success
    assert value.start(2).success
    assert value.session_active
    assert value.state is SystemState.STABILIZING_COMPLIANCE
    assert not value.accepting_hits

    assert value.compliance_stabilized().success
    assert value.state is SystemState.WAITING_FOR_HIT
    assert value.accepting_hits
    assert not value.compliance_stabilized().success


def test_error_state_and_release_failure_lock_block_restart():
    compliance = MockCompliance()
    value, sessions = manager(compliance)
    value.calibration_ready()
    assert value.start(1).success
    value.fault("watchdog fault")

    assert not value.start(1).success
    assert sessions == ["new"]

    second, sessions = manager()
    second.calibration_ready()
    second.compliance.release_failed_locked = True
    assert not second.start(1).success
    assert sessions == []

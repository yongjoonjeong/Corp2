"""ROS-independent hit-test lifecycle with session-long compliance."""

from dataclasses import dataclass
from typing import Callable

from mitt_hit_system.common.enums import SystemState


@dataclass(frozen=True)
class TransitionResult:
    success: bool
    message: str


class HitTestManager:
    def __init__(
        self,
        compliance: object,
        start_session: Callable[[], str],
        *,
        allow_continuous_session: bool = False,
    ) -> None:
        self.compliance = compliance
        self._start_session = start_session
        self.state = SystemState.CALIBRATING
        self.detail = "waiting for wrench zero calibration"
        self.accepting_hits = False
        self.target_hit_count = 0
        self.hit_count = 0
        self.last_release_message = "release not requested"
        self.allow_continuous_session = bool(allow_continuous_session)

    @property
    def active(self) -> bool:
        return self.accepting_hits

    @property
    def session_active(self) -> bool:
        return self.state in (
            SystemState.WAITING_FOR_HIT,
            SystemState.STABILIZING_COMPLIANCE,
            SystemState.ANALYZING,
            SystemState.RETURNING_TO_REFERENCE,
            SystemState.STOPPING,
        )

    def calibration_ready(self) -> None:
        if self.state is SystemState.CALIBRATING:
            self._set(SystemState.READY, "zero calibration complete")

    def begin_calibration(self) -> TransitionResult:
        """Return an idle system to zero calibration at its current pose."""
        if self.session_active or self.accepting_hits:
            return TransitionResult(False, "cannot recalibrate during an active hit test")
        if self.state is SystemState.ERROR:
            return TransitionResult(False, "recalibration is locked while system state is ERROR")
        self.target_hit_count = 0
        self.hit_count = 0
        self._set(SystemState.CALIBRATING, "recalibrating wrench zero at current pose")
        return TransitionResult(True, self.detail)

    def start(self, target_hit_count: int) -> TransitionResult:
        if self.state is SystemState.CALIBRATING:
            return TransitionResult(False, "zero calibration is not complete")
        if self.state is SystemState.ERROR:
            return TransitionResult(False, "Start is locked while system state is ERROR")
        if bool(getattr(self.compliance, "release_failed_locked", False)):
            return TransitionResult(False, "Start is locked after compliance release failure")
        if target_hit_count < 0 or (
            target_hit_count == 0 and not self.allow_continuous_session
        ):
            return TransitionResult(
                False,
                "target_hit_count must be greater than zero unless continuous session mode is enabled",
            )
        if self.accepting_hits or self.state in (
            SystemState.ENABLING_COMPLIANCE,
            SystemState.STABILIZING_COMPLIANCE,
            SystemState.STOPPING,
        ):
            return TransitionResult(False, "a hit test is already active")
        if not self.compliance.services_ready():
            return TransitionResult(False, "required robot service is unavailable")
        try:
            if not self.compliance.robot_is_standby():
                return TransitionResult(False, "robot is not in STATE_STANDBY")
        except Exception as error:
            return TransitionResult(False, f"RT state check failed: {error}")
        self._set(SystemState.ENABLING_COMPLIANCE, "starting test")
        success, message = self.compliance.enable()
        if not success:
            released, release_message = self.compliance.release()
            if not released:
                message = f"{message}; {release_message}"
            self._set(SystemState.ERROR, message)
            return TransitionResult(False, message)
        try:
            self._start_session()
        except Exception as error:
            detail = f"session creation failed: {error}"
            released, release_message = self.compliance.release()
            if not released:
                detail = f"{detail}; {release_message}"
            self._set(SystemState.ERROR, detail)
            return TransitionResult(False, self.detail)
        self.target_hit_count = int(target_hit_count)
        self.hit_count = 0
        compliance_enabled = bool(
            getattr(self.compliance.config, "enabled", False)
        )
        self.accepting_hits = not compliance_enabled
        if compliance_enabled:
            session_kind = "continuous punch session" if target_hit_count == 0 else "test"
            detail = (
                f"compliance active; {session_kind} waiting for post-activation stabilization"
            )
            self._set(SystemState.STABILIZING_COMPLIANCE, detail)
        else:
            detail = "test active"
            self._set(SystemState.WAITING_FOR_HIT, detail)
        return TransitionResult(True, detail)

    def compliance_stabilized(self) -> TransitionResult:
        if self.state is not SystemState.STABILIZING_COMPLIANCE:
            return TransitionResult(
                False, "compliance stabilization is not active"
            )
        self.accepting_hits = True
        detail = "test active; compliance remains enabled until session termination"
        self._set(SystemState.WAITING_FOR_HIT, detail)
        return TransitionResult(True, detail)

    def begin_analysis(self) -> bool:
        if not self.accepting_hits:
            return False
        self._set(SystemState.ANALYZING, "analyzing contact")
        return True

    def complete_hit(self, *, require_return: bool = False) -> TransitionResult | None:
        if not self.accepting_hits:
            return None
        self.hit_count += 1
        if require_return:
            self.accepting_hits = False
            self._set(
                SystemState.RETURNING_TO_REFERENCE,
                "waiting for compliant return to reference",
            )
            return None
        if self.target_hit_count > 0 and self.hit_count >= self.target_hit_count:
            return self.stop(completed=True)
        self._set(SystemState.WAITING_FOR_HIT, "waiting for hit")
        return None

    def complete_return(self) -> TransitionResult | None:
        if self.state is not SystemState.RETURNING_TO_REFERENCE:
            return None
        if self.target_hit_count > 0 and self.hit_count >= self.target_hit_count:
            return self.stop(completed=True)
        self.accepting_hits = True
        self._set(SystemState.WAITING_FOR_HIT, "return complete; waiting for hit")
        return None

    def interrupt_return_for_hit(self) -> TransitionResult:
        """Re-open hit capture when a continuous session is struck on return."""
        if self.state is not SystemState.RETURNING_TO_REFERENCE:
            return TransitionResult(False, "system is not returning")
        if self.target_hit_count > 0 and self.hit_count >= self.target_hit_count:
            return TransitionResult(False, "target hit count already reached")
        self.accepting_hits = True
        self._set(
            SystemState.WAITING_FOR_HIT,
            "return interrupted by a new contact",
        )
        return TransitionResult(True, self.detail)

    def stop(self, *, completed: bool = False) -> TransitionResult:
        """Terminate the session and perform its single normal release."""
        if not self.accepting_hits and self.state not in (
            SystemState.ENABLING_COMPLIANCE,
            SystemState.STABILIZING_COMPLIANCE,
            SystemState.ANALYZING,
            SystemState.RETURNING_TO_REFERENCE,
        ):
            return TransitionResult(False, "no hit test is active")
        self.accepting_hits = False
        self._set(SystemState.STOPPING, "releasing compliance")
        success, message = self.compliance.release()
        self.last_release_message = message
        self.target_hit_count = 0
        if not success:
            self._set(SystemState.ERROR, message)
            return TransitionResult(False, message)
        self._set(
            SystemState.TEST_COMPLETE if completed else SystemState.READY,
            "target hit count reached" if completed else "test stopped",
        )
        return TransitionResult(True, self.detail)

    def fault(self, detail: str) -> TransitionResult:
        """Force a safety termination; faults must not leave compliance active."""
        was_active = self.accepting_hits or bool(self.compliance.active)
        self.accepting_hits = False
        if was_active:
            success, message = self.compliance.release()
            self.last_release_message = message
            if not success:
                detail = f"{detail}; {message}"
        self._set(SystemState.ERROR, detail)
        return TransitionResult(False, detail)

    def shutdown(self) -> None:
        """Release only when node shutdown terminates an active session."""
        if self.accepting_hits or bool(self.compliance.active):
            self.accepting_hits = False
            _, self.last_release_message = self.compliance.release()

    def _set(self, state: SystemState, detail: str) -> None:
        self.state = state
        self.detail = detail

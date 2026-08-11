"""Isolated, opt-in adapter for Doosan task compliance services."""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

from mitt_hit_system.wrench_frame_adapter import (
    Matrix3,
    rotation_distance_degrees,
    rotation_from_zyz_degrees,
)


@dataclass(frozen=True)
class RtHealth:
    healthy: bool
    robot_state: int | None
    tcp_displacement_mm: float
    tcp_angular_displacement_deg: float
    detail: str


@dataclass(frozen=True)
class ComplianceConfig:
    enabled: bool = False
    stiffness_verified: bool = False
    stiffness: tuple[float, ...] = ()
    reference_id: int = 1
    activation_time_sec: float = 0.5
    service_timeout_sec: float = 0.2
    maximum_tcp_displacement_mm: float = 0.0
    maximum_tcp_angular_displacement_deg: float = 0.0
    maximum_activation_tcp_displacement_mm: float = 0.0
    maximum_activation_tcp_angular_displacement_deg: float = 0.0
    release_max_attempts: int = 3
    release_retry_interval_sec: float = 0.05
    require_normal_speed_mode: bool = False

    def validate_for_activation(self) -> None:
        if not self.enabled:
            return
        if not self.stiffness_verified:
            raise ValueError("compliance stiffness has not been verified")
        if len(self.stiffness) != 6 or not all(
            math.isfinite(value) and value > 0.0 for value in self.stiffness
        ):
            raise ValueError("six verified positive stiffness values are required")
        if self.reference_id != 1:
            raise ValueError("compliance reference must be DR_TOOL (1)")
        if not 0.0 <= self.activation_time_sec <= 1.0:
            raise ValueError("activation time must be between 0 and 1 second")
        if self.service_timeout_sec <= 0.0:
            raise ValueError("service timeout must be positive")
        if (
            not math.isfinite(self.maximum_tcp_displacement_mm)
            or self.maximum_tcp_displacement_mm <= 0.0
        ):
            raise ValueError(
                "maximum TCP displacement must be explicitly configured"
            )
        if (
            not math.isfinite(self.maximum_tcp_angular_displacement_deg)
            or self.maximum_tcp_angular_displacement_deg <= 0.0
        ):
            raise ValueError(
                "maximum TCP angular displacement must be explicitly configured"
            )
        if (
            not math.isfinite(self.maximum_activation_tcp_displacement_mm)
            or self.maximum_activation_tcp_displacement_mm <= 0.0
        ):
            raise ValueError(
                "maximum activation TCP displacement must be explicitly configured"
            )
        if (
            not math.isfinite(
                self.maximum_activation_tcp_angular_displacement_deg
            )
            or self.maximum_activation_tcp_angular_displacement_deg <= 0.0
        ):
            raise ValueError(
                "maximum activation TCP angular displacement must be explicitly configured"
            )
        if self.release_max_attempts <= 0:
            raise ValueError("release_max_attempts must be positive")
        if (
            not math.isfinite(self.release_retry_interval_sec)
            or self.release_retry_interval_sec < 0.0
        ):
            raise ValueError("release retry interval must be finite and non-negative")


class ComplianceController:
    """Small synchronous facade whose clients can be replaced by test doubles."""

    STATE_STANDBY = 1

    def __init__(
        self,
        config: ComplianceConfig,
        activate_client: Any | None = None,
        release_client: Any | None = None,
        rt_client: Any | None = None,
        speed_mode_client: Any | None = None,
        *,
        activate_request_factory: Callable[[], Any] | None = None,
        release_request_factory: Callable[[], Any] | None = None,
        rt_request_factory: Callable[[], Any] | None = None,
        speed_mode_request_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._activate_client = activate_client
        self._release_client = release_client
        self._rt_client = rt_client
        self._speed_mode_client = speed_mode_client
        self._activate_request_factory = activate_request_factory
        self._release_request_factory = release_request_factory
        self._rt_request_factory = rt_request_factory
        self._speed_mode_request_factory = speed_mode_request_factory
        self.active = False
        self.release_failed_locked = False
        self.reference_tcp_position: tuple[float, float, float] | None = None
        self.reference_tcp_rotation: Matrix3 | None = None
        self.maximum_observed_displacement_mm = 0.0
        self.maximum_observed_angular_displacement_deg = 0.0
        self.observed_robot_state_counts: dict[str, int] = {}
        self._moving_allowed_until = 0.0
        self._settled_reference_captured = False

    def services_ready(self) -> bool:
        if not self.config.enabled:
            return True
        clients = [
            self._activate_client,
            self._release_client,
            self._rt_client,
        ]
        if self.config.require_normal_speed_mode:
            clients.append(self._speed_mode_client)
        return all(
            client is not None and client.service_is_ready()
            for client in clients
        )

    def robot_is_standby(self) -> bool:
        if not self.config.enabled:
            return True
        if self.config.require_normal_speed_mode:
            response = self._call(
                self._speed_mode_client,
                self._speed_mode_request_factory,
            )
            if response is None or not bool(response.success):
                raise RuntimeError("GetRobotSpeedMode returned success=false")
            speed_mode = int(response.speed_mode)
            if speed_mode != 0:
                raise RuntimeError(
                    "robot speed mode is REDUCED(1); punch session requires "
                    "NORMAL(0) because the TP reduced-mode TCP force limit is lower"
                )
        state, pose = self._read_rt_state()
        if state == self.STATE_STANDBY:
            self.reference_tcp_position = pose[:3]  # type: ignore[assignment]
            self.reference_tcp_rotation = rotation_from_zyz_degrees(pose[3:])
            self.maximum_observed_displacement_mm = 0.0
            self.maximum_observed_angular_displacement_deg = 0.0
            self.observed_robot_state_counts = {str(state): 1}
            self._settled_reference_captured = False
            return True
        return False

    def enable(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return True, "compliance disabled by configuration"
        if self.release_failed_locked:
            return False, "Start locked after compliance release failure"
        try:
            self.config.validate_for_activation()
        except ValueError as error:
            return False, str(error)
        request = self._make(self._activate_request_factory)
        request.stx = list(self.config.stiffness)
        request.ref = self.config.reference_id
        request.time = self.config.activation_time_sec
        try:
            response = self._invoke(self._activate_client, request)
        except Exception as error:
            return False, f"compliance activation failed: {error}"
        if response is None or not bool(response.success):
            return False, "compliance service returned success=false"
        if self.reference_tcp_position is None:
            return False, "RT TCP reference position was not captured"
        self.active = True
        self._moving_allowed_until = time.monotonic() + self.config.activation_time_sec
        return True, "compliance enabled"

    def release(self) -> tuple[bool, str]:
        if not self.config.enabled:
            self.active = False
            return True, "compliance disabled by configuration"
        if self.release_failed_locked:
            return False, "compliance release failure lock is active"
        errors: list[str] = []
        for attempt in range(1, self.config.release_max_attempts + 1):
            try:
                response = self._invoke(
                    self._release_client,
                    self._make(self._release_request_factory),
                )
                if response is not None and bool(response.success):
                    self.active = False
                    return True, f"compliance released (attempt {attempt})"
                errors.append(f"attempt {attempt}: success=false")
            except Exception as error:
                errors.append(f"attempt {attempt}: {error}")
            if attempt < self.config.release_max_attempts:
                time.sleep(self.config.release_retry_interval_sec)
        self.release_failed_locked = True
        return False, "compliance release failed and Start is locked: " + "; ".join(
            errors
        )

    def check_rt_health(self, *, allow_moving: bool = False) -> RtHealth:
        if not self.config.enabled:
            return RtHealth(
                True, self.STATE_STANDBY, 0.0, 0.0, "compliance disabled"
            )
        try:
            state, pose = self._read_rt_state()
        except Exception as error:
            return RtHealth(
                False, None, 0.0, 0.0, f"RT communication failed: {error}"
            )
        return self.observe_rt_state(
            state,
            pose,
            allow_moving=allow_moving,
        )

    def recapture_reference_tcp_position(
        self, tcp_pose_mm_deg: tuple[float, ...] | list[float]
    ) -> tuple[float, float, float]:
        """Use the settled compliant pose as the session return reference."""
        if self.config.enabled and not self.active:
            raise RuntimeError("compliance must be active before reference recapture")
        try:
            pose = tuple(float(value) for value in tcp_pose_mm_deg)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid RT TCP pose: {error}") from error
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError("RT TCP pose must contain six finite values")
        self.reference_tcp_position = pose[:3]  # type: ignore[assignment]
        self.reference_tcp_rotation = rotation_from_zyz_degrees(pose[3:])
        self.maximum_observed_displacement_mm = 0.0
        self.maximum_observed_angular_displacement_deg = 0.0
        self._settled_reference_captured = True
        return self.reference_tcp_position

    def observe_rt_state(
        self,
        robot_state: int,
        tcp_pose_mm_deg: tuple[float, ...] | list[float],
        *,
        allow_moving: bool | None = None,
    ) -> RtHealth:
        try:
            state = int(robot_state)
            pose = tuple(float(value) for value in tcp_pose_mm_deg)
        except (TypeError, ValueError) as error:
            return RtHealth(
                False, None, 0.0, 0.0, f"invalid RT observation: {error}"
            )
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            return RtHealth(False, state, 0.0, 0.0, "invalid RT TCP pose")
        position = pose[:3]
        try:
            rotation = rotation_from_zyz_degrees(pose[3:])
        except ValueError as error:
            return RtHealth(False, state, 0.0, 0.0, f"invalid RT TCP pose: {error}")
        state_key = str(state)
        self.observed_robot_state_counts[state_key] = (
            self.observed_robot_state_counts.get(state_key, 0) + 1
        )
        moving_is_allowed = (
            time.monotonic() < self._moving_allowed_until
            if allow_moving is None
            else allow_moving
        )
        if self.reference_tcp_position is None:
            return RtHealth(
                False, state, 0.0, 0.0, "RT TCP reference position is missing"
            )
        if self.reference_tcp_rotation is None:
            return RtHealth(
                False, state, 0.0, 0.0, "RT TCP reference rotation is missing"
            )
        displacement = math.dist(self.reference_tcp_position, position)
        angular_displacement = rotation_distance_degrees(
            self.reference_tcp_rotation, rotation
        )
        self.maximum_observed_displacement_mm = max(
            self.maximum_observed_displacement_mm, displacement
        )
        self.maximum_observed_angular_displacement_deg = max(
            self.maximum_observed_angular_displacement_deg,
            angular_displacement,
        )
        if state != self.STATE_STANDBY and not (moving_is_allowed and state == 2):
            return RtHealth(
                False,
                state,
                displacement,
                angular_displacement,
                f"robot left STATE_STANDBY: state={state}",
            )
        displacement_limit = (
            self.config.maximum_tcp_displacement_mm
            if self._settled_reference_captured
            else self.config.maximum_activation_tcp_displacement_mm
        )
        angular_displacement_limit = (
            self.config.maximum_tcp_angular_displacement_deg
            if self._settled_reference_captured
            else self.config.maximum_activation_tcp_angular_displacement_deg
        )
        limit_phase = "session" if self._settled_reference_captured else "activation"
        if displacement > displacement_limit:
            return RtHealth(
                False,
                state,
                displacement,
                angular_displacement,
                f"TCP displacement exceeded {limit_phase} limit: "
                f"{displacement:.3f} > {displacement_limit:.3f} mm",
            )
        if angular_displacement > angular_displacement_limit:
            return RtHealth(
                False,
                state,
                displacement,
                angular_displacement,
                f"TCP angular displacement exceeded {limit_phase} limit: "
                f"{angular_displacement:.3f} > "
                f"{angular_displacement_limit:.3f} deg",
            )
        detail = (
            "bounded compliance activation STATE_MOVING with normal "
            "TCP displacement"
            if state == 2
            else "RT state and TCP pose displacement normal"
        )
        return RtHealth(True, state, displacement, angular_displacement, detail)

    def _read_rt_state(self) -> tuple[int, tuple[float, ...]]:
        response = self._call(self._rt_client, self._rt_request_factory)
        if response is None:
            raise RuntimeError("ReadDataRt returned no response")
        state = int(response.data.robot_state)
        pose = tuple(float(value) for value in response.data.actual_tcp_position)
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError("actual_tcp_position must contain six finite values")
        return state, pose

    @staticmethod
    def _make(factory: Callable[[], Any] | None) -> Any:
        if factory is None:
            raise RuntimeError("request factory is not configured")
        return factory()

    def _call(self, client: Any, factory: Callable[[], Any] | None) -> Any:
        return self._invoke(client, self._make(factory))

    def _invoke(
        self,
        client: Any,
        request: Any,
        *,
        timeout_sec: float | None = None,
    ) -> Any:
        # call_async keeps every request bounded, including RT health checks and
        # release. Another executor worker completes production rclpy futures.
        future = client.call_async(request)
        bounded_timeout_sec = (
            self.config.service_timeout_sec
            if timeout_sec is None
            else float(timeout_sec)
        )
        deadline = time.monotonic() + bounded_timeout_sec
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError("service response timeout")
            time.sleep(0.001)
        return future.result()

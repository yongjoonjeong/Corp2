"""Bounded linear rebound motion while task compliance remains enabled."""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Callable, Sequence

from mitt_hit_system.wrench_frame_adapter import rotation_from_zyz_degrees


class ReboundPhase(str, Enum):
    IDLE = "IDLE"
    RETREATING = "RETREATING"
    RETURNING = "RETURNING"


@dataclass(frozen=True)
class ReboundMotionConfig:
    enabled: bool = False
    minimum_distance_mm: float = 0.0
    maximum_distance_mm: float = 0.0
    force_for_maximum_distance_n: float = 0.0
    retreat_velocity_mm_s: float = 0.0
    retreat_acceleration_mm_s2: float = 0.0
    return_velocity_mm_s: float = 0.0
    return_acceleration_mm_s2: float = 0.0
    target_tolerance_mm: float = 0.0
    motion_timeout_sec: float = 0.0
    service_timeout_sec: float = 0.2
    stop_state_moving_grace_sec: float = 0.5
    tool_z_direction_sign: int = -1
    rehit_enabled: bool = True
    rehit_arm_return_progress_mm: float = 0.0
    soft_stop_mode: int = 2

    def validate(self) -> None:
        if not self.enabled:
            return
        positive = (
            self.minimum_distance_mm,
            self.maximum_distance_mm,
            self.force_for_maximum_distance_n,
            self.retreat_velocity_mm_s,
            self.retreat_acceleration_mm_s2,
            self.return_velocity_mm_s,
            self.return_acceleration_mm_s2,
            self.target_tolerance_mm,
            self.motion_timeout_sec,
            self.service_timeout_sec,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("enabled rebound motion values must be finite and positive")
        if self.maximum_distance_mm < self.minimum_distance_mm:
            raise ValueError("maximum rebound distance must not be below minimum")
        if self.tool_z_direction_sign not in (-1, 1):
            raise ValueError("rebound Tool Z direction sign must be -1 or 1")
        if self.soft_stop_mode != 2:
            raise ValueError("rebound interruption must use DR_SSTO(2)")
        if (
            not math.isfinite(self.rehit_arm_return_progress_mm)
            or self.rehit_arm_return_progress_mm < 0.0
            or self.rehit_arm_return_progress_mm >= self.minimum_distance_mm
        ):
            raise ValueError(
                "rehit arm return progress must be finite, non-negative, "
                "and below minimum rebound distance"
            )
        if (
            not math.isfinite(self.stop_state_moving_grace_sec)
            or self.stop_state_moving_grace_sec < 0.0
        ):
            raise ValueError("rebound stop grace must be finite and non-negative")


class ReboundMotionController:
    """Issue one asynchronous retreat and one asynchronous return MoveLine."""

    DR_BASE = 0
    DR_MV_MOD_ABS = 0
    DR_SYNC_ASYNC = 1

    def __init__(
        self,
        config: ReboundMotionConfig,
        move_line_client: Any | None = None,
        move_stop_client: Any | None = None,
        *,
        move_line_request_factory: Callable[[], Any] | None = None,
        move_stop_request_factory: Callable[[], Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._move_line_client = move_line_client
        self._move_stop_client = move_stop_client
        self._move_line_request_factory = move_line_request_factory
        self._move_stop_request_factory = move_stop_request_factory
        self.phase = ReboundPhase.IDLE
        self.reference_pose_mm_deg: tuple[float, ...] | None = None
        self.target_pose_mm_deg: tuple[float, ...] | None = None
        self.commanded_distance_mm = 0.0
        self._phase_started_monotonic = 0.0
        self._moving_allowed_until = 0.0

    @property
    def moving(self) -> bool:
        return self.phase is not ReboundPhase.IDLE

    @property
    def allows_state_moving(self) -> bool:
        return self.moving or time.monotonic() < self._moving_allowed_until

    def services_ready(self) -> bool:
        if not self.config.enabled:
            return True
        return all(
            client is not None and client.service_is_ready()
            for client in (self._move_line_client, self._move_stop_client)
        )

    def distance_for_force(self, normal_force_n: float) -> float:
        force = float(normal_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise ValueError("normal force must be finite and non-negative")
        ratio = min(force / self.config.force_for_maximum_distance_n, 1.0)
        return max(
            self.config.minimum_distance_mm,
            ratio * self.config.maximum_distance_mm,
        )

    def start_retreat(
        self, normal_force_n: float, reference_pose_mm_deg: Sequence[float]
    ) -> tuple[bool, str]:
        if not self.config.enabled:
            return True, "commanded rebound disabled"
        if self.moving:
            return False, "rebound motion is already active"
        reference = self._pose(reference_pose_mm_deg)
        rotation = rotation_from_zyz_degrees(reference[3:])
        distance = self.distance_for_force(normal_force_n)
        tool_z_in_base = tuple(rotation[row][2] for row in range(3))
        sign = float(self.config.tool_z_direction_sign)
        target_position = tuple(
            reference[index] + sign * distance * tool_z_in_base[index]
            for index in range(3)
        )
        target = (*target_position, *reference[3:])
        success, detail = self._move_line(
            target,
            self.config.retreat_velocity_mm_s,
            self.config.retreat_acceleration_mm_s2,
        )
        if not success:
            return False, detail
        self.reference_pose_mm_deg = reference
        self.target_pose_mm_deg = target
        self.commanded_distance_mm = distance
        self.phase = ReboundPhase.RETREATING
        self._phase_started_monotonic = time.monotonic()
        return True, f"retreat commanded: {distance:.3f} mm"

    def retreat_target_reached(self, actual_pose_mm_deg: Sequence[float]) -> bool:
        if self.phase is not ReboundPhase.RETREATING:
            return False
        if self.reference_pose_mm_deg is None or self.target_pose_mm_deg is None:
            return False
        actual = self._pose(actual_pose_mm_deg)
        commanded = tuple(
            self.target_pose_mm_deg[index] - self.reference_pose_mm_deg[index]
            for index in range(3)
        )
        commanded_length = math.sqrt(sum(value * value for value in commanded))
        if commanded_length <= 0.0:
            return False
        # Task compliance may leave a small transverse offset even though the
        # controller completed the commanded Tool-Z travel. Gate the phase
        # transition on progress along the commanded retreat axis; independent
        # TCP displacement/rotation watchdogs still bound all other motion.
        actual_delta = tuple(
            actual[index] - self.reference_pose_mm_deg[index]
            for index in range(3)
        )
        axial_progress = sum(
            actual_delta[index] * commanded[index] / commanded_length
            for index in range(3)
        )
        return axial_progress >= commanded_length - self.config.target_tolerance_mm

    def start_return(self) -> tuple[bool, str]:
        if self.phase is not ReboundPhase.RETREATING:
            return False, "rebound is not retreating"
        if self.reference_pose_mm_deg is None:
            return False, "rebound reference pose is missing"
        success, detail = self._move_line(
            self.reference_pose_mm_deg,
            self.config.return_velocity_mm_s,
            self.config.return_acceleration_mm_s2,
        )
        if not success:
            return False, detail
        self.target_pose_mm_deg = self.reference_pose_mm_deg
        self.phase = ReboundPhase.RETURNING
        self._phase_started_monotonic = time.monotonic()
        return True, "return commanded"

    def timed_out(self) -> bool:
        return self.moving and (
            time.monotonic() - self._phase_started_monotonic
            > self.config.motion_timeout_sec
        )

    def soft_stop(self) -> tuple[bool, str]:
        if not self.moving:
            return True, "rebound motion already idle"
        request = self._make(self._move_stop_request_factory)
        request.stop_mode = self.config.soft_stop_mode
        try:
            response = self._invoke(self._move_stop_client, request)
        except Exception as error:
            return False, f"rebound soft stop failed: {error}"
        if response is None or not bool(response.success):
            return False, "rebound soft stop returned success=false"
        self.finish()
        self._moving_allowed_until = (
            time.monotonic() + self.config.stop_state_moving_grace_sec
        )
        return True, "rebound motion soft-stopped"

    def finish(self) -> None:
        self.phase = ReboundPhase.IDLE
        self.target_pose_mm_deg = None
        self.commanded_distance_mm = 0.0
        self._phase_started_monotonic = 0.0

    def _move_line(
        self,
        target_pose_mm_deg: Sequence[float],
        velocity_mm_s: float,
        acceleration_mm_s2: float,
    ) -> tuple[bool, str]:
        request = self._make(self._move_line_request_factory)
        request.pos = list(self._pose(target_pose_mm_deg))
        request.vel = [float(velocity_mm_s), 10.0]
        request.acc = [float(acceleration_mm_s2), 20.0]
        request.time = 0.0
        request.radius = 0.0
        request.ref = self.DR_BASE
        request.mode = self.DR_MV_MOD_ABS
        request.blend_type = 0
        request.sync_type = self.DR_SYNC_ASYNC
        try:
            response = self._invoke(self._move_line_client, request)
        except Exception as error:
            return False, f"MoveLine request failed: {error}"
        if response is None or not bool(response.success):
            return False, "MoveLine returned success=false"
        return True, "MoveLine accepted"

    def _invoke(self, client: Any, request: Any) -> Any:
        future = client.call_async(request)
        deadline = time.monotonic() + self.config.service_timeout_sec
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError("service response timeout")
            time.sleep(0.001)
        return future.result()

    @staticmethod
    def _make(factory: Callable[[], Any] | None) -> Any:
        if factory is None:
            raise RuntimeError("request factory is not configured")
        return factory()

    @staticmethod
    def _pose(values: Sequence[float]) -> tuple[float, ...]:
        pose = tuple(float(value) for value in values)
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError("pose must contain six finite values")
        return pose

    @staticmethod
    def _translation_error(
        actual_pose_mm_deg: Sequence[float], target_pose_mm_deg: Sequence[float] | None
    ) -> float:
        if target_pose_mm_deg is None:
            return math.inf
        actual = ReboundMotionController._pose(actual_pose_mm_deg)
        target = ReboundMotionController._pose(target_pose_mm_deg)
        return math.dist(actual[:3], target[:3])

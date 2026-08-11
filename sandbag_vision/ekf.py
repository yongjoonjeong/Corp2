from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterEstimate:
    stamp_ns: int
    position_mm: np.ndarray
    velocity_mm_s: np.ndarray
    position_std_mm: float
    measurement_age_ms: float
    valid: bool


class ConstantVelocityEkf:
    """Six-state [x,y,z,vx,vy,vz] filter with timestamped updates."""

    def __init__(
        self,
        acceleration_std_mm_s2: float,
        initial_position_std_mm: float,
        initial_velocity_std_mm_s: float,
        maximum_innovation_sigma: float,
    ) -> None:
        self.acceleration_std = max(float(acceleration_std_mm_s2), 1.0)
        self.initial_position_std = max(float(initial_position_std_mm), 1.0)
        self.initial_velocity_std = max(float(initial_velocity_std_mm_s), 1.0)
        self.maximum_innovation_sigma = max(float(maximum_innovation_sigma), 1.0)
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.stamp_ns: int | None = None
        self.last_measurement_ns: int | None = None
        self.rejected_updates = 0

    def reset(self) -> None:
        self.stamp_ns = None
        self.last_measurement_ns = None
        self.x.fill(0.0)
        self.P[:] = np.eye(6)

    def _transition(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt_s), 0.0)
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3) * dt
        q = self.acceleration_std**2
        process = np.zeros((6, 6), dtype=np.float64)
        process[:3, :3] = np.eye(3) * (dt**4 / 4.0) * q
        process[:3, 3:] = np.eye(3) * (dt**3 / 2.0) * q
        process[3:, :3] = process[:3, 3:]
        process[3:, 3:] = np.eye(3) * dt**2 * q
        return transition, process

    def _predict_mutating(self, stamp_ns: int) -> None:
        if self.stamp_ns is None:
            return
        if stamp_ns <= self.stamp_ns:
            return
        dt_s = (int(stamp_ns) - self.stamp_ns) / 1e9
        transition, process = self._transition(dt_s)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + process
        self.stamp_ns = int(stamp_ns)

    def update(self, position_mm: np.ndarray, stamp_ns: int, measurement_std_mm: float) -> bool:
        measurement = np.asarray(position_mm, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(measurement)):
            return False
        stamp_ns = int(stamp_ns)
        if self.stamp_ns is None:
            self.x[:3] = measurement
            self.P = np.diag(
                [self.initial_position_std**2] * 3 + [self.initial_velocity_std**2] * 3
            )
            self.stamp_ns = stamp_ns
            self.last_measurement_ns = stamp_ns
            return True
        if stamp_ns <= self.stamp_ns:
            return False
        self._predict_mutating(stamp_ns)
        observation = np.hstack((np.eye(3), np.zeros((3, 3))))
        noise = np.eye(3) * max(float(measurement_std_mm), 1.0) ** 2
        innovation = measurement - observation @ self.x
        innovation_covariance = observation @ self.P @ observation.T + noise
        mahalanobis_sq = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
        if mahalanobis_sq > self.maximum_innovation_sigma**2:
            self.rejected_updates += 1
            return False
        gain = self.P @ observation.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + gain @ innovation
        identity = np.eye(6)
        # Joseph form preserves positive semidefiniteness under float error.
        correction = identity - gain @ observation
        self.P = correction @ self.P @ correction.T + gain @ noise @ gain.T
        self.last_measurement_ns = stamp_ns
        return True

    def estimate(self, now_ns: int, maximum_age_ms: float) -> FilterEstimate | None:
        if self.stamp_ns is None or self.last_measurement_ns is None:
            return None
        now_ns = max(int(now_ns), self.stamp_ns)
        dt_s = (now_ns - self.stamp_ns) / 1e9
        transition, process = self._transition(dt_s)
        state = transition @ self.x
        covariance = transition @ self.P @ transition.T + process
        age_ms = (now_ns - self.last_measurement_ns) / 1e6
        return FilterEstimate(
            stamp_ns=now_ns,
            position_mm=state[:3].copy(),
            velocity_mm_s=state[3:].copy(),
            position_std_mm=float(np.sqrt(max(np.max(np.diag(covariance)[:3]), 0.0))),
            measurement_age_ms=age_ms,
            valid=age_ms <= float(maximum_age_ms),
        )

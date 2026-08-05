from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import doosan_posx_to_transform, transform_error


@dataclass(frozen=True)
class RobotPoseSample:
    T_base_flange_mm: np.ndarray
    flange_posx_mm_deg: np.ndarray
    rotation_source: str
    requested_monotonic_ns: int
    received_monotonic_ns: int
    stability_translation_mm: float = 0.0
    stability_rotation_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_frame": "robot_base",
            "moving_frame": "robot_flange",
            "units": {"translation": "mm", "angles": "deg"},
            "T_base_flange_mm": self.T_base_flange_mm.tolist(),
            "flange_posx_mm_deg": self.flange_posx_mm_deg.tolist(),
            "rotation_source": self.rotation_source,
            "orientation_convention": "intrinsic_Z-Y'-Z''_degrees",
            "requested_monotonic_ns": int(self.requested_monotonic_ns),
            "received_monotonic_ns": int(self.received_monotonic_ns),
            "stability": {
                "translation_spread_mm": float(self.stability_translation_mm),
                "rotation_spread_deg": float(self.stability_rotation_deg),
            },
        }


class DoosanFlangePoseProvider:
    """Read the M0609 tool-flange pose in DR_BASE.

    Both translation and rotation come from GetCurrentToolFlangePosx, so the
    recorded pose is independent of the currently selected TCP. The service's
    default orientation is Doosan intrinsic Z-Y'-Z'' Euler angles; converting
    those angles to a rotation matrix is exact apart from floating-point error.
    """

    def __init__(self, namespace: str = "/dsr01", timeout_s: float = 5.0) -> None:
        try:
            import rclpy
            from dsr_msgs2.srv import GetCurrentToolFlangePosx
        except ImportError as error:
            raise RuntimeError(
                "ROS 2/Doosan Python packages are unavailable. Source ROS Humble "
                "and the Doosan workspace install/setup.bash first."
            ) from error
        self.rclpy = rclpy
        self.GetCurrentToolFlangePosx = GetCurrentToolFlangePosx
        self.timeout_s = float(timeout_s)
        if not rclpy.ok():
            rclpy.init(args=[])
            self.owns_rclpy = True
        else:
            self.owns_rclpy = False
        self.node = rclpy.create_node("sandbag_external_calibration_pose_reader")
        root = "/" + namespace.strip("/") if namespace.strip("/") else ""
        self.posx_name = f"{root}/aux_control/get_current_tool_flange_posx"
        self.posx_client = self.node.create_client(
            GetCurrentToolFlangePosx, self.posx_name
        )
        if not self.posx_client.wait_for_service(timeout_sec=self.timeout_s):
            self.close()
            raise RuntimeError(f"Service unavailable: {self.posx_name}")

    def _call(self, request: Any) -> Any:
        future = self.posx_client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=self.timeout_s
        )
        response = future.result() if future.done() else None
        if response is None:
            raise RuntimeError(f"Robot service timeout: {self.posx_name}")
        if hasattr(response, "success") and not bool(response.success):
            raise RuntimeError(f"Robot service rejected request: {self.posx_name}")
        return response

    def sample(self) -> RobotPoseSample:
        requested_ns = time.monotonic_ns()
        request = self.GetCurrentToolFlangePosx.Request()
        request.ref = 0  # DR_BASE
        response = self._call(request)
        values = np.asarray(response.pos, dtype=np.float64).reshape(-1)
        if values.size < 6 or not np.all(np.isfinite(values[:6])):
            raise RuntimeError(f"Invalid flange posx: {values.tolist()}")
        posx = values[:6]
        T_base_flange = doosan_posx_to_transform(posx)
        return RobotPoseSample(
            T_base_flange_mm=T_base_flange,
            flange_posx_mm_deg=posx,
            rotation_source="get_current_tool_flange_posx",
            requested_monotonic_ns=requested_ns,
            received_monotonic_ns=time.monotonic_ns(),
        )

    def stable_sample(
        self,
        count: int = 5,
        interval_s: float = 0.08,
        max_translation_spread_mm: float = 0.3,
        max_rotation_spread_deg: float = 0.2,
    ) -> RobotPoseSample:
        samples: list[RobotPoseSample] = []
        for index in range(max(1, count)):
            if index:
                time.sleep(interval_s)
            samples.append(self.sample())
        translation_spread = 0.0
        rotation_spread = 0.0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                t_error, r_error = transform_error(
                    samples[i].T_base_flange_mm,
                    samples[j].T_base_flange_mm,
                )
                translation_spread = max(translation_spread, t_error)
                rotation_spread = max(rotation_spread, r_error)
        if (
            translation_spread > max_translation_spread_mm
            or rotation_spread > max_rotation_spread_deg
        ):
            raise RuntimeError(
                "Robot is not stationary: "
                f"translation spread={translation_spread:.3f} mm, "
                f"rotation spread={rotation_spread:.3f} deg"
            )
        last = samples[-1]
        return RobotPoseSample(
            T_base_flange_mm=last.T_base_flange_mm,
            flange_posx_mm_deg=last.flange_posx_mm_deg,
            rotation_source=last.rotation_source,
            requested_monotonic_ns=last.requested_monotonic_ns,
            received_monotonic_ns=last.received_monotonic_ns,
            stability_translation_mm=translation_spread,
            stability_rotation_deg=rotation_spread,
        )

    def close(self) -> None:
        node = getattr(self, "node", None)
        if node is not None:
            node.destroy_node()
            self.node = None
        if getattr(self, "owns_rclpy", False) and self.rclpy.ok():
            self.rclpy.shutdown()

    def __enter__(self) -> "DoosanFlangePoseProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


CAMERAS = ("left", "front", "right")
SIDES = ("left", "right")
POSE_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True)
class FramePacket:
    camera: str
    stamp_ns: int
    sequence: int
    color_bgr: np.ndarray
    depth_raw: np.ndarray | None = None
    depth_scale_mm: float = 1.0


@dataclass(frozen=True)
class Landmark2D:
    pixel: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class PoseSample:
    camera: str
    stamp_ns: int
    frame_sequence: int
    image_size: tuple[int, int]
    roi_xyxy: tuple[int, int, int, int]
    landmarks: Mapping[str, Landmark2D]
    inference_ms: float


@dataclass(frozen=True)
class TriangulationResult:
    point_base_mm: np.ndarray
    cameras: tuple[str, ...]
    camera_mask: int
    confidence: float
    reprojection_rms_px: float
    minimum_ray_angle_deg: float
    position_std_mm: float
    depth_used: bool = False
    depth_agreement_mm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_base_mm",
            np.asarray(self.point_base_mm, dtype=np.float64).reshape(3),
        )


@dataclass(frozen=True)
class FistState:
    side: str
    stamp_ns: int
    position_base_mm: np.ndarray
    velocity_base_mm_s: np.ndarray
    position_std_mm: float
    measurement_age_ms: float
    reprojection_error_px: float
    confidence: float
    camera_count: int
    camera_mask: int
    minimum_ray_angle_deg: float
    depth_used: bool
    valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_base_mm",
            np.asarray(self.position_base_mm, dtype=np.float64).reshape(3),
        )
        object.__setattr__(
            self,
            "velocity_base_mm_s",
            np.asarray(self.velocity_base_mm_s, dtype=np.float64).reshape(3),
        )


@dataclass(frozen=True)
class ImpactEvent:
    impact_id: int
    stamp_ns: int
    detected_stamp_ns: int
    side: str
    source: str
    wrist_pixel_front: tuple[float, float]
    confidence: float
    peak_speed_body_s: float
    displacement_body: float
    metadata: Mapping[str, float | int | str | bool] = field(default_factory=dict)

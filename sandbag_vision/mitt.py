from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class MittTrack:
    stamp_ns: int
    roi_normalized: tuple[float, float, float, float] | None
    state: str
    processing_ms: float
    contour_area_px: float


class RedMittTracker:
    """Low-latency red-pad tracker with progressive lost-target reacquisition."""

    def __init__(self, config: Mapping[str, object], initial_roi: object) -> None:
        self.config = dict(config)
        self.initial_roi = (
            np.asarray(initial_roi, dtype=np.float64).reshape(4)
            if isinstance(initial_roi, (list, tuple)) and len(initial_roi) == 4
            else None
        )
        self._bbox: np.ndarray | None = None
        self._last_good_bbox: np.ndarray | None = None
        self._last_good_area = 0.0
        self._velocity = np.zeros(2, dtype=np.float64)
        self._misses = 0
        self.latest = MittTrack(0, None, "STARTING", 0.0, 0.0)
        self._open_kernel = np.ones((3, 3), dtype=np.uint8)
        self._close_kernel = np.ones((5, 5), dtype=np.uint8)

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return 0.5 * (bbox[:2] + bbox[2:])

    def _seed_bbox(self, width: int, height: int) -> np.ndarray | None:
        if self.initial_roi is None:
            return None
        return self.initial_roi * np.asarray([width, height, width, height], dtype=np.float64)

    def _normalized_padded(
        self,
        bbox: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        padding = float(self.config.get("padding_px", 6.0))
        padded = bbox + np.asarray([-padding, -padding, padding, padding], dtype=np.float64)
        padded[[0, 2]] = np.clip(padded[[0, 2]], 0.0, float(width - 1))
        padded[[1, 3]] = np.clip(padded[[1, 3]], 0.0, float(height - 1))
        normalized = padded / np.asarray([width, height, width, height], dtype=np.float64)
        return tuple(float(value) for value in normalized)

    def update(self, stamp_ns: int, image_bgr: np.ndarray) -> MittTrack:
        started = time.perf_counter()
        height, width = image_bgr.shape[:2]
        if not self.enabled:
            roi = tuple(float(value) for value in self.initial_roi) if self.initial_roi is not None else None
            self.latest = MittTrack(int(stamp_ns), roi, "STATIC", 0.0, 0.0)
            return self.latest

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = int(self.config.get("saturation_min", 150))
        value = int(self.config.get("value_min", 80))
        low_hue_max = int(self.config.get("low_hue_max", 10))
        high_hue_min = int(self.config.get("high_hue_min", 170))
        mask = cv2.inRange(hsv, (0, saturation, value), (low_hue_max, 255, 255))
        mask |= cv2.inRange(hsv, (high_hue_min, saturation, value), (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        reference = self._bbox
        if reference is None:
            reference = self._last_good_bbox
        if reference is None:
            reference = self._seed_bbox(width, height)
        reference_center = self._center(reference) if reference is not None else None
        if self._bbox is not None and reference_center is not None:
            reference_center += self._velocity
        minimum_area = float(self.config.get("minimum_area_px", 1200.0))
        maximum_area = float(self.config.get("maximum_area_px", 20000.0))
        maximum_jump = float(self.config.get("maximum_center_jump_px", 120.0))
        jump_growth = max(0.0, float(self.config.get("reacquire_jump_growth", 1.0)))
        maximum_reacquire_jump = max(
            maximum_jump,
            float(self.config.get("maximum_reacquire_jump_px", 520.0)),
        )
        allowed_jump = min(
            maximum_reacquire_jump,
            maximum_jump * (1.0 + jump_growth * self._misses),
        )
        global_after = max(1, int(self.config.get("global_reacquire_after_misses", 3)))
        global_search = self._misses >= global_after
        minimum_fill_ratio = float(np.clip(self.config.get("minimum_fill_ratio", 0.40), 0.0, 1.0))
        area_similarity_weight = max(0.0, float(self.config.get("area_similarity_weight", 60.0)))
        candidates: list[tuple[float, float, float, np.ndarray]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not minimum_area <= area <= maximum_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 8 or box_height < 8:
                continue
            fill_ratio = area / max(float(box_width * box_height), 1.0)
            if fill_ratio < minimum_fill_ratio:
                continue
            bbox = np.asarray([x, y, x + box_width, y + box_height], dtype=np.float64)
            distance = float(np.linalg.norm(self._center(bbox) - reference_center)) if reference_center is not None else 0.0
            if reference_center is not None and not global_search and distance > allowed_jump:
                continue
            area_delta = (
                abs(float(np.log((area + 1.0) / (self._last_good_area + 1.0))))
                if self._last_good_area > 0.0
                else 0.0
            )
            score = distance + area_similarity_weight * area_delta
            candidates.append((score, -fill_ratio, -area, bbox))

        contour_area = 0.0
        if candidates:
            _, _, negative_area, measured = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            contour_area = -negative_area
            old_center = self._center(self._bbox) if self._bbox is not None else self._center(measured)
            if self._bbox is None:
                self._bbox = measured
            else:
                predicted = self._bbox + np.asarray(
                    [self._velocity[0], self._velocity[1], self._velocity[0], self._velocity[1]],
                    dtype=np.float64,
                )
                alpha = float(np.clip(self.config.get("smoothing_alpha", 0.80), 0.0, 1.0))
                self._bbox = alpha * measured + (1.0 - alpha) * predicted
            measured_velocity = self._center(self._bbox) - old_center
            velocity_alpha = float(np.clip(self.config.get("velocity_alpha", 0.50), 0.0, 1.0))
            self._velocity = velocity_alpha * measured_velocity + (1.0 - velocity_alpha) * self._velocity
            self._last_good_bbox = self._bbox.copy()
            self._last_good_area = contour_area
            self._misses = 0
            state = "TRACKED"
        else:
            self._misses += 1
            hold_frames = int(self.config.get("lost_hold_frames", 2))
            if self._bbox is not None and self._misses <= hold_frames:
                delta = np.asarray(
                    [self._velocity[0], self._velocity[1], self._velocity[0], self._velocity[1]],
                    dtype=np.float64,
                )
                self._bbox += delta
                state = "PREDICTED"
            else:
                self._bbox = None
                self._velocity.fill(0.0)
                state = "LOST"

        roi = self._normalized_padded(self._bbox, width, height) if self._bbox is not None else None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latest = MittTrack(int(stamp_ns), roi, state, elapsed_ms, contour_area)
        return self.latest

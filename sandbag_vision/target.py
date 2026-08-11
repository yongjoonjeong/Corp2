from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Mapping

import numpy as np

from .calibration import CameraModel, workspace_contains
from .pose import map_rotated_normalized_to_original, rotate_image
from .types import FramePacket


@dataclass(frozen=True)
class TargetSnapshot:
    stamp_ns: int
    state: str
    local_track_id: int | None
    bbox_xyxy: tuple[int, int, int, int] | None
    torso_base_mm: np.ndarray | None
    confidence: float
    people_count: int
    detected_people_count: int = 0
    aligned_to_front: bool = False
    alignment_distance_px: float | None = None


@dataclass(frozen=True)
class TargetCandidate:
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    anchor_pixel: tuple[float, float]
    point_base_mm: np.ndarray | None = None


def _point_to_bbox_distance(
    point: tuple[float, float],
    bbox: tuple[int, int, int, int],
) -> float:
    """Distance to a box, with a small center penalty to break overlaps."""
    x, y = point
    x1, y1, x2, y2 = bbox
    outside_x = max(float(x1) - x, 0.0, x - float(x2))
    outside_y = max(float(y1) - y, 0.0, y - float(y2))
    outside = float(np.hypot(outside_x, outside_y))
    center = float(np.hypot(x - (x1 + x2) * 0.5, y - (y1 + y2) * 0.5))
    return outside + 0.08 * center


def select_side_candidate(
    candidates: list[TargetCandidate],
    preferred_pixel: tuple[float, float] | None,
    locked_id: int | None,
    last_bbox: tuple[int, int, int, int] | None,
    maximum_alignment_distance_px: float,
) -> tuple[TargetCandidate | None, bool, float | None]:
    """Prefer the person selected by front-camera alignment, then local continuity."""
    if not candidates:
        return None, False, None
    if preferred_pixel is not None and np.all(np.isfinite(preferred_pixel)):
        ranked = [
            (_point_to_bbox_distance(preferred_pixel, candidate.bbox_xyxy), candidate)
            for candidate in candidates
        ]
        distance, candidate = min(ranked, key=lambda item: item[0])
        if distance <= maximum_alignment_distance_px:
            return candidate, True, distance
    selected = next((candidate for candidate in candidates if candidate.track_id == locked_id), None)
    if selected is not None:
        return selected, False, None
    if last_bbox is not None:
        last_center = ((last_bbox[0] + last_bbox[2]) * 0.5, (last_bbox[1] + last_bbox[3]) * 0.5)
        selected = min(
            candidates,
            key=lambda candidate: np.hypot(
                candidate.anchor_pixel[0] - last_center[0],
                candidate.anchor_pixel[1] - last_center[1],
            ),
        )
        return selected, False, None
    return max(candidates, key=lambda candidate: candidate.confidence), False, None


def _bbox_to_raw_pixels(
    raw_box: np.ndarray,
    raw_width: int,
    raw_height: int,
    rotation: str,
) -> tuple[int, int, int, int]:
    mode = str(rotation).strip().lower()
    if mode in ("none", "0", ""):
        rotated_width, rotated_height = raw_width, raw_height
    elif mode in (
        "counterclockwise_90",
        "ccw90",
        "90_ccw",
        "clockwise_90",
        "cw90",
        "90_cw",
    ):
        rotated_width, rotated_height = raw_height, raw_width
    elif mode in ("180", "rotate_180"):
        rotated_width, rotated_height = raw_width, raw_height
    else:
        raise ValueError(f"unsupported camera rotation: {rotation}")
    x1, y1, x2, y2 = (float(value) for value in raw_box)
    corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    mapped = [
        map_rotated_normalized_to_original(
            np.clip(x / max(rotated_width - 1, 1), 0.0, 1.0),
            np.clip(y / max(rotated_height - 1, 1), 0.0, 1.0),
            raw_width,
            raw_height,
            mode,
        )
        for x, y in corners
    ]
    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]
    return (
        int(np.clip(np.floor(min(xs)), 0, raw_width - 1)),
        int(np.clip(np.floor(min(ys)), 0, raw_height - 1)),
        int(np.clip(np.ceil(max(xs)), 1, raw_width)),
        int(np.clip(np.ceil(max(ys)), 1, raw_height)),
    )


def _median_depth_mm(packet: FramePacket, pixel: tuple[float, float], radius: int) -> float | None:
    if packet.depth_raw is None:
        return None
    x = int(round(pixel[0]))
    y = int(round(pixel[1]))
    height, width = packet.depth_raw.shape[:2]
    patch = packet.depth_raw[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    values = patch[patch > 0]
    if not values.size:
        return None
    return float(np.median(values)) * float(packet.depth_scale_mm)


def resolve_device(requested: str) -> str:
    requested = str(requested).strip()
    if requested.lower() != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class LowRateBoxerWorker:
    """Per-camera YOLO+BoT-SORT identity lock; never used for fist pixels."""

    def __init__(
        self,
        model_path: str | Path,
        front_camera: CameraModel,
        target_config: Mapping[str, object],
        workspace: Mapping[str, tuple[float, float] | list[float]],
        camera_name: str = "front",
        rotation: str = "none",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.camera_model = front_camera
        self.camera_name = str(camera_name)
        self.rotation = str(rotation)
        self.config = dict(target_config)
        self.workspace = workspace
        self.period_ns = int(1e9 / max(float(self.config.get("detector_hz", 4.0)), 0.5))
        self._condition = threading.Condition()
        self._pending: tuple[FramePacket, np.ndarray | None] | None = None
        self._latest: TargetSnapshot | None = None
        self._latest_lock = threading.Lock()
        self._running = True
        self._last_submit_ns = 0
        self.error: str | None = None
        self.device = resolve_device(str(self.config.get("device", "auto")))
        self.inference_ms = 0.0
        self._thread = threading.Thread(target=self._run, name=f"boxer-target-{self.camera_name}", daemon=True)
        self._thread.start()

    def due(self, stamp_ns: int) -> bool:
        return int(stamp_ns) - self._last_submit_ns >= self.period_ns

    def submit(
        self,
        packet: FramePacket,
        preferred_torso_base_mm: np.ndarray | None = None,
    ) -> bool:
        if not self.due(packet.stamp_ns):
            return False
        with self._condition:
            preferred = (
                None
                if preferred_torso_base_mm is None
                else np.asarray(preferred_torso_base_mm, dtype=np.float64).reshape(3).copy()
            )
            self._pending = (packet, preferred)
            self._last_submit_ns = packet.stamp_ns
            self._condition.notify()
        return True

    def latest(self) -> TargetSnapshot | None:
        with self._latest_lock:
            return self._latest

    def _run(self) -> None:
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            from ultralytics import YOLO

            try:
                import torch

                torch.set_num_threads(max(1, int(self.config.get("torch_threads", 2))))
            except Exception:
                pass

            model = YOLO(str(self.model_path))
            locked_id: int | None = None
            last_seen_ns = 0
            last_bbox: tuple[int, int, int, int] | None = None
            last_base: np.ndarray | None = None
            last_aligned_to_front = False
            lost_hold_ns = int(float(self.config.get("lost_hold_s", 1.0)) * 1e9)
            reacquire_mm = float(self.config.get("reacquire_distance_mm", 500.0))
            workspace_center = np.asarray(
                [np.mean(self.workspace[axis]) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or not self._running)
                    if not self._running:
                        return
                    packet, preferred_torso_base_mm = self._pending
                    self._pending = None
                detector_image = rotate_image(packet.color_bgr, self.rotation)
                result = model.track(
                    detector_image,
                    persist=True,
                    tracker=str(self.config.get("tracker", "botsort.yaml")),
                    classes=[0],
                    conf=float(self.config.get("confidence", 0.30)),
                    imgsz=int(self.config.get("imgsz", 416)),
                    device=self.device,
                    verbose=False,
                    max_det=8,
                )[0]
                self.inference_ms = float(result.speed.get("inference", 0.0))
                boxes = result.boxes
                detected_people_count = int(len(boxes)) if boxes is not None else 0
                candidates: list[TargetCandidate] = []
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confidence = boxes.conf.detach().cpu().numpy()
                    ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else np.arange(1, len(xyxy) + 1)
                    for track_id, raw_box, score in zip(ids, xyxy, confidence):
                        x1, y1, x2, y2 = _bbox_to_raw_pixels(
                            raw_box,
                            packet.color_bgr.shape[1],
                            packet.color_bgr.shape[0],
                            self.rotation,
                        )
                        torso_pixel = (float((x1 + x2) * 0.5), float(y1 + 0.43 * (y2 - y1)))
                        point_base = None
                        if self.camera_name == "front":
                            depth_mm = _median_depth_mm(packet, torso_pixel, int(self.config.get("depth_radius_px", 5)))
                            if depth_mm is None or not float(self.config.get("depth_min_mm", 500.0)) <= depth_mm <= float(self.config.get("depth_max_mm", 4000.0)):
                                continue
                            point_base = self.camera_model.depth_pixel_to_base(torso_pixel, depth_mm)
                            if not workspace_contains(point_base, self.workspace):
                                continue
                        candidates.append(
                            TargetCandidate(
                                track_id=int(track_id),
                                bbox_xyxy=(x1, y1, x2, y2),
                                confidence=float(score),
                                anchor_pixel=torso_pixel,
                                point_base_mm=point_base,
                            )
                        )
                aligned_to_front = False
                alignment_distance_px = None
                if self.camera_name == "front":
                    selected = next((item for item in candidates if item.track_id == locked_id), None)
                    if selected is None and candidates:
                        if last_base is not None:
                            nearest = min(
                                candidates,
                                key=lambda item: np.linalg.norm(item.point_base_mm - last_base),
                            )
                            if np.linalg.norm(nearest.point_base_mm - last_base) <= reacquire_mm:
                                selected = nearest
                        if selected is None:
                            selected = min(
                                candidates,
                                key=lambda item: np.linalg.norm(item.point_base_mm - workspace_center) - 80.0 * item.confidence,
                            )
                else:
                    preferred_pixel = None
                    if preferred_torso_base_mm is not None:
                        projected = self.camera_model.project_base(preferred_torso_base_mm)
                        if np.all(np.isfinite(projected)):
                            preferred_pixel = (float(projected[0]), float(projected[1]))
                    selected, aligned_to_front, alignment_distance_px = select_side_candidate(
                        candidates,
                        preferred_pixel,
                        locked_id,
                        last_bbox,
                        float(self.config.get("side_alignment_max_distance_px", 180.0)),
                    )
                if selected is not None:
                    locked_id = selected.track_id
                    last_bbox = selected.bbox_xyxy
                    score = selected.confidence
                    if selected.point_base_mm is not None:
                        last_base = selected.point_base_mm
                    last_aligned_to_front = aligned_to_front
                    last_seen_ns = packet.stamp_ns
                    snapshot = TargetSnapshot(
                        packet.stamp_ns,
                        "LOCKED",
                        locked_id,
                        last_bbox,
                        None if last_base is None else last_base.copy(),
                        score,
                        len(candidates),
                        detected_people_count,
                        aligned_to_front,
                        alignment_distance_px,
                    )
                elif last_bbox is not None and packet.stamp_ns - last_seen_ns <= lost_hold_ns:
                    snapshot = TargetSnapshot(
                        packet.stamp_ns,
                        "TEMPORARILY_LOST",
                        locked_id,
                        last_bbox,
                        None if last_base is None else last_base.copy(),
                        0.0,
                        len(candidates),
                        detected_people_count,
                        last_aligned_to_front,
                        None,
                    )
                else:
                    locked_id = None
                    last_bbox = None
                    last_aligned_to_front = False
                    snapshot = TargetSnapshot(
                        packet.stamp_ns,
                        "SEARCHING",
                        None,
                        None,
                        None,
                        0.0,
                        len(candidates),
                        detected_people_count,
                    )
                with self._latest_lock:
                    self._latest = snapshot
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            with self._latest_lock:
                self._latest = TargetSnapshot(
                    stamp_ns=time.time_ns(),
                    state="ERROR",
                    local_track_id=None,
                    bbox_xyxy=None,
                    torso_base_mm=None,
                    confidence=0.0,
                    people_count=0,
                )

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .capture import FrameRingBuffer
from .pose import draw_pose, rotate_image
from .triangulation import PoseHistory
from .types import CAMERAS, FramePacket, ImpactEvent, PoseSample


@dataclass(frozen=True)
class SnapshotResult:
    directory: Path
    triptych_bgr: np.ndarray
    jpeg_bytes: bytes
    metadata: dict


def _pose_score(sample: PoseSample | None) -> float:
    if sample is None:
        return 0.0
    required = (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    )
    values = [sample.landmarks[name].confidence for name in required if name in sample.landmarks]
    return float(np.mean(values)) if len(values) >= 6 else 0.0


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class ImpactSnapshotWriter:
    def __init__(
        self,
        root: str | Path,
        jpeg_quality: int,
        offsets_ms: list[float],
        rotations: Mapping[str, str] | None = None,
        mitt_roi_normalized: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.jpeg_quality = int(np.clip(jpeg_quality, 50, 100))
        self.offsets_ns = [int(float(value) * 1e6) for value in offsets_ms]
        self.rotations = dict(rotations or {})
        self.mitt_roi_normalized = (
            tuple(float(value) for value in mitt_roi_normalized)
            if isinstance(mitt_roi_normalized, (list, tuple)) and len(mitt_roi_normalized) == 4
            else None
        )

    def _select(
        self,
        event: ImpactEvent,
        ring: FrameRingBuffer,
        history: PoseHistory,
    ) -> tuple[FramePacket | None, PoseSample | None]:
        candidates = ring.around(event.stamp_ns, self.offsets_ns)
        if not candidates:
            return None, None
        ranked: list[tuple[float, int, FramePacket, PoseSample | None]] = []
        for packet in candidates:
            pose = history.nearest(packet.stamp_ns, maximum_skew_ns=70_000_000)
            time_error_ms = abs(packet.stamp_ns - event.stamp_ns) / 1e6
            ranked.append((_pose_score(pose) - 0.004 * time_error_ms, -int(time_error_ms), packet, pose))
        _, _, packet, pose = max(ranked, key=lambda item: (item[0], item[1]))
        return packet, pose

    def save(
        self,
        event: ImpactEvent,
        rings: Mapping[str, FrameRingBuffer],
        histories: Mapping[str, PoseHistory],
    ) -> SnapshotResult | None:
        selected = {camera: self._select(event, rings[camera], histories[camera]) for camera in CAMERAS}
        if any(packet is None for packet, _ in selected.values()):
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = self.root / f"impact_{event.impact_id:05d}_{timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        annotated: dict[str, np.ndarray] = {}
        camera_metadata: dict[str, dict[str, float | int]] = {}
        for camera in CAMERAS:
            packet, pose = selected[camera]
            assert packet is not None
            raw = packet.color_bgr.copy()
            displayed = draw_pose(raw, pose)
            if camera == "front":
                dynamic_roi_values = tuple(
                    event.metadata.get(name)
                    for name in ("mitt_roi_x1", "mitt_roi_y1", "mitt_roi_x2", "mitt_roi_y2")
                )
                dynamic_roi = (
                    tuple(float(value) for value in dynamic_roi_values)
                    if all(isinstance(value, (float, int)) and float(value) >= 0.0 for value in dynamic_roi_values)
                    else None
                )
                display_roi = dynamic_roi or self.mitt_roi_normalized
                if display_roi is not None:
                    height, width = displayed.shape[:2]
                    x1, y1, x2, y2 = (
                        int(round(display_roi[0] * width)),
                        int(round(display_roi[1] * height)),
                        int(round(display_roi[2] * width)),
                        int(round(display_roi[3] * height)),
                    )
                    cv2.rectangle(displayed, (x1, y1), (x2, y2), (80, 255, 80), 2)
                contact_x = event.metadata.get("contact_pixel_front_x")
                contact_y = event.metadata.get("contact_pixel_front_y")
                if isinstance(contact_x, (float, int)) and isinstance(contact_y, (float, int)):
                    contact = (int(round(float(contact_x))), int(round(float(contact_y))))
                    cv2.drawMarker(
                        displayed,
                        contact,
                        (0, 255, 255),
                        cv2.MARKER_CROSS,
                        22,
                        2,
                        cv2.LINE_AA,
                    )
            annotated[camera] = rotate_image(
                displayed,
                self.rotations.get(camera, "none"),
            )
            cv2.imwrite(str(directory / f"{camera}_raw.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            cv2.imwrite(str(directory / f"{camera}.jpg"), annotated[camera], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            camera_metadata[camera] = {
                "capture_stamp_ns": packet.stamp_ns,
                "impact_offset_ms": round((packet.stamp_ns - event.stamp_ns) / 1e6, 3),
                "pose_confidence": round(_pose_score(pose), 4),
            }
        tile_width = 640
        tile_height = 480
        tiles = []
        for camera in CAMERAS:
            tile = _letterbox(annotated[camera], tile_width, tile_height)
            cv2.rectangle(tile, (0, 0), (tile_width, 36), (0, 0, 0), -1)
            cv2.putText(tile, camera.upper(), (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        triptych = np.hstack(tiles)
        cv2.imwrite(str(directory / "impact_triptych.jpg"), triptych, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        metadata = {
            "impact_id": event.impact_id,
            "source": event.source,
            "side": event.side,
            "impact_stamp_ns": event.stamp_ns,
            "detected_stamp_ns": event.detected_stamp_ns,
            "confidence": round(event.confidence, 4),
            "peak_speed_body_s": round(event.peak_speed_body_s, 4),
            "displacement_body": round(event.displacement_body, 4),
            "wrist_pixel_front": list(event.wrist_pixel_front),
            "cameras": camera_metadata,
            **dict(event.metadata),
        }
        with (directory / "impact_metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        ok, encoded = cv2.imencode(".jpg", triptych, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        return SnapshotResult(directory, triptych, encoded.tobytes(), metadata)

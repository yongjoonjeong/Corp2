import json

import numpy as np

from sandbag_vision.capture import FrameRingBuffer
from sandbag_vision.snapshot import ImpactSnapshotWriter
from sandbag_vision.triangulation import PoseHistory
from sandbag_vision.types import CAMERAS, FramePacket, ImpactEvent, Landmark2D, PoseSample


def test_snapshot_writes_three_raw_three_annotated_triptych_and_json(tmp_path) -> None:
    stamp = 2_000_000_000
    rings = {camera: FrameRingBuffer(4) for camera in CAMERAS}
    histories = {camera: PoseHistory() for camera in CAMERAS}
    landmarks = {
        "left_shoulder": Landmark2D((200.0, 180.0), 0.9),
        "right_shoulder": Landmark2D((300.0, 180.0), 0.9),
        "left_elbow": Landmark2D((180.0, 250.0), 0.9),
        "right_elbow": Landmark2D((320.0, 250.0), 0.9),
        "left_wrist": Landmark2D((150.0, 300.0), 0.9),
        "right_wrist": Landmark2D((350.0, 300.0), 0.9),
        "left_hip": Landmark2D((220.0, 380.0), 0.9),
        "right_hip": Landmark2D((280.0, 380.0), 0.9),
    }
    for sequence, camera in enumerate(CAMERAS, 1):
        image = np.full((480, 640, 3), sequence * 30, dtype=np.uint8)
        rings[camera].append(FramePacket(camera, stamp, sequence, image))
        histories[camera].add(
            PoseSample(camera, stamp, sequence, (640, 480), (0, 0, 640, 480), landmarks, 2.0)
        )
    event = ImpactEvent(1, stamp, stamp + 60_000_000, "right", "vision_candidate", (350.0, 300.0), 0.8, 2.0, 0.5)
    result = ImpactSnapshotWriter(tmp_path, 90, [-34.0, 0.0, 34.0]).save(event, rings, histories)
    assert result is not None
    expected = {
        "left_raw.jpg",
        "front_raw.jpg",
        "right_raw.jpg",
        "left.jpg",
        "front.jpg",
        "right.jpg",
        "impact_triptych.jpg",
        "impact_metadata.json",
    }
    assert expected <= {path.name for path in result.directory.iterdir()}
    metadata = json.loads((result.directory / "impact_metadata.json").read_text(encoding="utf-8"))
    assert metadata["impact_id"] == 1
    assert metadata["cameras"]["front"]["impact_offset_ms"] == 0.0
    assert len(result.jpeg_bytes) > 1000

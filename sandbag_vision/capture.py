from __future__ import annotations

from collections import deque
from pathlib import Path
import re
import threading
import time
from typing import Iterable

import cv2
import numpy as np
import yaml

from .types import CAMERAS, FramePacket


class FrameRingBuffer:
    def __init__(self, capacity: int) -> None:
        self._items: deque[FramePacket] = deque(maxlen=max(2, int(capacity)))
        self._lock = threading.Lock()

    def append(self, packet: FramePacket) -> None:
        with self._lock:
            self._items.append(packet)

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def nearest(self, stamp_ns: int, maximum_delta_ns: int | None = None) -> FramePacket | None:
        with self._lock:
            if not self._items:
                return None
            packet = min(self._items, key=lambda item: abs(item.stamp_ns - int(stamp_ns)))
        if maximum_delta_ns is not None and abs(packet.stamp_ns - int(stamp_ns)) > maximum_delta_ns:
            return None
        return packet

    def around(self, stamp_ns: int, offsets_ns: Iterable[int]) -> tuple[FramePacket, ...]:
        selected: list[FramePacket] = []
        seen: set[int] = set()
        for offset in offsets_ns:
            packet = self.nearest(int(stamp_ns) + int(offset))
            if packet is not None and packet.sequence not in seen:
                selected.append(packet)
                seen.add(packet.sequence)
        return tuple(selected)


def _opencv_source(device: str) -> int | str:
    path = Path(str(device)).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        resolved = path
    match = re.fullmatch(r"/dev/video(\d+)", str(resolved))
    return int(match.group(1)) if match else str(resolved)


def discover_c270_devices(sysfs_root: str | Path = "/sys/class/video4linux") -> list[str]:
    discovered: list[tuple[str, str]] = []
    for entry in Path(sysfs_root).glob("video*"):
        try:
            name = (entry / "name").read_text(encoding="utf-8").strip().lower()
            index = int((entry / "index").read_text(encoding="utf-8").strip())
            physical = str((entry / "device").resolve())
        except (OSError, ValueError):
            continue
        if "c270" in name and index == 0 and Path("/dev", entry.name).exists():
            discovered.append((physical, str(Path("/dev", entry.name))))
    return [device for _, device in sorted(discovered)]


def resolve_usb_roles(camera_config: dict, config_directory: Path) -> tuple[str, str]:
    left = str(camera_config.get("left_device", "auto")).strip()
    right = str(camera_config.get("right_device", "auto")).strip()
    if left.lower() == "auto" and right.lower() == "auto":
        role_path = config_directory / str(camera_config.get("role_map", "camera_roles.yaml"))
        if role_path.is_file():
            with role_path.open(encoding="utf-8") as stream:
                role_data = yaml.safe_load(stream) or {}
            roles = role_data.get("c270_roles", {})
            mapped = (str(roles.get("left", "")).strip(), str(roles.get("right", "")).strip())
            if all(mapped) and all(Path(device).exists() for device in mapped):
                left, right = mapped
            else:
                devices = discover_c270_devices()
                if len(devices) != 2:
                    raise RuntimeError(f"C270 cameras must be exactly two, found {devices}")
                left, right = devices
        else:
            devices = discover_c270_devices()
            if len(devices) != 2:
                raise RuntimeError(f"C270 cameras must be exactly two, found {devices}")
            left, right = devices
    elif left.lower() == "auto" or right.lower() == "auto":
        raise ValueError("left_device and right_device must both be auto or both explicit")
    if bool(camera_config.get("swap_webcams", False)):
        left, right = right, left
    return left, right


class _CaptureWorker:
    def __init__(self, name: str, ring: FrameRingBuffer) -> None:
        self.name = name
        self.ring = ring
        self._running = threading.Event()
        self._running.set()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.frames = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_guarded, name=f"capture-{self.name}", daemon=True)
        self._thread.start()

    def _run_guarded(self) -> None:
        try:
            self._capture_loop()
        except Exception as error:  # surfaced through status instead of silently dying
            self.error = f"{type(error).__name__}: {error}"
            self._running.clear()

    def _capture_loop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.5)


class UsbCaptureWorker(_CaptureWorker):
    def __init__(self, name: str, device: str, width: int, height: int, fps: int, ring: FrameRingBuffer, time_offset_ns: int = 0) -> None:
        super().__init__(name, ring)
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.time_offset_ns = int(time_offset_ns)

    def _capture_loop(self) -> None:
        source = _opencv_source(self.device)
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {self.name} camera: {self.device}")
        sequence = 0
        try:
            while self._running.is_set():
                before_ns = time.time_ns()
                ok = capture.grab()
                if not ok:
                    time.sleep(0.005)
                    continue
                ok, frame = capture.retrieve()
                after_ns = time.time_ns()
                if not ok or frame is None:
                    continue
                sequence += 1
                frame.setflags(write=False)
                self.ring.append(
                    FramePacket(
                        camera=self.name,
                        stamp_ns=(before_ns + after_ns) // 2 + self.time_offset_ns,
                        sequence=sequence,
                        color_bgr=frame,
                    )
                )
                self.frames = sequence
        finally:
            capture.release()


class RealSenseCaptureWorker(_CaptureWorker):
    def __init__(self, serial: str, width: int, height: int, fps: int, ring: FrameRingBuffer, time_offset_ns: int = 0) -> None:
        super().__init__("front", ring)
        self.serial = str(serial).strip()
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.time_offset_ns = int(time_offset_ns)

    def _capture_loop(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("pyrealsense2 is required for the front camera") from error
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        depth_scale_mm = float(profile.get_device().first_depth_sensor().get_depth_scale()) * 1000.0
        align = rs.align(rs.stream.color)
        sequence = 0
        try:
            while self._running.is_set():
                before_ns = time.time_ns()
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=250)
                except RuntimeError:
                    continue
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                after_ns = time.time_ns()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                color.setflags(write=False)
                depth.setflags(write=False)
                sequence += 1
                self.ring.append(
                    FramePacket(
                        camera="front",
                        stamp_ns=(before_ns + after_ns) // 2 + self.time_offset_ns,
                        sequence=sequence,
                        color_bgr=color,
                        depth_raw=depth,
                        depth_scale_mm=depth_scale_mm,
                    )
                )
                self.frames = sequence
        finally:
            pipeline.stop()


class CameraHub:
    def __init__(self, camera_config: dict, config_directory: Path) -> None:
        width = int(camera_config.get("width", 640))
        height = int(camera_config.get("height", 480))
        fps = int(camera_config.get("fps", 30))
        capacity = max(int(fps * float(camera_config.get("ring_buffer_s", 2.0))), 4)
        offsets = camera_config.get("time_offset_ms", {})
        offset_ns = {
            name: int(float(offsets.get(name, 0.0)) * 1e6) for name in CAMERAS
        }
        self.rings = {name: FrameRingBuffer(capacity) for name in CAMERAS}
        left, right = resolve_usb_roles(camera_config, config_directory)
        self.devices = {"left": left, "front": "realsense", "right": right}
        self.workers: dict[str, _CaptureWorker] = {
            "left": UsbCaptureWorker("left", left, width, height, fps, self.rings["left"], offset_ns["left"]),
            "front": RealSenseCaptureWorker(str(camera_config.get("realsense_serial", "")), width, height, fps, self.rings["front"], offset_ns["front"]),
            "right": UsbCaptureWorker("right", right, width, height, fps, self.rings["right"], offset_ns["right"]),
        }

    def start(self) -> None:
        for worker in self.workers.values():
            worker.start()

    def latest(self) -> dict[str, FramePacket | None]:
        return {name: ring.latest() for name, ring in self.rings.items()}

    def nearest(self, camera: str, stamp_ns: int, maximum_delta_ns: int | None = None) -> FramePacket | None:
        return self.rings[camera].nearest(stamp_ns, maximum_delta_ns)

    def status(self) -> dict[str, dict[str, int | str | None]]:
        return {
            name: {"frames": worker.frames, "error": worker.error, "device": self.devices[name]}
            for name, worker in self.workers.items()
        }

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()

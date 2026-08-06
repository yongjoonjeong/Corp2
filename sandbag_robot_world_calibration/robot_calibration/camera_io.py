from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import load_yaml
from .device_discovery import discover_c270_sources, format_device_report


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    timestamp_ns: int


@dataclass(frozen=True)
class CapturedRgbdFrame:
    """One synchronized RealSense RGB-D frame.

    color_image is BGR uint8, depth_image_mm is uint16 millimetres, and
    depth_colormap is a BGR preview aligned to the color image.
    """

    color_image: np.ndarray
    depth_image_mm: np.ndarray
    depth_colormap: np.ndarray
    timestamp_ns: int


class CameraSource:
    name: str

    def read(self) -> CapturedFrame:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class UsbCameraSource(CameraSource):
    def __init__(
        self,
        name: str,
        device: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str = "MJPG",
    ) -> None:
        self.name = name
        self.device = device
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open USB camera {name}: {device}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):
            capture.grab()
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (actual_width, actual_height) != (width, height):
            capture.release()
            raise RuntimeError(
                f"{name} opened at {actual_width}x{actual_height}; expected {width}x{height}"
            )
        self.capture = capture
        self.actual_fps = actual_fps
        print(
            f"[CAMERA] {name}: {device}, {actual_width}x{actual_height}, "
            f"reported {actual_fps:.2f} FPS, FOURCC={fourcc}"
        )

    def read(self) -> CapturedFrame:
        ok, image = self.capture.read()
        timestamp_ns = time.monotonic_ns()
        if not ok or image is None:
            raise RuntimeError(f"Failed to read USB camera: {self.name}")
        return CapturedFrame(image=image, timestamp_ns=timestamp_ns)

    def close(self) -> None:
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()
            self.capture = None


class RealSenseColorSource(CameraSource):
    """RealSense color source with optional synchronized, color-aligned depth.

    Calibration scripts call read(), which always returns the RGB image only.
    Camera checking can request include_depth=True and use read_rgbd().
    """

    def __init__(
        self,
        name: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
        *,
        include_depth: bool = False,
        depth_width: int | None = None,
        depth_height: int | None = None,
        depth_fps: int | None = None,
        depth_preview_min_mm: int = 200,
        depth_preview_max_mm: int = 4000,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "pyrealsense2 is required for the RealSense camera"
            ) from error

        self.rs = rs
        self.name = name
        self.include_depth = bool(include_depth)
        self.depth_preview_min_mm = int(depth_preview_min_mm)
        self.depth_preview_max_mm = int(depth_preview_max_mm)
        if self.depth_preview_max_mm <= self.depth_preview_min_mm:
            raise ValueError("depth_preview_max_mm must be greater than depth_preview_min_mm")

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        if self.include_depth:
            dw = int(depth_width if depth_width is not None else width)
            dh = int(depth_height if depth_height is not None else height)
            dfps = int(depth_fps if depth_fps is not None else fps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
        else:
            dw, dh, dfps = width, height, fps

        profile = pipeline.start(config)
        for _ in range(20):
            pipeline.wait_for_frames(timeout_ms=3000)

        color_active = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_actual = (color_active.width(), color_active.height(), color_active.fps())
        if color_actual != (width, height, fps):
            pipeline.stop()
            raise RuntimeError(
                f"{name} color opened at {color_actual}; expected {(width, height, fps)}"
            )

        if self.include_depth:
            depth_active = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_actual = (depth_active.width(), depth_active.height(), depth_active.fps())
            if depth_actual != (dw, dh, dfps):
                pipeline.stop()
                raise RuntimeError(
                    f"{name} depth opened at {depth_actual}; expected {(dw, dh, dfps)}"
                )
            self.align = rs.align(rs.stream.color)
        else:
            depth_actual = None
            self.align = None

        device = profile.get_device()
        device_serial = device.get_info(rs.camera_info.serial_number)
        depth_sensor = device.first_depth_sensor() if self.include_depth else None
        self.depth_scale_m = float(depth_sensor.get_depth_scale()) if depth_sensor else 0.001
        self.pipeline = pipeline

        if self.include_depth:
            print(
                f"[CAMERA] {name}: RealSense serial={device_serial}, "
                f"RGB={color_actual[0]}x{color_actual[1]}@{color_actual[2]}, "
                f"DEPTH={depth_actual[0]}x{depth_actual[1]}@{depth_actual[2]}, "
                f"aligned_to=color"
            )
        else:
            print(
                f"[CAMERA] {name}: RealSense serial={device_serial}, "
                f"RGB={color_actual[0]}x{color_actual[1]}@{color_actual[2]}"
            )

    def _wait_frames(self) -> Any:
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        if self.include_depth:
            frames = self.align.process(frames)
        return frames

    def read(self) -> CapturedFrame:
        frames = self._wait_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"No RealSense color frame: {self.name}")
        return CapturedFrame(
            image=np.asanyarray(color.get_data()).copy(),
            timestamp_ns=time.monotonic_ns(),
        )

    def read_rgbd(self) -> CapturedRgbdFrame:
        if not self.include_depth:
            raise RuntimeError(
                f"RealSense depth was not enabled for camera '{self.name}'. "
                "Open it with include_depth=True."
            )

        frames = self._wait_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"No synchronized RealSense RGB-D frame: {self.name}")

        color_image = np.asanyarray(color.get_data()).copy()
        depth_raw = np.asanyarray(depth.get_data()).copy()

        # Convert the sensor's raw depth unit to millimetres. Most D4xx devices
        # use 0.001 m, but reading the actual scale avoids hard-coding it.
        depth_mm = np.rint(depth_raw.astype(np.float64) * self.depth_scale_m * 1000.0)
        depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        depth_colormap = make_depth_colormap(
            depth_mm,
            min_mm=self.depth_preview_min_mm,
            max_mm=self.depth_preview_max_mm,
        )

        return CapturedRgbdFrame(
            color_image=color_image,
            depth_image_mm=depth_mm,
            depth_colormap=depth_colormap,
            timestamp_ns=time.monotonic_ns(),
        )

    def close(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.stop()
            self.pipeline = None


def make_depth_colormap(
    depth_mm: np.ndarray,
    *,
    min_mm: int = 200,
    max_mm: int = 4000,
) -> np.ndarray:
    """Create a stable BGR depth preview with invalid pixels shown in black."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = depth > 0
    clipped = np.clip(depth, float(min_mm), float(max_mm))
    normalized = (clipped - float(min_mm)) / float(max_mm - min_mm)
    # Near = warm, far = cool. Invert before applying TURBO.
    preview_u8 = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def open_camera(
    config_path: str | Path,
    camera_name: str,
    *,
    include_depth: bool = False,
) -> CameraSource:
    config = load_yaml(config_path)
    width = int(config["image_width"])
    height = int(config["image_height"])
    fps = int(config["fps"])
    cameras = config.get("cameras", {})
    if camera_name not in cameras:
        raise KeyError(f"Camera '{camera_name}' not found in {config_path}")
    camera = cameras[camera_name]
    camera_type = str(camera["type"]).lower()

    if camera_type == "usb":
        if include_depth:
            raise ValueError(f"USB camera '{camera_name}' does not provide a depth stream")

        device_value = str(camera.get("device", "auto")).strip()
        if device_value.lower() == "auto":
            role_map_name = str(camera.get("role_map", "camera_roles.yaml"))
            role_map_path = Path(config_path).resolve().parent / role_map_name
            if not role_map_path.exists():
                candidates = discover_c270_sources(
                    product_contains=str(camera.get("product_contains", "C270"))
                )
                raise RuntimeError(
                    f"C270 role map not found: {role_map_path}\n"
                    "Run this once before calibration:\n"
                    "  python3 00_check_cameras.py --assign-c270\n\n"
                    "Detected C270 candidates:\n"
                    f"{format_device_report(candidates)}"
                )
            role_map = load_yaml(role_map_path)
            roles = role_map.get("c270_roles", {})
            device_value = str(roles.get(camera_name, "")).strip()
            if not device_value:
                raise RuntimeError(
                    f"Camera role '{camera_name}' is missing in {role_map_path}. "
                    "Run: python3 00_check_cameras.py --assign-c270"
                )

        if not Path(device_value).exists():
            raise RuntimeError(
                f"Configured {camera_name} camera path does not exist: {device_value}\n"
                "Reconnect both C270 cameras to their assigned USB ports or run:\n"
                "  python3 00_check_cameras.py --assign-c270"
            )

        return UsbCameraSource(
            name=camera_name,
            device=device_value,
            width=width,
            height=height,
            fps=fps,
            fourcc=str(camera.get("fourcc", "MJPG")),
        )

    if camera_type == "realsense":
        depth_config = camera.get("depth", {}) or {}
        return RealSenseColorSource(
            name=camera_name,
            serial=str(camera.get("serial", "")),
            width=width,
            height=height,
            fps=fps,
            include_depth=include_depth,
            depth_width=int(depth_config.get("width", width)),
            depth_height=int(depth_config.get("height", height)),
            depth_fps=int(depth_config.get("fps", fps)),
            depth_preview_min_mm=int(depth_config.get("preview_min_mm", 200)),
            depth_preview_max_mm=int(depth_config.get("preview_max_mm", 4000)),
        )

    raise ValueError(f"Unsupported camera type for {camera_name}: {camera_type}")

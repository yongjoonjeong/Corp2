from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VideoDeviceCandidate:
    """One stable V4L2 capture endpoint.

    stable_path should normally be a /dev/v4l/by-path/*-video-index0 symlink so
    reconnecting devices does not reshuffle the selected camera when /dev/videoN
    numbers change.
    """

    stable_path: str
    resolved_device: str
    product: str
    serial: str
    usb_path: str
    vendor_id: str
    model_id: str

    @property
    def display_name(self) -> str:
        product = self.product or "Unknown V4L2 camera"
        return f"{product} ({self.stable_path} -> {self.resolved_device})"


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def udev_properties(device: str) -> dict[str, str]:
    output = _run_text(["udevadm", "info", "--query=property", "--name", device])
    properties: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key.strip()] = value.strip()
    return properties


def _candidate_from_path(stable_path: str) -> VideoDeviceCandidate | None:
    try:
        resolved = os.path.realpath(stable_path)
    except OSError:
        return None
    if not resolved.startswith("/dev/video") or not Path(resolved).exists():
        return None

    props = udev_properties(resolved)
    capabilities = props.get("ID_V4L_CAPABILITIES", "")
    # A metadata endpoint is not a normal image capture node. The index0 symlink
    # normally points to the RGB capture endpoint; this capability check adds a
    # second guard when udev exposes it.
    if capabilities and ":capture:" not in capabilities:
        return None

    product = (
        props.get("ID_V4L_PRODUCT")
        or props.get("ID_MODEL_FROM_DATABASE")
        or props.get("ID_MODEL")
        or ""
    ).replace("_", " ")
    serial = props.get("ID_SERIAL_SHORT") or props.get("ID_SERIAL") or ""
    usb_path = props.get("ID_PATH") or stable_path
    return VideoDeviceCandidate(
        stable_path=stable_path,
        resolved_device=resolved,
        product=product,
        serial=serial,
        usb_path=usb_path,
        vendor_id=props.get("ID_VENDOR_ID", ""),
        model_id=props.get("ID_MODEL_ID", ""),
    )


def _deduplicate(candidates: Iterable[VideoDeviceCandidate]) -> list[VideoDeviceCandidate]:
    by_device: dict[str, VideoDeviceCandidate] = {}
    for candidate in candidates:
        # Prefer a stable by-path symlink over a raw /dev/videoN endpoint.
        previous = by_device.get(candidate.resolved_device)
        if previous is None or candidate.stable_path.startswith("/dev/v4l/by-path/"):
            by_device[candidate.resolved_device] = candidate
    return sorted(by_device.values(), key=lambda item: item.stable_path)


def enumerate_capture_devices() -> list[VideoDeviceCandidate]:
    """Enumerate image capture endpoints while avoiding RealSense IR metadata.

    We first use by-path video-index0 links because they are stable across boots
    and represent the first image endpoint of each physical USB camera. A raw
    /dev/video* fallback is included for systems where /dev/v4l/by-path is absent.
    """

    candidates: list[VideoDeviceCandidate] = []
    for stable_path in sorted(glob.glob("/dev/v4l/by-path/*-video-index0")):
        candidate = _candidate_from_path(stable_path)
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        return _deduplicate(candidates)

    for device in sorted(glob.glob("/dev/video*")):
        candidate = _candidate_from_path(device)
        if candidate is not None:
            candidates.append(candidate)
    return _deduplicate(candidates)


def is_c270_candidate(
    candidate: VideoDeviceCandidate,
    *,
    product_contains: str = "C270",
) -> bool:
    token = product_contains.strip().lower()
    searchable = " ".join(
        [
            candidate.product,
            candidate.serial,
            candidate.usb_path,
            candidate.vendor_id,
            candidate.model_id,
        ]
    ).lower()
    if token and token in searchable:
        return True

    # Logitech vendor ID. Common C270 model IDs include 0825 and 0826, but the
    # product-name match above remains the primary method.
    return candidate.vendor_id.lower() == "046d" and candidate.model_id.lower() in {
        "0825",
        "0826",
    }


def discover_c270_sources(product_contains: str = "C270") -> list[VideoDeviceCandidate]:
    return [
        candidate
        for candidate in enumerate_capture_devices()
        if is_c270_candidate(candidate, product_contains=product_contains)
    ]


def format_device_report(candidates: Iterable[VideoDeviceCandidate]) -> str:
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"[{index}] {candidate.product or 'Unknown camera'}",
                f"    stable : {candidate.stable_path}",
                f"    actual : {candidate.resolved_device}",
                f"    usb    : {candidate.usb_path}",
                f"    serial : {candidate.serial or '-'}",
                f"    vid:pid: {candidate.vendor_id or '-'}:{candidate.model_id or '-'}",
            ]
        )
    return "\n".join(lines) if lines else "No V4L2 capture devices found."

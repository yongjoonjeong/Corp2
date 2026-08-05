from __future__ import annotations

import re
from pathlib import Path

import yaml


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
        if part
    )


def discover_c270_capture_devices(
    sysfs_root: str | Path = "/sys/class/video4linux",
    dev_root: str | Path = "/dev",
) -> list[str]:
    """Return C270 index-0 video nodes ordered by physical USB path."""
    sysfs = Path(sysfs_root)
    devices = Path(dev_root)
    discovered: list[tuple[tuple[object, ...], str]] = []

    for entry in sysfs.glob("video*"):
        try:
            name = (entry / "name").read_text(encoding="utf-8").strip()
            stream_index = int(
                (entry / "index").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            continue
        if "c270" not in name.lower() or stream_index != 0:
            continue

        device_node = devices / entry.name
        if not device_node.exists():
            continue
        physical_path = str((entry / "device").resolve())
        discovered.append(
            (
                _natural_key(physical_path),
                str(device_node),
            )
        )

    discovered.sort(key=lambda item: item[0])
    return [device for _, device in discovered]


def resolve_stereo_webcam_devices(
    left_device: str,
    right_device: str,
    swap_webcams: bool = False,
    sysfs_root: str | Path = "/sys/class/video4linux",
    dev_root: str | Path = "/dev",
) -> tuple[str, str, bool]:
    """Resolve explicit paths or an auto/auto C270 pair.

    Returns ``(left, right, auto_detected)``.
    """
    left = left_device.strip()
    right = right_device.strip()
    auto_left = left.lower() == "auto"
    auto_right = right.lower() == "auto"
    if auto_left != auto_right:
        raise ValueError(
            "left_device와 right_device는 둘 다 auto이거나 둘 다 명시해야 합니다"
        )

    auto_detected = auto_left and auto_right
    if auto_detected:
        candidates = discover_c270_capture_devices(sysfs_root, dev_root)
        if len(candidates) != 2:
            raise RuntimeError(
                "C270 영상 캡처 장치가 정확히 2대 필요하지만 "
                f"{len(candidates)}대가 발견됐습니다: {candidates}"
            )
        left, right = candidates

    if swap_webcams:
        left, right = right, left
    return left, right, auto_detected


def load_camera_role_devices(path: str | Path) -> tuple[str, str]:
    """Load user-relative C270 roles written by robot-world calibration."""
    role_path = Path(path).expanduser().resolve()
    with role_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    roles = data.get("c270_roles", {}) if isinstance(data, dict) else {}
    left = str(roles.get("left", "")).strip()
    right = str(roles.get("right", "")).strip()
    if not left or not right:
        raise ValueError(f"Invalid C270 role map: {role_path}")
    missing = [device for device in (left, right) if not Path(device).exists()]
    if missing:
        raise RuntimeError(
            f"C270 role-map device paths do not exist: {missing}. "
            "Run 00_check_cameras.py --assign-c270 again."
        )
    return left, right

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sandbag_vision.capture import _opencv_source, discover_c270_devices  # noqa: E402


def stable_path(device: str) -> str:
    actual = Path(device).resolve()
    directory = Path("/dev/v4l/by-path")
    if directory.is_dir():
        for candidate in directory.glob("*-video-index0"):
            try:
                if candidate.resolve() == actual:
                    return str(candidate)
            except OSError:
                pass
    return str(actual)


def main() -> int:
    devices = discover_c270_devices()
    if len(devices) != 2:
        print(f"[ERROR] C270 두 대가 필요합니다: {devices}")
        return 1
    captures = [cv2.VideoCapture(_opencv_source(device), cv2.CAP_V4L2) for device in devices]
    for capture in captures:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("화면에서 복서 기준 LEFT 카메라를 확인하고 1 또는 2를 누르세요. Q는 취소입니다.")
    choice = None
    try:
        while choice is None:
            frames = []
            for index, capture in enumerate(captures, 1):
                ok, frame = capture.read()
                if not ok:
                    print(f"[ERROR] candidate {index} 프레임을 읽지 못했습니다")
                    return 1
                cv2.putText(frame, f"CANDIDATE {index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                frames.append(frame)
            cv2.imshow("Assign C270 roles", cv2.hconcat(frames))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("1"), ord("2")):
                choice = key - ord("1")
            elif key in (ord("q"), 27):
                return 1
    finally:
        for capture in captures:
            capture.release()
        cv2.destroyAllWindows()
    left = stable_path(devices[choice])
    right = stable_path(devices[1 - choice])
    role_path = PROJECT / "config" / "camera_roles.yaml"
    with role_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"schema_version": 1, "c270_roles": {"left": left, "right": right}},
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"[OK] left={left}")
    print(f"[OK] right={right}")
    print(f"saved: {role_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

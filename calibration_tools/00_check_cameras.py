#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_io import RealSenseColorSource, UsbCameraSource, open_camera
from robot_calibration.common import load_yaml, put_status_lines, save_yaml
from robot_calibration.device_discovery import (
    VideoDeviceCandidate,
    discover_c270_sources,
    format_device_report,
)


TILE_WIDTH = 480
TILE_HEIGHT = 360


def _fit_tile(image: np.ndarray, title: str, extra_lines: list[str] | None = None) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    src_h, src_w = source.shape[:2]
    scale = min(TILE_WIDTH / src_w, TILE_HEIGHT / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(source, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    x0 = (TILE_WIDTH - resized_w) // 2
    y0 = (TILE_HEIGHT - resized_h) // 2
    tile[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized

    lines = [title]
    if extra_lines:
        lines.extend(extra_lines)
    return put_status_lines(tile, lines)


def _blank_tile(text: str = "") -> np.ndarray:
    tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    return put_status_lines(tile, [text]) if text else tile


def _front_preview(camera: RealSenseColorSource) -> np.ndarray:
    rgbd = camera.read_rgbd()
    return np.hstack(
        (
            _fit_tile(rgbd.color_image, "FRONT / RealSense RGB"),
            _fit_tile(
                rgbd.depth_colormap,
                "FRONT / RealSense DEPTH",
                ["aligned to RGB"],
            ),
        )
    )


def _all_preview(front: RealSenseColorSource, left, right) -> np.ndarray:
    rgbd = front.read_rgbd()
    left_frame = left.read()
    right_frame = right.read()

    top = np.hstack(
        (
            _fit_tile(left_frame.image, "USER LEFT / C270 RGB"),
            _fit_tile(rgbd.color_image, "FRONT / RealSense RGB"),
            _fit_tile(right_frame.image, "USER RIGHT / C270 RGB"),
        )
    )
    bottom = np.hstack(
        (
            _blank_tile(),
            _fit_tile(
                rgbd.depth_colormap,
                "FRONT / RealSense DEPTH",
                ["aligned to RGB"],
            ),
            _blank_tile("Q or Esc: quit"),
        )
    )
    return np.vstack((top, bottom))


def _camera_settings(config_path: str | Path) -> tuple[int, int, int, str]:
    config = load_yaml(config_path)
    left = config["cameras"]["left"]
    return (
        int(config["image_width"]),
        int(config["image_height"]),
        int(config["fps"]),
        str(left.get("fourcc", "MJPG")),
    )


def _candidate_tile(camera: UsbCameraSource, candidate: VideoDeviceCandidate, number: int) -> np.ndarray:
    frame = camera.read()
    return _fit_tile(
        frame.image,
        f"C270 candidate {number}",
        [
            candidate.stable_path,
            f"Press {number} if this is USER LEFT",
        ],
    )


def assign_c270_roles(config_path: str | Path) -> int:
    """Discover only C270 RGB nodes and save their physical left/right roles."""

    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    left_cfg = config["cameras"]["left"]
    product_contains = str(left_cfg.get("product_contains", "C270"))
    candidates = discover_c270_sources(product_contains=product_contains)

    print("\n[C270 DISCOVERY]")
    print(format_device_report(candidates))
    if len(candidates) != 2:
        raise RuntimeError(
            "Exactly two C270 RGB capture devices are required. "
            f"Detected {len(candidates)}.\n"
            "RealSense nodes are intentionally excluded. Check USB connections and run again."
        )

    width, height, fps, fourcc = _camera_settings(config_path)
    with ExitStack() as stack:
        cameras = [
            stack.enter_context(
                UsbCameraSource(
                    name=f"c270_candidate_{index + 1}",
                    device=candidate.stable_path,
                    width=width,
                    height=height,
                    fps=fps,
                    fourcc=fourcc,
                )
            )
            for index, candidate in enumerate(candidates)
        ]

        print("\nLook at the two RGB previews.")
        print("Press 1 when candidate 1 is physically USER LEFT.")
        print("Press 2 when candidate 2 is physically USER LEFT.")
        print("Press Q or Esc to cancel.\n")

        selected_left: int | None = None
        while selected_left is None:
            preview = np.hstack(
                (
                    _candidate_tile(cameras[0], candidates[0], 1),
                    _candidate_tile(cameras[1], candidates[1], 2),
                )
            )
            preview = put_status_lines(
                preview,
                [
                    "C270 ROLE ASSIGNMENT",
                    "1: candidate 1 = USER LEFT",
                    "2: candidate 2 = USER LEFT",
                    "Q/Esc: cancel",
                ],
            )
            cv2.imshow("assign_c270_roles", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                selected_left = 0
            elif key == ord("2"):
                selected_left = 1
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("C270 role assignment cancelled.")
                return 1

    cv2.destroyAllWindows()
    selected_right = 1 - selected_left
    role_map_name = str(left_cfg.get("role_map", "camera_roles.yaml"))
    role_map_path = config_path.parent / role_map_name
    save_yaml(
        role_map_path,
        {
            "schema_version": 1,
            "c270_roles": {
                "left": candidates[selected_left].stable_path,
                "right": candidates[selected_right].stable_path,
            },
            "resolved_at_assignment": {
                "left": candidates[selected_left].resolved_device,
                "right": candidates[selected_right].resolved_device,
            },
            "usb_paths": {
                "left": candidates[selected_left].usb_path,
                "right": candidates[selected_right].usb_path,
            },
        },
    )
    print(f"\nSaved C270 roles: {role_map_path}")
    print(f"  USER LEFT  -> {candidates[selected_left].stable_path}")
    print(f"  USER RIGHT -> {candidates[selected_right].stable_path}")
    print("These stable USB-port paths are now used by every calibration script.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify camera placement. RealSense is always opened through "
            "pyrealsense2 as RGB+Depth. Only Logitech C270 RGB capture nodes "
            "can be assigned to user-left and user-right."
        )
    )
    parser.add_argument(
        "--camera",
        default="all",
        choices=["all", "front", "left", "right"],
        help="Default: all cameras in physical placement",
    )
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument(
        "--assign-c270",
        action="store_true",
        help="One-time interactive assignment of the two C270 RGB cameras",
    )
    parser.add_argument(
        "--list-c270",
        action="store_true",
        help="List only detected C270 RGB capture endpoints and exit",
    )
    args = parser.parse_args()

    if args.list_c270:
        config = load_yaml(args.cameras)
        token = str(config["cameras"]["left"].get("product_contains", "C270"))
        print(format_device_report(discover_c270_sources(token)))
        return 0

    if args.assign_c270:
        return assign_c270_roles(args.cameras)

    if args.camera == "all":
        with ExitStack() as stack:
            front = stack.enter_context(
                open_camera(args.cameras, "front", include_depth=True)
            )
            left = stack.enter_context(open_camera(args.cameras, "left"))
            right = stack.enter_context(open_camera(args.cameras, "right"))
            if not isinstance(front, RealSenseColorSource):
                raise TypeError("Configured front camera must be a RealSense RGB-D camera")

            while True:
                cv2.imshow("camera_check_all", _all_preview(front, left, right))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    elif args.camera == "front":
        with open_camera(args.cameras, "front", include_depth=True) as camera:
            if not isinstance(camera, RealSenseColorSource):
                raise TypeError("Configured front camera must be a RealSense RGB-D camera")
            while True:
                preview = put_status_lines(
                    _front_preview(camera),
                    ["FRONT RGB-D CHECK", "Q or Esc: quit"],
                )
                cv2.imshow("camera_front_rgbd", preview)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    else:
        with open_camera(args.cameras, args.camera) as camera:
            while True:
                frame = camera.read()
                title = (
                    "USER LEFT / C270 RGB"
                    if args.camera == "left"
                    else "USER RIGHT / C270 RGB"
                )
                cv2.imshow(
                    f"camera_{args.camera}",
                    _fit_tile(frame.image, title, ["Q or Esc: quit"]),
                )
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

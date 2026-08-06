#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_io import open_camera
from robot_calibration.common import (
    IntrinsicCalibration,
    estimate_board_pose,
    load_yaml,
    put_status_lines,
    save_json,
    transform_error,
)
from robot_calibration.doosan_pose import DoosanFlangePoseProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a solved camera-to-BASE transform using new robot/ChArUco poses"
    )
    parser.add_argument("--camera", required=True, choices=["front", "left", "right"])
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--intrinsic", default="")
    parser.add_argument("--calibration", default="calibration/results/robot_world.yaml")
    parser.add_argument("--namespace", default="/dsr01")
    parser.add_argument("--minimum-corners", type=int, default=20)
    parser.add_argument("--output-dir", default="calibration/validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intrinsic_path = Path(args.intrinsic) if args.intrinsic else Path("calibration/intrinsics") / f"{args.camera}.yaml"
    intrinsic = IntrinsicCalibration.load(intrinsic_path)
    calibration = load_yaml(args.calibration)
    try:
        T_base_camera = np.asarray(
            calibration["cameras"][args.camera]["T_base_camera_mm"], dtype=np.float64
        ).reshape(4, 4)
        T_flange_board = np.asarray(
            calibration["board_mount"]["T_flange_board_mm"], dtype=np.float64
        ).reshape(4, 4)
    except KeyError as error:
        raise RuntimeError(f"Missing calibration entry: {error}") from error
    output_dir = Path(args.output_dir) / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[VALIDATE] Move to poses that were NOT used for calibration, release hand-guide, then press S.")

    with open_camera(args.cameras, args.camera) as camera, DoosanFlangePoseProvider(
        namespace=args.namespace
    ) as robot:
        while True:
            frame = camera.read()
            board_pose, annotated = estimate_board_pose(
                frame.image, intrinsic, minimum_corners=args.minimum_corners
            )
            ready = board_pose is not None
            preview = put_status_lines(
                annotated,
                [
                    f"VALIDATION: {args.camera}",
                    "board READY" if ready else "board NOT READY",
                    (
                        f"corners={board_pose.corner_count}, reproj={board_pose.reprojection_error_px:.3f}px"
                        if ready
                        else ""
                    ),
                    "S: measure validation error   Q: quit",
                ],
            )
            cv2.imshow(f"validate_{args.camera}", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue
            try:
                robot_pose = robot.stable_sample()
                fresh = camera.read()
                pose, annotated_saved = estimate_board_pose(
                    fresh.image, intrinsic, minimum_corners=args.minimum_corners
                )
                if pose is None:
                    raise RuntimeError("Board pose unavailable")
                T_base_board_robot = robot_pose.T_base_flange_mm @ T_flange_board
                T_base_board_camera = T_base_camera @ pose.T_camera_board_mm
                translation_error_mm, rotation_error_deg = transform_error(
                    T_base_board_robot, T_base_board_camera
                )
                sample_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(output_dir / f"{sample_id}.png"), annotated_saved)
                save_json(
                    output_dir / f"{sample_id}.json",
                    {
                        "camera": args.camera,
                        "translation_error_mm": translation_error_mm,
                        "rotation_error_deg": rotation_error_deg,
                        "board_reprojection_error_px": pose.reprojection_error_px,
                        "corner_count": pose.corner_count,
                        "T_base_board_from_robot_mm": T_base_board_robot.tolist(),
                        "T_base_board_from_camera_mm": T_base_board_camera.tolist(),
                        "robot_pose": robot_pose.to_dict(),
                    },
                )
                print(
                    f"[VALIDATION] camera={args.camera}, "
                    f"translation={translation_error_mm:.3f}mm, "
                    f"rotation={rotation_error_deg:.3f}deg, "
                    f"reproj={pose.reprojection_error_px:.3f}px"
                )
            except RuntimeError as error:
                print(f"[REJECTED] {error}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

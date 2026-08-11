#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_io import open_camera
from robot_calibration.common import (
    IntrinsicCalibration,
    estimate_board_pose,
    put_status_lines,
    save_json,
    transform_error,
)
from robot_calibration.doosan_pose import DoosanFlangePoseProvider


def load_existing_robot_poses(dataset_dir: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for path in sorted(dataset_dir.glob("*/metadata.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            poses.append(np.asarray(data["robot_pose"]["T_base_flange_mm"], dtype=np.float64))
        except (KeyError, ValueError, json.JSONDecodeError):
            print(f"[WARN] Ignoring malformed metadata: {path}")
    return poses


def sufficiently_different(
    candidate: np.ndarray,
    existing: list[np.ndarray],
    minimum_translation_mm: float,
    minimum_rotation_deg: float,
) -> tuple[bool, float, float]:
    if not existing:
        return True, float("inf"), float("inf")
    errors = [transform_error(reference, candidate) for reference in existing]
    nearest = min(errors, key=lambda item: item[0] / max(minimum_translation_mm, 1e-9) + item[1] / max(minimum_rotation_deg, 1e-9))
    accepted = nearest[0] >= minimum_translation_mm or nearest[1] >= minimum_rotation_deg
    return accepted, nearest[0], nearest[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect eye-to-hand ChArUco samples for one fixed camera"
    )
    parser.add_argument("--camera", required=True, choices=["front", "left", "right"])
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--intrinsic", default="")
    parser.add_argument("--output-root", default="calibration/external_samples")
    parser.add_argument("--namespace", default="/dsr01")
    parser.add_argument("--target-samples", type=int, default=25)
    parser.add_argument("--minimum-corners", type=int, default=20)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=1.2)
    parser.add_argument("--stability-samples", type=int, default=5)
    parser.add_argument("--stability-interval-s", type=float, default=0.08)
    parser.add_argument("--max-stability-translation-mm", type=float, default=0.3)
    parser.add_argument("--max-stability-rotation-deg", type=float, default=0.2)
    parser.add_argument("--minimum-pose-translation-mm", type=float, default=15.0)
    parser.add_argument("--minimum-pose-rotation-deg", type=float, default=4.0)
    parser.add_argument("--allow-similar-pose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intrinsic_path = Path(args.intrinsic) if args.intrinsic else Path("calibration/intrinsics") / f"{args.camera}.yaml"
    intrinsic = IntrinsicCalibration.load(intrinsic_path)
    dataset_dir = Path(args.output_root) / args.camera
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing_poses = load_existing_robot_poses(dataset_dir)
    saved_count = len(existing_poses)
    print(f"[DATASET] {dataset_dir}: existing samples={saved_count}")
    print("[WORKFLOW] Hand-guide/teach the robot, release it, wait until fully stationary, then press S.")

    with open_camera(args.cameras, args.camera) as camera, DoosanFlangePoseProvider(
        namespace=args.namespace
    ) as robot:
        while True:
            frame = camera.read()
            pose, annotated = estimate_board_pose(
                frame.image, intrinsic, minimum_corners=args.minimum_corners
            )
            if pose is None:
                pose_text = "board pose: NOT READY"
                reprojection_text = ""
            else:
                pose_text = f"board corners={pose.corner_count}"
                reprojection_text = f"reprojection={pose.reprojection_error_px:.3f}px"
            preview = put_status_lines(
                annotated,
                [
                    f"EXTERNAL CAPTURE: {args.camera}",
                    f"saved={saved_count}/{args.target_samples}",
                    pose_text,
                    reprojection_text,
                    "S: save stable sample   Q: quit",
                ],
            )
            cv2.imshow(f"external_{args.camera}", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue
            try:
                robot_pose = robot.stable_sample(
                    count=args.stability_samples,
                    interval_s=args.stability_interval_s,
                    max_translation_spread_mm=args.max_stability_translation_mm,
                    max_rotation_spread_deg=args.max_stability_rotation_deg,
                )
                fresh = camera.read()
                board_pose, saved_annotated = estimate_board_pose(
                    fresh.image,
                    intrinsic,
                    minimum_corners=args.minimum_corners,
                )
                if board_pose is None:
                    raise RuntimeError("ChArUco board pose is unavailable or has too few corners")
                if board_pose.reprojection_error_px > args.maximum_reprojection_error_px:
                    raise RuntimeError(
                        f"Reprojection error {board_pose.reprojection_error_px:.3f}px exceeds "
                        f"{args.maximum_reprojection_error_px:.3f}px"
                    )
                different, nearest_t, nearest_r = sufficiently_different(
                    robot_pose.T_base_flange_mm,
                    existing_poses,
                    args.minimum_pose_translation_mm,
                    args.minimum_pose_rotation_deg,
                )
                if not different and not args.allow_similar_pose:
                    raise RuntimeError(
                        "Pose is too similar to an existing sample: "
                        f"nearest translation={nearest_t:.2f}mm, rotation={nearest_r:.2f}deg. "
                        "Change board tilt/roll or position."
                    )
                sample_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                sample_dir = dataset_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=False)
                image_path = sample_dir / "image.png"
                annotated_path = sample_dir / "annotated.png"
                if not cv2.imwrite(str(image_path), fresh.image):
                    raise RuntimeError(f"Failed to save {image_path}")
                cv2.imwrite(str(annotated_path), saved_annotated)
                metadata = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "camera_name": args.camera,
                    "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "image_path": "image.png",
                    "camera_frame_timestamp_ns": int(fresh.timestamp_ns),
                    "intrinsic_path": str(intrinsic_path),
                    "board_pose_preview": {
                        "T_camera_board_mm": board_pose.T_camera_board_mm.tolist(),
                        "reprojection_error_px": float(board_pose.reprojection_error_px),
                        "corner_count": int(board_pose.corner_count),
                    },
                    "robot_pose": robot_pose.to_dict(),
                }
                save_json(sample_dir / "metadata.json", metadata)
                existing_poses.append(robot_pose.T_base_flange_mm)
                saved_count += 1
                print(
                    f"[SAVED] {sample_dir} corners={board_pose.corner_count}, "
                    f"reproj={board_pose.reprojection_error_px:.3f}px, "
                    f"robot stability={robot_pose.stability_translation_mm:.3f}mm/"
                    f"{robot_pose.stability_rotation_deg:.3f}deg"
                )
                if saved_count >= args.target_samples:
                    print("[INFO] Target reached. You may quit or collect more diverse poses.")
            except RuntimeError as error:
                print(f"[REJECTED] {error}")
    cv2.destroyAllWindows()
    print(f"[DONE] {args.camera}: total samples={saved_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

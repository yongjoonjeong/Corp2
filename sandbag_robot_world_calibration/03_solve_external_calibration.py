#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.common import (
    IntrinsicCalibration,
    estimate_board_pose,
    invert_transform,
    load_yaml,
    save_yaml,
)
from robot_calibration.external_solver import ExternalSample, solve_robot_world_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly solve camera-to-robot BASE extrinsics with a shared flange-to-board transform"
    )
    parser.add_argument("--samples-root", default="calibration/external_samples")
    parser.add_argument("--intrinsics-root", default="calibration/intrinsics")
    parser.add_argument("--cameras", nargs="+", default=["front", "left", "right"])
    parser.add_argument("--output", default="calibration/results/robot_world.yaml")
    parser.add_argument("--minimum-samples-per-camera", type=int, default=12)
    parser.add_argument("--minimum-corners", type=int, default=20)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=1.5)
    parser.add_argument("--outlier-translation-mm", type=float, default=12.0)
    parser.add_argument("--outlier-rotation-deg", type=float, default=5.0)
    parser.add_argument("--translation-scale-mm", type=float, default=5.0)
    parser.add_argument("--rotation-scale-deg", type=float, default=2.0)
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> tuple[list[ExternalSample], dict[str, IntrinsicCalibration]]:
    samples: list[ExternalSample] = []
    intrinsics: dict[str, IntrinsicCalibration] = {}
    for camera_name in args.cameras:
        intrinsic_path = Path(args.intrinsics_root) / f"{camera_name}.yaml"
        intrinsic = IntrinsicCalibration.load(intrinsic_path)
        intrinsics[camera_name] = intrinsic
        dataset_dir = Path(args.samples_root) / camera_name
        metadata_paths = sorted(dataset_dir.glob("*/metadata.json"))
        print(f"[LOAD] {camera_name}: metadata files={len(metadata_paths)}")
        for metadata_path in metadata_paths:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                image_path = metadata_path.parent / metadata.get("image_path", "image.png")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot read image: {image_path}")
                pose, _ = estimate_board_pose(
                    image,
                    intrinsic,
                    minimum_corners=args.minimum_corners,
                )
                if pose is None:
                    print(f"[SKIP] {metadata_path.parent.name}: board pose unavailable")
                    continue
                T_base_flange = np.asarray(
                    metadata["robot_pose"]["T_base_flange_mm"], dtype=np.float64
                ).reshape(4, 4)
                samples.append(
                    ExternalSample(
                        camera_name=camera_name,
                        sample_id=str(metadata.get("sample_id", metadata_path.parent.name)),
                        T_base_flange_mm=T_base_flange,
                        T_camera_board_mm=pose.T_camera_board_mm,
                        reprojection_error_px=pose.reprojection_error_px,
                        corner_count=pose.corner_count,
                    )
                )
                print(
                    f"  [OK] {metadata_path.parent.name}: corners={pose.corner_count}, "
                    f"reproj={pose.reprojection_error_px:.3f}px"
                )
            except (KeyError, ValueError, json.JSONDecodeError, RuntimeError) as error:
                print(f"[SKIP] {metadata_path}: {error}")
    return samples, intrinsics


def main() -> int:
    args = parse_args()
    samples, intrinsics = load_samples(args)
    result = solve_robot_world_calibration(
        samples,
        minimum_samples_per_camera=args.minimum_samples_per_camera,
        maximum_reprojection_error_px=args.maximum_reprojection_error_px,
        translation_scale_mm=args.translation_scale_mm,
        rotation_scale_deg=args.rotation_scale_deg,
        outlier_translation_mm=args.outlier_translation_mm,
        outlier_rotation_deg=args.outlier_rotation_deg,
    )
    cameras_payload = {}
    for camera_name, T_base_camera in result.T_base_camera_mm.items():
        cameras_payload[camera_name] = {
            "camera_frame": f"{camera_name}_camera_optical_frame",
            "T_base_camera_mm": T_base_camera.tolist(),
            "T_camera_base_mm": invert_transform(T_base_camera).tolist(),
            "intrinsic_file": str(Path(args.intrinsics_root) / f"{camera_name}.yaml"),
            "image_width": intrinsics[camera_name].image_width,
            "image_height": intrinsics[camera_name].image_height,
        }
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "units": {"translation": "mm", "rotation": "3x3 rotation matrix"},
        "frame_convention": {
            "transform_name": "T_target_source maps source-frame coordinates into target frame",
            "equation": "T_base_flange * T_flange_board = T_base_camera * T_camera_board",
            "camera_axes": "OpenCV optical frame: +X right, +Y down, +Z forward",
            "robot_frame": "Doosan DR_BASE",
        },
        "board_mount": {
            "T_flange_board_mm": result.T_flange_board_mm.tolist(),
            "T_board_flange_mm": invert_transform(result.T_flange_board_mm).tolist(),
            "note": "Estimated automatically; no tape-measure flange-to-board input was used.",
        },
        "cameras": cameras_payload,
        "metrics": result.metrics,
        "used_samples": [
            {
                "camera": sample.camera_name,
                "sample_id": sample.sample_id,
                "reprojection_error_px": float(sample.reprojection_error_px),
                "corner_count": int(sample.corner_count),
            }
            for sample in result.samples_used
        ],
        "rejected_samples": [
            {
                "camera": sample.camera_name,
                "sample_id": sample.sample_id,
                "reprojection_error_px": float(sample.reprojection_error_px),
                "corner_count": int(sample.corner_count),
            }
            for sample in result.samples_rejected
        ],
        "opencv_version": cv2.__version__,
    }
    save_yaml(args.output, payload)
    print(f"[DONE] Robot-world calibration saved: {args.output}")
    print("[RESULT] Shared T_flange_board_mm:")
    print(result.T_flange_board_mm)
    for camera_name, matrix in result.T_base_camera_mm.items():
        print(f"[RESULT] T_base_{camera_name}_camera_mm:")
        print(matrix)
    print("[METRICS]")
    print(result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

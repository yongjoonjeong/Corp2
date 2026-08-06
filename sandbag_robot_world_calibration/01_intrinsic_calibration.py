#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_io import open_camera
from robot_calibration.common import (
    BoardSpec,
    IntrinsicCalibration,
    detect_charuco,
    load_yaml,
    put_status_lines,
)


def collect_images(args: argparse.Namespace, board: BoardSpec, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.png"))
    count = len(existing)
    with open_camera(args.cameras, args.camera) as camera:
        while True:
            frame = camera.read()
            observation, annotated = detect_charuco(frame.image, board)
            corner_count = 0 if observation is None else len(observation.ids)
            status = [
                f"INTRINSIC: {args.camera}",
                f"saved={count}/{args.target_images}, corners={corner_count}",
                "S: save   C: solve   Q: quit",
                "Move board across corners, distances, and tilts",
            ]
            preview = put_status_lines(annotated, status)
            cv2.imshow(f"intrinsic_{args.camera}", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                break
            if key == ord("s"):
                if observation is None or corner_count < board.minimum_charuco_corners:
                    print(
                        f"[SKIP] Need at least {board.minimum_charuco_corners} corners; "
                        f"detected {corner_count}"
                    )
                    continue
                count += 1
                filename = output_dir / f"{count:04d}_{frame.timestamp_ns}.png"
                if not cv2.imwrite(str(filename), frame.image):
                    raise RuntimeError(f"Failed to save {filename}")
                print(f"[SAVED] {filename} corners={corner_count}")
                if count >= args.target_images:
                    print("[INFO] Target image count reached. Press C to calculate or keep collecting.")
    cv2.destroyAllWindows()


def per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    tvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        projected = np.asarray(projected).reshape(-1, 2)
        actual = np.asarray(img).reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum((projected - actual) ** 2, axis=1)))))
    return errors


def calibrate(args: argparse.Namespace, board: BoardSpec, image_dir: Path) -> Path:
    camera_config = load_yaml(args.cameras)
    image_size = (int(camera_config["image_width"]), int(camera_config["image_height"]))
    image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) < args.minimum_images:
        raise RuntimeError(
            f"Only {len(image_paths)} images found; collect at least {args.minimum_images}"
        )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    valid_paths: list[Path] = []
    corner_counts: list[int] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[SKIP] Cannot read {path}")
            continue
        if (image.shape[1], image.shape[0]) != image_size:
            print(f"[SKIP] Wrong resolution {path}: {image.shape[1]}x{image.shape[0]}")
            continue
        observation, _ = detect_charuco(image, board)
        if observation is None or len(observation.ids) < board.minimum_charuco_corners:
            print(f"[SKIP] Insufficient ChArUco corners: {path}")
            continue
        object_points.append(observation.object_points_mm.astype(np.float32))
        image_points.append(observation.corners_px.reshape(-1, 1, 2).astype(np.float32))
        valid_paths.append(path)
        corner_counts.append(len(observation.ids))
    if len(valid_paths) < args.minimum_images:
        raise RuntimeError(
            f"Only {len(valid_paths)} valid views remain; need {args.minimum_images}"
        )

    flags = 0
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    errors = per_view_errors(object_points, image_points, rvecs, tvecs, K, D)
    error_array = np.asarray(errors)
    median = float(np.median(error_array))
    mad = float(np.median(np.abs(error_array - median)))
    robust_limit = median + 3.0 * max(1.4826 * mad, 0.05)
    limit = min(float(args.maximum_view_error_px), robust_limit)
    keep_indices = [index for index, error in enumerate(errors) if error <= limit]
    if len(keep_indices) >= args.minimum_images and len(keep_indices) < len(valid_paths):
        rejected = [valid_paths[index] for index in range(len(valid_paths)) if index not in keep_indices]
        print(f"[OUTLIER] Removing {len(rejected)} views above {limit:.3f} px")
        for path in rejected:
            print(f"  - {path.name}")
        object_points = [object_points[index] for index in keep_indices]
        image_points = [image_points[index] for index in keep_indices]
        valid_paths = [valid_paths[index] for index in keep_indices]
        corner_counts = [corner_counts[index] for index in keep_indices]
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            K,
            D,
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        )
        errors = per_view_errors(object_points, image_points, rvecs, tvecs, K, D)

    mean_error = float(np.mean(errors))
    intrinsic = IntrinsicCalibration(
        camera_name=args.camera,
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=K,
        distortion=D,
        rms_px=float(rms),
        mean_reprojection_error_px=mean_error,
        board=board,
    )
    output_path = Path(args.output) if args.output else Path("calibration/intrinsics") / f"{args.camera}.yaml"
    intrinsic.save(
        output_path,
        extra={
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "valid_view_count": len(valid_paths),
            "source_images": [str(path) for path in valid_paths],
            "per_view_reprojection_error_px": [float(value) for value in errors],
            "charuco_corner_counts": corner_counts,
            "opencv_version": cv2.__version__,
        },
    )
    print(f"[DONE] Intrinsic calibration saved: {output_path}")
    print(f"       RMS={rms:.4f} px, mean view error={mean_error:.4f} px")
    print(f"       K=\n{K}")
    print(f"       D={D.reshape(-1)}")
    if mean_error > 0.8:
        print("[WARN] Mean error is high. Re-shoot blurry/repetitive views and cover image edges.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and solve one camera's ChArUco intrinsic calibration"
    )
    parser.add_argument("--camera", required=True, choices=["front", "left", "right"])
    parser.add_argument("--board", default="config/board.yaml")
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--target-images", type=int, default=35)
    parser.add_argument("--minimum-images", type=int, default=15)
    parser.add_argument("--maximum-view-error-px", type=float, default=1.2)
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--fix-principal-point", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    board = BoardSpec.load(args.board)
    image_dir = (
        Path(args.image_dir)
        if args.image_dir
        else Path("calibration/intrinsic_images") / args.camera
    )
    if not args.solve_only:
        collect_images(args, board, image_dir)
    calibrate(args, board, image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from camera_device_discovery import load_camera_role_devices, resolve_stereo_webcam_devices
from punch_feedback_3d_core import (
    LANDMARK_NAMES_3D,
    ClassifiedPunch3D,
    Landmark3D,
    LandmarkObservation2D,
    PoseSample3D,
    PunchDetector3D,
    PunchEvent3D,
    classify_punch_3d,
    fuse_realsense_depth,
    load_three_camera_calibration,
    score_punch_3d,
    triangulate_landmark,
)
from punch_feedback_core import ClassifiedPunch, ScoreResult
from punch_feedback_ui import (
    PUNCH_COLOR_BGR,
    draw_guard_gauge,
    draw_pose,
    feature_error_dict,
    load_reference,
    render_evidence,
)
from yolo_pose_backend import (
    CalibratedPoseSelector,
    YoloPoseBackend,
    YoloPoseDetection,
    depth_mask_from_pose_box,
)


CAMERA_DISPLAY_ORDER = ("left", "front", "right")
CAMERA_DISPLAY_LABELS = {
    "left": "LEFT C270",
    "front": "FRONT REALSENSE RGB + DEPTH FUSION",
    "right": "RIGHT C270",
}
CAMERA_LABEL_HEIGHT = 34
SKELETON_EDGES_3D = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)


@dataclass(frozen=True)
class TimedFrame:
    image: np.ndarray
    stamp_s: float


class LatestUsbCamera:
    """Continuously keeps the freshest frame to reduce multi-camera skew."""

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.device = device
        source: str | int = int(device) if device.isdigit() else device
        self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.capture.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다: {device}")

        self._lock = threading.Lock()
        self._latest: TimedFrame | None = None
        self.read_success_count = 0
        self.read_failure_count = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{Path(device).name}",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            stamp_s = time.monotonic()
            if not ok or frame is None:
                self.read_failure_count += 1
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = TimedFrame(frame, stamp_s)
                self.read_success_count += 1

    def latest(self) -> TimedFrame | None:
        with self._lock:
            if self._latest is None:
                return None
            return TimedFrame(self._latest.image.copy(), self._latest.stamp_s)

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self.read_success_count, self.read_failure_count

    def close(self) -> None:
        self._running = False
        self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def landmark_observations(
    detection: YoloPoseDetection | None,
    width: int,
    height: int,
    minimum_confidence: float,
) -> dict[str, LandmarkObservation2D]:
    if detection is None:
        return {}
    return detection.observations(width, height, minimum_confidence)


def median_depth_m(
    depth_frame: Any,
    pixel: tuple[float, float],
    radius: int,
    width: int,
    height: int,
) -> float | None:
    center_x = int(round(pixel[0]))
    center_y = int(round(pixel[1]))
    values: list[float] = []
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            distance = float(depth_frame.get_distance(x, y))
            if 0.15 <= distance <= 8.0:
                values.append(distance)
    return float(np.median(values)) if values else None


def colored_person_point_cloud_base(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    segmentation_mask: np.ndarray | None,
    intrinsics: Any,
    depth_scale_m: float,
    T_base_front_mm: np.ndarray,
    *,
    stride: int = 4,
    segmentation_threshold: float = 0.55,
    minimum_depth_mm: float = 250.0,
    maximum_depth_mm: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize aligned RealSense RGB-D pixels into a masked BASE point cloud."""
    if segmentation_mask is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    step = max(int(stride), 1)
    depth = np.asarray(depth_raw)[::step, ::step].astype(np.float64)
    mask = np.asarray(segmentation_mask, dtype=np.float32)
    if mask.shape != depth_raw.shape:
        mask = cv2.resize(mask, (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = mask[::step, ::step]
    colors = np.asarray(color_bgr, dtype=np.uint8)[::step, ::step]
    z_mm = depth * float(depth_scale_m) * 1000.0
    valid = (
        (mask >= float(segmentation_threshold))
        & (z_mm >= float(minimum_depth_mm))
        & (z_mm <= float(maximum_depth_mm))
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    u = columns * step
    v = rows * step
    z = z_mm[valid]
    x = (u[valid] - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y = (v[valid] - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    points_front = np.column_stack((x, y, z, np.ones_like(z)))
    points_base = (np.asarray(T_base_front_mm, dtype=np.float64) @ points_front.T).T[:, :3]
    return points_base, colors[valid]


def resize_with_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize a camera frame without changing its aspect ratio."""
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def compose_three_camera_strip(
    camera_views: Mapping[str, np.ndarray],
    total_width: int,
    total_height: int,
) -> np.ndarray:
    """Place left, front, and right camera frames side by side."""
    base_width = total_width // len(CAMERA_DISPLAY_ORDER)
    remaining_width = total_width - base_width * len(CAMERA_DISPLAY_ORDER)
    tiles: list[np.ndarray] = []
    for index, camera in enumerate(CAMERA_DISPLAY_ORDER):
        tile_width = base_width + int(index < remaining_width)
        tiles.append(
            resize_with_letterbox(
                camera_views[camera],
                tile_width,
                total_height,
            )
        )
    strip = np.hstack(tiles)
    separator_x = 0
    for tile in tiles[:-1]:
        separator_x += tile.shape[1]
        cv2.line(
            strip,
            (separator_x, 0),
            (separator_x, total_height - 1),
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
    return strip


def render_robot_base_3d_map(
    calibration: Any,
    sample: PoseSample3D | None,
    width: int,
    height: int,
    person_points_base_mm: np.ndarray | None = None,
    person_colors_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """Render an isometric BASE-frame map without an extra 3D dependency."""
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    if calibration.T_base_front_mm is None:
        cv2.putText(canvas, "No robot BASE transform", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        return canvas

    camera_poses = {
        name: calibration.camera_pose_in_base(name)
        for name in CAMERA_DISPLAY_ORDER
    }
    skeleton = {}
    if sample is not None:
        skeleton = {
            name: calibration.front_point_to_base(point.xyz)
            for name, point in sample.landmarks.items()
        }

    cloud_points = np.asarray(person_points_base_mm, dtype=np.float64).reshape(-1, 3) if person_points_base_mm is not None else np.empty((0, 3))
    cloud_colors = np.asarray(person_colors_bgr, dtype=np.uint8).reshape(-1, 3) if person_colors_bgr is not None else np.empty((0, 3), dtype=np.uint8)
    if len(cloud_points) != len(cloud_colors):
        raise ValueError("Point-cloud positions and colors must have equal length")
    points = [np.zeros(3), *(pose[:3, 3] for pose in camera_poses.values()), *skeleton.values(), *cloud_points[::max(len(cloud_points) // 1000, 1)]]
    xy = np.asarray([point[:2] for point in points], dtype=np.float64)
    center_xy = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    span_xy = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1200.0)
    z_max = max([float(point[2]) for point in points] + [1200.0])

    azimuth = np.radians(-42.0)
    elevation = np.radians(28.0)
    screen_right = np.asarray((-np.sin(azimuth), np.cos(azimuth), 0.0))
    screen_up = np.asarray(
        (-np.sin(elevation) * np.cos(azimuth), -np.sin(elevation) * np.sin(azimuth), np.cos(elevation))
    )
    scale = min(width * 0.72 / span_xy, height * 0.62 / max(span_xy, z_max))
    world_center = np.asarray((center_xy[0], center_xy[1], z_max * 0.35))

    def project(point: np.ndarray) -> tuple[int, int]:
        relative = np.asarray(point, dtype=np.float64).reshape(3) - world_center
        return (
            int(round(width * 0.5 + np.dot(relative, screen_right) * scale)),
            int(round(height * 0.56 - np.dot(relative, screen_up) * scale)),
        )

    grid_step = 500.0
    grid_radius = max(1500.0, np.ceil(span_xy / grid_step) * grid_step)
    x0 = np.floor((center_xy[0] - grid_radius) / grid_step) * grid_step
    x1 = np.ceil((center_xy[0] + grid_radius) / grid_step) * grid_step
    y0 = np.floor((center_xy[1] - grid_radius) / grid_step) * grid_step
    y1 = np.ceil((center_xy[1] + grid_radius) / grid_step) * grid_step
    for x in np.arange(x0, x1 + 1.0, grid_step):
        cv2.line(canvas, project(np.asarray((x, y0, 0.0))), project(np.asarray((x, y1, 0.0))), (48, 48, 48), 1, cv2.LINE_AA)
    for y in np.arange(y0, y1 + 1.0, grid_step):
        cv2.line(canvas, project(np.asarray((x0, y, 0.0))), project(np.asarray((x1, y, 0.0))), (48, 48, 48), 1, cv2.LINE_AA)

    origin = np.zeros(3)
    axis_length = 500.0
    for vector, color, label in (
        (np.asarray((axis_length, 0.0, 0.0)), (40, 40, 255), "+X"),
        (np.asarray((0.0, axis_length, 0.0)), (40, 255, 40), "+Y"),
        (np.asarray((0.0, 0.0, axis_length)), (255, 120, 40), "+Z"),
    ):
        endpoint = project(vector)
        cv2.arrowedLine(canvas, project(origin), endpoint, color, 2, cv2.LINE_AA, tipLength=0.14)
        cv2.putText(canvas, label, (endpoint[0] + 4, endpoint[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, project(origin), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "M0609 BASE", (project(origin)[0] + 10, project(origin)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    camera_colors = {"left": (255, 180, 20), "front": (40, 220, 255), "right": (255, 80, 200)}
    for name, pose in camera_poses.items():
        center = pose[:3, 3]
        forward = center + pose[:3, 2] * 350.0
        color = camera_colors[name]
        cv2.circle(canvas, project(center), 8, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, project(center), project(forward), color, 2, cv2.LINE_AA, tipLength=0.18)
        label_at = project(center)
        cv2.putText(canvas, name.upper(), (label_at[0] + 10, label_at[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    if len(cloud_points):
        relative = cloud_points - world_center
        pixel_x = np.rint(width * 0.5 + relative @ screen_right * scale).astype(np.int32)
        pixel_y = np.rint(height * 0.56 - relative @ screen_up * scale).astype(np.int32)
        visible = (pixel_x >= 1) & (pixel_x < width - 1) & (pixel_y >= 1) & (pixel_y < height - 1)
        pixel_x = pixel_x[visible]
        pixel_y = pixel_y[visible]
        visible_colors = cloud_colors[visible]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            canvas[pixel_y + dy, pixel_x + dx] = visible_colors

    if skeleton:
        torso_names = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        if all(name in skeleton for name in torso_names):
            torso = np.asarray([project(skeleton[name]) for name in torso_names], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, torso, (55, 145, 205), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.52, canvas, 0.48, 0.0, canvas)
        for first, second in SKELETON_EDGES_3D:
            if first in skeleton and second in skeleton:
                thickness = 18 if "hip" not in (first, second) else 12
                cv2.line(canvas, project(skeleton[first]), project(skeleton[second]), (25, 45, 60), thickness + 6, cv2.LINE_AA)
                cv2.line(canvas, project(skeleton[first]), project(skeleton[second]), (75, 205, 245), thickness, cv2.LINE_AA)
        if "nose" in skeleton and "left_shoulder" in skeleton and "right_shoulder" in skeleton:
            nose_px = project(skeleton["nose"])
            shoulders_px = np.asarray((project(skeleton["left_shoulder"]), project(skeleton["right_shoulder"])))
            shoulder_width_px = float(np.linalg.norm(shoulders_px[0] - shoulders_px[1]))
            head_radius = max(12, int(round(shoulder_width_px * 0.32)))
            cv2.circle(canvas, nose_px, head_radius + 4, (25, 45, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, nose_px, head_radius, (90, 190, 235), -1, cv2.LINE_AA)
        for name, point in skeleton.items():
            color = (40, 80, 255) if name.endswith("wrist") else (80, 255, 180)
            radius = 10 if name.endswith("wrist") else 6
            cv2.circle(canvas, project(point), radius, color, -1, cv2.LINE_AA)
        right_wrist = skeleton.get("right_wrist")
        if right_wrist is not None:
            cv2.putText(canvas, f"R wrist BASE [{right_wrist[0]:.0f}, {right_wrist[1]:.0f}, {right_wrist[2]:.0f}] mm", (20, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2, cv2.LINE_AA)
        status = f"RGB-D PERSON TRACKED | POINTS {len(cloud_points)}"
        status_color = (60, 240, 60)
    else:
        status = "WAITING FOR 3D PERSON"
        status_color = (60, 180, 255)
    cv2.putText(canvas, "ROBOT BASE 3D MAP", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
    return canvas


class ThreeCameraPunchFeedbackNode(Node):
    def __init__(self) -> None:
        super().__init__("three_camera_punch_feedback")
        project_dir = Path(__file__).resolve().parent
        direct_calibration = (
            project_dir / "calibration" / "three_camera_charuco_calibration.npz"
        )
        local_robot_world = project_dir / "calibration" / "results" / "robot_world.yaml"
        robot_world_project = (
            project_dir
            if local_robot_world.is_file()
            else project_dir.parent / "sandbag_robot_world_calibration"
        )
        robot_world_calibration = (
            robot_world_project / "calibration" / "results" / "robot_world.yaml"
        )
        default_calibration = (
            robot_world_calibration
            if robot_world_calibration.is_file()
            else direct_calibration
        )
        default_role_map = robot_world_project / "config" / "camera_roles.yaml"
        default_reference = project_dir / "config" / "temporary_form_reference_3d.yaml"
        default_feedback_dir = project_dir / "feedback_images_3d"
        default_pose_model = project_dir / "models" / "yolo11n-pose.pt"

        self.declare_parameter("left_device", "auto")
        self.declare_parameter("right_device", "auto")
        self.declare_parameter("swap_webcams", False)
        self.declare_parameter("camera_role_map", str(default_role_map))
        self.declare_parameter("calibration_path", str(default_calibration))
        self.declare_parameter("reference_path", str(default_reference))
        self.declare_parameter("capture_fps", 30.0)
        self.declare_parameter("processing_fps", 10.0)
        self.declare_parameter("max_sync_spread_ms", 100.0)
        self.declare_parameter("pose_model", str(default_pose_model))
        self.declare_parameter("pose_device", "auto")
        self.declare_parameter("pose_image_size", 640)
        self.declare_parameter("pose_detection_confidence", 0.25)
        self.declare_parameter("pose_maximum_detections", 5)
        self.declare_parameter("min_landmark_visibility", 0.35)
        self.declare_parameter("pose_association_max_reprojection_px", 40.0)
        self.declare_parameter("pose_target_lost_timeout_frames", 30)
        self.declare_parameter("max_reprojection_error_px", 10.0)
        self.declare_parameter("fist_max_reprojection_error_px", 25.0)
        self.declare_parameter("depth_fusion_weight", 0.35)
        self.declare_parameter("max_depth_disagreement_mm", 220.0)
        self.declare_parameter("depth_patch_radius", 3)
        self.declare_parameter("missing_pose_reset_frames", 20)
        self.declare_parameter("diagnostic_log_interval_s", 5.0)
        self.declare_parameter("camera_startup_timeout_s", 4.0)
        self.declare_parameter("display", True)
        self.declare_parameter("map_display", True)
        self.declare_parameter("map_width", 900)
        self.declare_parameter("map_height", 700)
        self.declare_parameter("person_point_cloud", True)
        self.declare_parameter("point_cloud_stride", 4)
        self.declare_parameter("segmentation_threshold", 0.55)
        self.declare_parameter("point_cloud_depth_band_mm", 350.0)
        self.declare_parameter("point_cloud_bbox_margin", 0.05)
        self.declare_parameter("point_cloud_minimum_depth_mm", 250.0)
        self.declare_parameter("point_cloud_maximum_depth_mm", 4000.0)
        self.declare_parameter("mirror_display", True)
        self.declare_parameter("display_width", 1440)
        self.declare_parameter("display_height", 360)
        self.declare_parameter("status_panel_height", 184)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("feedback_image_dir", str(default_feedback_dir))
        self.declare_parameter("score_topic", "/sandbag/form/score")
        self.declare_parameter("fist_coordinate_topic", "/sandbag/fist_coordinates")
        self.declare_parameter(
            "image_topic", "/sandbag/form/joint_evidence/compressed"
        )

        requested_left_device = str(self.get_parameter("left_device").value)
        requested_right_device = str(self.get_parameter("right_device").value)
        self.swap_webcams = bool(self.get_parameter("swap_webcams").value)
        role_map_path = Path(str(self.get_parameter("camera_role_map").value))
        use_role_map = (
            requested_left_device.lower() == "auto"
            and requested_right_device.lower() == "auto"
            and role_map_path.is_file()
        )
        if use_role_map:
            self.left_device, self.right_device = load_camera_role_devices(role_map_path)
            devices_auto_detected = False
            if self.swap_webcams:
                self.left_device, self.right_device = self.right_device, self.left_device
        else:
            (
                self.left_device,
                self.right_device,
                devices_auto_detected,
            ) = resolve_stereo_webcam_devices(
                requested_left_device,
                requested_right_device,
                self.swap_webcams,
            )
        calibration_path = Path(str(self.get_parameter("calibration_path").value))
        reference_path = Path(str(self.get_parameter("reference_path").value))
        self.capture_fps = float(self.get_parameter("capture_fps").value)
        self.processing_fps = float(self.get_parameter("processing_fps").value)
        self.max_sync_spread_ms = float(
            self.get_parameter("max_sync_spread_ms").value
        )
        self.pose_model = Path(str(self.get_parameter("pose_model").value)).expanduser()
        self.pose_device = str(self.get_parameter("pose_device").value)
        self.pose_image_size = max(
            int(self.get_parameter("pose_image_size").value), 256
        )
        self.pose_detection_confidence = float(
            self.get_parameter("pose_detection_confidence").value
        )
        self.pose_maximum_detections = max(
            int(self.get_parameter("pose_maximum_detections").value), 1
        )
        self.min_landmark_visibility = float(
            self.get_parameter("min_landmark_visibility").value
        )
        self.pose_association_max_reprojection_px = float(
            self.get_parameter("pose_association_max_reprojection_px").value
        )
        self.pose_target_lost_timeout_frames = max(
            int(self.get_parameter("pose_target_lost_timeout_frames").value), 1
        )
        self.max_reprojection_error_px = float(
            self.get_parameter("max_reprojection_error_px").value
        )
        self.fist_max_reprojection_error_px = float(
            self.get_parameter("fist_max_reprojection_error_px").value
        )
        self.depth_fusion_weight = float(
            self.get_parameter("depth_fusion_weight").value
        )
        self.max_depth_disagreement_mm = float(
            self.get_parameter("max_depth_disagreement_mm").value
        )
        self.depth_patch_radius = max(
            int(self.get_parameter("depth_patch_radius").value), 0
        )
        self.missing_pose_reset_frames = max(
            int(self.get_parameter("missing_pose_reset_frames").value), 1
        )
        self.diagnostic_log_interval_s = max(
            float(self.get_parameter("diagnostic_log_interval_s").value), 0.0
        )
        self.camera_startup_timeout_s = max(
            float(self.get_parameter("camera_startup_timeout_s").value), 1.0
        )
        self.display = bool(self.get_parameter("display").value)
        self.map_display = bool(self.get_parameter("map_display").value)
        self.map_width = max(int(self.get_parameter("map_width").value), 640)
        self.map_height = max(int(self.get_parameter("map_height").value), 480)
        self.person_point_cloud = bool(self.get_parameter("person_point_cloud").value)
        self.point_cloud_stride = max(int(self.get_parameter("point_cloud_stride").value), 1)
        self.segmentation_threshold = float(self.get_parameter("segmentation_threshold").value)
        self.point_cloud_depth_band_mm = max(
            float(self.get_parameter("point_cloud_depth_band_mm").value), 1.0
        )
        self.point_cloud_bbox_margin = max(
            float(self.get_parameter("point_cloud_bbox_margin").value), 0.0
        )
        self.point_cloud_minimum_depth_mm = float(self.get_parameter("point_cloud_minimum_depth_mm").value)
        self.point_cloud_maximum_depth_mm = float(self.get_parameter("point_cloud_maximum_depth_mm").value)
        self.mirror_display = bool(self.get_parameter("mirror_display").value)
        self.display_width = max(int(self.get_parameter("display_width").value), 960)
        self.display_height = max(
            int(self.get_parameter("display_height").value), 240
        )
        self.status_panel_height = max(
            int(self.get_parameter("status_panel_height").value), 180
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.feedback_image_dir = Path(
            str(self.get_parameter("feedback_image_dir").value)
        ).expanduser()
        score_topic = str(self.get_parameter("score_topic").value)
        fist_coordinate_topic = str(
            self.get_parameter("fist_coordinate_topic").value
        )
        image_topic = str(self.get_parameter("image_topic").value)

        self.calibration = load_three_camera_calibration(calibration_path)
        self.frame_width = self.calibration.image_width
        self.frame_height = self.calibration.image_height
        self.reference = load_reference(reference_path)
        self.detector = PunchDetector3D(self.reference["detector"])
        self.save_score_threshold = float(
            self.reference["feedback"].get("save_score_threshold", 30.0)
        )
        self.pass_score_threshold = float(
            self.reference["feedback"].get("pass_score_threshold", 30.0)
        )

        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "pyrealsense2가 없습니다. 먼저 ./setup_3d.sh를 실행하세요."
            ) from error
        self.rs = rs
        self.pose_backend = YoloPoseBackend(
            self.pose_model,
            device=self.pose_device,
            image_size=self.pose_image_size,
            detector_confidence=self.pose_detection_confidence,
            maximum_detections=self.pose_maximum_detections,
        )
        self.pose_selector = CalibratedPoseSelector(
            keypoint_confidence=self.min_landmark_visibility,
            maximum_reprojection_error_px=self.pose_association_max_reprojection_px,
            lost_timeout_frames=self.pose_target_lost_timeout_frames,
        )

        self.left_camera: LatestUsbCamera | None = None
        self.right_camera: LatestUsbCamera | None = None
        self.pipeline: Any | None = None
        try:
            self.left_camera = LatestUsbCamera(
                self.left_device,
                self.frame_width,
                self.frame_height,
                self.capture_fps,
            )
            self.right_camera = LatestUsbCamera(
                self.right_device,
                self.frame_width,
                self.frame_height,
                self.capture_fps,
            )
            self.pipeline = self._start_realsense()
        except Exception:
            self._close_cameras()
            self.pose_backend.close()
            raise

        event_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.score_publisher = self.create_publisher(String, score_topic, event_qos)
        self.fist_coordinate_publisher = self.create_publisher(
            String, fist_coordinate_topic, event_qos
        )
        self.image_publisher = self.create_publisher(
            CompressedImage, image_topic, image_qos
        )

        self.window_name = "Sandbag Punch Feedback 3D"
        self.map_window_name = "M0609 Robot Base 3D Map"
        if self.display:
            cv2.namedWindow(
                self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
            )
            cv2.resizeWindow(
                self.window_name,
                self.display_width,
                self.display_height
                + self.status_panel_height
                + CAMERA_LABEL_HEIGHT,
            )
            if self.map_display:
                cv2.namedWindow(
                    self.map_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
                )
                cv2.resizeWindow(self.map_window_name, self.map_width, self.map_height)

        self.punch_id = 0
        self.missing_pose_frames = 0
        self.sync_drop_count = 0
        self.synced_triplet_count = 0
        self.pose_3d_sample_count = 0
        self.last_diagnostic_log_at = time.monotonic()
        self.camera_started_at = time.monotonic()
        self.last_sync_spread_ms = 0.0
        self.last_reprojection_rms_px = 0.0
        self.last_depth_fused_count = 0
        self.last_pose_detected = {
            camera: False for camera in CAMERA_DISPLAY_ORDER
        }
        self.last_people_count = {camera: 0 for camera in CAMERA_DISPLAY_ORDER}
        self.last_3d_sample_valid = False
        self.latest_3d_sample: PoseSample3D | None = None
        self.last_3d_failure_reason = "waiting"
        self.last_result_display: dict[str, Any] | None = None
        self.impact_overlay: dict[str, Any] | None = None
        self.timer = self.create_timer(
            1.0 / max(self.processing_fps, 1.0), self.on_frame
        )
        self.get_logger().info(
            "3D MVP ready: front=RealSense, "
            f"left={self.left_device}, right={self.right_device}, "
            f"calibration={calibration_path}"
        )
        self.get_logger().info(
            "YOLO11n-Pose enabled: "
            f"model={self.pose_backend.model_path.name}, "
            f"device={self.pose_backend.device}, imgsz={self.pose_image_size}, "
            f"keypoint_conf={self.min_landmark_visibility:.2f}"
        )
        if self.pose_backend.device_fallback_reason is not None:
            self.get_logger().warning(
                "YOLO auto device fallback to CPU: "
                + self.pose_backend.device_fallback_reason
            )
        if devices_auto_detected:
            self.get_logger().info(
                "C270 devices were auto-detected by physical USB path"
                + (" with left/right swapped" if self.swap_webcams else "")
            )
        elif use_role_map:
            self.get_logger().info(f"C270 roles loaded from {role_map_path}")
        self.get_logger().info(
            f"Topics: {score_topic}, {fist_coordinate_topic}, {image_topic}"
        )

    def _start_realsense(self):
        pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_stream(
            self.rs.stream.color,
            self.frame_width,
            self.frame_height,
            self.rs.format.bgr8,
            int(self.capture_fps),
        )
        config.enable_stream(
            self.rs.stream.depth,
            self.frame_width,
            self.frame_height,
            self.rs.format.z16,
            int(self.capture_fps),
        )
        profile = pipeline.start(config)
        self.align_to_color = self.rs.align(self.rs.stream.color)
        color_profile = profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()
        self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
        return pipeline

    def _close_cameras(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
        if self.left_camera is not None:
            self.left_camera.close()
            self.left_camera = None
        if self.right_camera is not None:
            self.right_camera.close()
            self.right_camera = None

    def _get_frames(self) -> tuple[dict[str, TimedFrame], Any] | None:
        if self.pipeline is None or self.left_camera is None or self.right_camera is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=700)
        except RuntimeError as error:
            self.get_logger().warning(f"RealSense frame read failed: {error}")
            return None
        aligned = self.align_to_color.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        front = TimedFrame(np.asanyarray(color_frame.get_data()).copy(), time.monotonic())
        left = self.left_camera.latest()
        right = self.right_camera.latest()
        if left is None or right is None:
            if time.monotonic() - self.camera_started_at >= self.camera_startup_timeout_s:
                missing = []
                if left is None:
                    missing.append(f"left={self.left_device}")
                if right is None:
                    missing.append(f"right={self.right_device}")
                raise RuntimeError(
                    "웹캠 프레임 시작 실패: "
                    + ", ".join(missing)
                    + ". 두 웹캠을 서로 다른 USB 루트 버스에 연결하고 "
                    "장치 경로를 다시 확인하세요."
                )
            return None
        timed = {"left": left, "front": front, "right": right}
        stamps = [item.stamp_s for item in timed.values()]
        spread_ms = (max(stamps) - min(stamps)) * 1000.0
        self.last_sync_spread_ms = spread_ms
        if spread_ms > self.max_sync_spread_ms:
            self.sync_drop_count += 1
            return None
        return timed, depth_frame

    def _depth_point_mm(
        self,
        depth_frame: Any,
        observation: LandmarkObservation2D | None,
    ) -> np.ndarray | None:
        if observation is None:
            return None
        depth_m = median_depth_m(
            depth_frame,
            observation.pixel,
            self.depth_patch_radius,
            self.frame_width,
            self.frame_height,
        )
        if depth_m is None:
            return None
        point_m = self.rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics,
            [float(observation.pixel[0]), float(observation.pixel[1])],
            depth_m,
        )
        return np.asarray(point_m, dtype=np.float64) * 1000.0

    def _make_sample(
        self,
        timed_frames: Mapping[str, TimedFrame],
        depth_frame: Any,
        pose_detections: Mapping[str, YoloPoseDetection | None],
        front_pose: Any | None,
    ) -> PoseSample3D | None:
        if front_pose is None:
            self.last_3d_failure_reason = "front_pose"
            return None
        observations_by_camera = {
            camera: landmark_observations(
                pose_detections[camera],
                self.frame_width,
                self.frame_height,
                self.min_landmark_visibility,
            )
            for camera in ("left", "front", "right")
        }
        landmarks: dict[str, Landmark3D] = {}
        depth_fused_count = 0
        for name in LANDMARK_NAMES_3D:
            joint_observations = {
                camera: camera_observations[name]
                for camera, camera_observations in observations_by_camera.items()
                if name in camera_observations
            }
            triangulated = triangulate_landmark(
                self.calibration,
                joint_observations,
                self.max_reprojection_error_px,
            )
            if triangulated is None:
                self.last_3d_failure_reason = (
                    f"{name}(obs={len(joint_observations)})"
                )
                return None
            depth_point = self._depth_point_mm(
                depth_frame, observations_by_camera["front"].get(name)
            )
            xyz_mm, depth_used = fuse_realsense_depth(
                triangulated.xyz_mm,
                depth_point,
                self.depth_fusion_weight,
                self.max_depth_disagreement_mm,
            )
            depth_fused_count += int(depth_used)
            landmarks[name] = Landmark3D(
                *xyz_mm,
                confidence=triangulated.confidence,
                reprojection_error_px=triangulated.reprojection_rms_px,
                camera_count=len(triangulated.cameras),
            )

        self.last_reprojection_rms_px = float(
            np.mean([point.reprojection_error_px for point in landmarks.values()])
        )
        self.last_depth_fused_count = depth_fused_count
        self.last_3d_failure_reason = "none"
        return PoseSample3D(
            stamp_s=timed_frames["front"].stamp_s,
            landmarks=landmarks,
            front_pose=front_pose,
            sync_spread_ms=self.last_sync_spread_ms,
            depth_fused_count=depth_fused_count,
        )

    def on_frame(self) -> None:
        captured = self._get_frames()
        if captured is None:
            self.log_runtime_diagnostics()
            return
        self.synced_triplet_count += 1
        timed_frames, depth_frame = captured
        pose_candidates = self.pose_backend.infer(
            {
                camera: timed_frames[camera].image
                for camera in CAMERA_DISPLAY_ORDER
            }
        )
        pose_detections = self.pose_selector.select(
            pose_candidates,
            self.calibration,
            self.frame_width,
            self.frame_height,
        )
        self.last_people_count = {
            camera: len(pose_candidates[camera]) for camera in CAMERA_DISPLAY_ORDER
        }
        depth_raw = np.asanyarray(depth_frame.get_data())
        person_points_base = np.empty((0, 3), dtype=np.float64)
        person_colors = np.empty((0, 3), dtype=np.uint8)
        if (
            self.person_point_cloud
            and self.calibration.T_base_front_mm is not None
        ):
            person_mask = depth_mask_from_pose_box(
                depth_raw,
                pose_detections["front"],
                self.depth_scale_m,
                minimum_keypoint_confidence=self.min_landmark_visibility,
                depth_band_mm=self.point_cloud_depth_band_mm,
                bbox_margin_ratio=self.point_cloud_bbox_margin,
            )
            person_points_base, person_colors = colored_person_point_cloud_base(
                timed_frames["front"].image,
                depth_raw,
                person_mask,
                self.color_intrinsics,
                self.depth_scale_m,
                self.calibration.T_base_front_mm,
                stride=self.point_cloud_stride,
                segmentation_threshold=self.segmentation_threshold,
                minimum_depth_mm=self.point_cloud_minimum_depth_mm,
                maximum_depth_mm=self.point_cloud_maximum_depth_mm,
            )
        self.last_pose_detected = {
            camera: pose_detections[camera] is not None
            for camera in CAMERA_DISPLAY_ORDER
        }
        front_pose = (
            pose_detections["front"].to_pose_sample(
                timed_frames["front"].stamp_s,
                timed_frames["front"].image,
            )
            if pose_detections["front"] is not None
            else None
        )
        self.publish_detected_wrist_coordinates(
            timed_frames, depth_frame, pose_detections
        )
        sample = self._make_sample(
            timed_frames,
            depth_frame,
            pose_detections,
            front_pose,
        )
        self.last_3d_sample_valid = sample is not None
        if sample is not None:
            self.latest_3d_sample = sample
        if sample is None:
            self.missing_pose_frames += 1
            if self.missing_pose_frames >= self.missing_pose_reset_frames:
                self.detector.reset()
        else:
            self.missing_pose_frames = 0
            self.pose_3d_sample_count += 1
            event = self.detector.update(sample)
            if event is not None:
                self.handle_punch(event)

        if self.display:
            display_samples = {"front": front_pose}
            for camera in ("left", "right"):
                detection = pose_detections[camera]
                display_samples[camera] = (
                    detection.to_pose_sample(
                        timed_frames[camera].stamp_s,
                        timed_frames[camera].image,
                    )
                    if detection is not None
                    else None
                )
            camera_views: dict[str, np.ndarray] = {}
            for camera in CAMERA_DISPLAY_ORDER:
                camera_view = (
                    cv2.flip(timed_frames[camera].image, 1)
                    if self.mirror_display
                    else timed_frames[camera].image.copy()
                )
                pose_sample = display_samples[camera]
                if pose_sample is not None:
                    draw_pose(camera_view, pose_sample, self.mirror_display)
                camera_views[camera] = camera_view
            self.draw_cooldown_effect(camera_views["front"])
            cv2.imshow(self.window_name, self.build_display_frame(camera_views))
            if self.map_display:
                cv2.imshow(
                    self.map_window_name,
                    render_robot_base_3d_map(
                        self.calibration,
                        self.latest_3d_sample if sample is not None else None,
                        self.map_width,
                        self.map_height,
                        person_points_base,
                        person_colors,
                    ),
                )
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in (ord("r"), ord("R")):
                self.pose_selector.reset()
                self.detector.reset()
                self.get_logger().info("YOLO boxer target lock reset by user")
            elif pressed_key in (ord("q"), 27):
                rclpy.shutdown()
        self.log_runtime_diagnostics()

    def publish_detected_wrist_coordinates(
        self,
        timed_frames: Mapping[str, TimedFrame],
        depth_frame: Any,
        pose_detections: Mapping[str, YoloPoseDetection | None],
    ) -> None:
        """Publish wrists independently of full-body 3D reconstruction."""
        observations_by_camera = {
            camera: landmark_observations(
                pose_detections[camera],
                self.frame_width,
                self.frame_height,
                self.min_landmark_visibility,
            )
            for camera in CAMERA_DISPLAY_ORDER
        }
        wrists: dict[str, Any] = {}
        for side in ("left", "right"):
            name = f"{side}_wrist"
            joint_observations = {
                camera: observations[name]
                for camera, observations in observations_by_camera.items()
                if name in observations
            }
            triangulated = triangulate_landmark(
                self.calibration,
                joint_observations,
                self.fist_max_reprojection_error_px,
            )
            if triangulated is None:
                continue
            depth_point = self._depth_point_mm(
                depth_frame, observations_by_camera["front"].get(name)
            )
            xyz_mm, depth_used = fuse_realsense_depth(
                triangulated.xyz_mm,
                depth_point,
                self.depth_fusion_weight,
                self.max_depth_disagreement_mm,
            )
            front_xyz = {
                "x_mm": round(float(xyz_mm[0]), 2),
                "y_mm": round(float(xyz_mm[1]), 2),
                "z_mm": round(float(xyz_mm[2]), 2),
            }
            wrist_payload = {
                **front_xyz,
                "confidence": round(float(triangulated.confidence), 4),
                "camera_count": int(len(triangulated.cameras)),
                "reprojection_error_px": round(
                    float(triangulated.reprojection_rms_px), 3
                ),
                "depth_fused": bool(depth_used),
            }
            if self.calibration.T_base_front_mm is not None:
                base_xyz = self.calibration.front_point_to_base(xyz_mm)
                wrist_payload["base_mm"] = {
                    "x": round(float(base_xyz[0]), 2),
                    "y": round(float(base_xyz[1]), 2),
                    "z": round(float(base_xyz[2]), 2),
                }
            wrists[side] = wrist_payload
        if not wrists:
            return
        message = String()
        message.data = json.dumps(
            {
                "coordinate_frame": "front_realsense_color_optical_frame",
                "units": "mm",
                "axis_convention": {"x": "right", "y": "down", "z": "forward"},
                "sync_spread_ms": round(float(self.last_sync_spread_ms), 2),
                "fist_proxy": "yolo11n_pose_wrist",
                "robot_base_frame": "Doosan DR_BASE",
                "wrists": wrists,
            },
            ensure_ascii=False,
        )
        self.fist_coordinate_publisher.publish(message)

    def log_runtime_diagnostics(self) -> None:
        if self.diagnostic_log_interval_s <= 0.0:
            return
        now = time.monotonic()
        if now - self.last_diagnostic_log_at < self.diagnostic_log_interval_s:
            return
        self.last_diagnostic_log_at = now
        left_stats = (0, 0)
        right_stats = (0, 0)
        if self.left_camera is not None:
            left_stats = self.left_camera.stats()
        if self.right_camera is not None:
            right_stats = self.right_camera.stats()
        telemetry = self.detector.telemetry
        self.get_logger().info(
            f"3D frames: synced={self.synced_triplet_count} "
            f"pose3d={self.pose_3d_sample_count} "
            f"missing_streak={self.missing_pose_frames} "
            f"sync_drops={self.sync_drop_count} "
            f"state={self.detector.state} "
            f"sync={self.last_sync_spread_ms:.1f}ms "
            f"reproj={self.last_reprojection_rms_px:.1f}px "
            f"depth={self.last_depth_fused_count}/9 "
            f"yolo=L{int(self.last_pose_detected['left'])}"
            f"F{int(self.last_pose_detected['front'])}"
            f"R{int(self.last_pose_detected['right'])} "
            f"target={self.pose_selector.state}/"
            f"{self.pose_selector.last_score:.3f} "
            f"3d={'OK' if self.last_3d_sample_valid else self.last_3d_failure_reason} "
            f"guard=L{telemetry['left_guard_speed']:.2f}/"
            f"{telemetry['left_guard_face_ratio']:.2f} "
            f"R{telemetry['right_guard_speed']:.2f}/"
            f"{telemetry['right_guard_face_ratio']:.2f} "
            f"V{telemetry['guard_quality']:.2f} "
            f"usb=L{left_stats[0]}/{left_stats[1]} "
            f"R{right_stats[0]}/{right_stats[1]}"
        )

    def build_camera_label_strip(self) -> np.ndarray:
        labels = np.full(
            (CAMERA_LABEL_HEIGHT, self.display_width, 3),
            22,
            dtype=np.uint8,
        )
        base_width = self.display_width // len(CAMERA_DISPLAY_ORDER)
        remaining_width = self.display_width - base_width * len(
            CAMERA_DISPLAY_ORDER
        )
        tile_left = 0
        for index, camera in enumerate(CAMERA_DISPLAY_ORDER):
            tile_width = base_width + int(index < remaining_width)
            label = CAMERA_DISPLAY_LABELS[camera]
            pose_detected = self.last_pose_detected[camera]
            pose_label = (
                f"YOLO OK ({self.last_people_count[camera]})"
                if pose_detected
                else f"YOLO NO TARGET ({self.last_people_count[camera]})"
            )
            pose_color = (60, 240, 60) if pose_detected else (40, 40, 255)
            cv2.putText(
                labels,
                label,
                (tile_left + 9, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            (pose_width, _), _ = cv2.getTextSize(
                pose_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                2,
            )
            cv2.putText(
                labels,
                pose_label,
                (tile_left + tile_width - pose_width - 10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                pose_color,
                2,
                cv2.LINE_AA,
            )
            tile_left += tile_width
            if index < len(CAMERA_DISPLAY_ORDER) - 1:
                cv2.line(
                    labels,
                    (tile_left, 0),
                    (tile_left, CAMERA_LABEL_HEIGHT - 1),
                    (230, 230, 230),
                    2,
                    cv2.LINE_AA,
                )
        return labels

    def build_display_frame(
        self,
        camera_views: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        camera_strip = compose_three_camera_strip(
            camera_views,
            self.display_width,
            self.display_height,
        )
        status = np.zeros(
            (self.status_panel_height, self.display_width, 3), dtype=np.uint8
        )
        self.draw_status(status)
        return np.vstack((status, self.build_camera_label_strip(), camera_strip))

    def draw_status(self, frame: np.ndarray) -> None:
        telemetry = self.detector.telemetry
        cv2.putText(
            frame,
            f"STATE: {self.detector.state} | MODE: 3D",
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        draw_guard_gauge(
            frame,
            self.detector.state,
            int(telemetry["guard_frames"]),
            int(telemetry["ready_frames"]),
        )
        for row, side in enumerate(("left", "right")):
            if self.detector.state == "WAIT_GUARD":
                line = (
                    f"{side[0].upper()} GSPD "
                    f"{telemetry[f'{side}_guard_speed']:.2f}/"
                    f"{telemetry['guard_speed_limit']:.2f} | "
                    f"FACE {telemetry[f'{side}_guard_face_ratio']:.2f}/"
                    f"{telemetry['guard_face_limit']:.2f} | "
                    f"VIS {telemetry['guard_quality']:.2f}/"
                    f"{telemetry['guard_quality_limit']:.2f}"
                )
                guard_ok = (
                    telemetry[f"{side}_guard_speed"]
                    <= telemetry["guard_speed_limit"]
                    and telemetry[f"{side}_guard_face_ratio"]
                    <= telemetry["guard_face_limit"]
                    and telemetry["guard_quality"]
                    >= telemetry["guard_quality_limit"]
                )
                line_color = (80, 255, 80) if guard_ok else (0, 180, 255)
            else:
                line = (
                    f"{side[0].upper()} SPD {telemetry[f'{side}_speed']:.2f} | "
                    f"MOVE {telemetry[f'{side}_displacement']:.2f} | "
                    f"ELB+ {telemetry[f'{side}_elbow_delta']:.1f} | "
                    f"START {int(telemetry[f'{side}_start_confirm'])}/"
                    f"{int(telemetry['start_confirm_frames'])}"
                )
                line_color = (230, 230, 230)
            cv2.putText(
                frame,
                line,
                (12, 57 + row * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                line_color,
                1,
                cv2.LINE_AA,
            )
        pose_state = " ".join(
            f"{camera[0].upper()}:{'OK' if self.last_pose_detected[camera] else '--'}"
            for camera in CAMERA_DISPLAY_ORDER
        )
        sample_state = (
            "OK"
            if self.last_3d_sample_valid
            else f"MISS:{self.last_3d_failure_reason}"
        )
        cv2.putText(
            frame,
            (
                f"YOLO {pose_state} | TARGET {self.pose_selector.state} | "
                f"3D {sample_state} | "
                f"MISSING {self.missing_pose_frames}"
            ),
            (12, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (80, 255, 80) if self.last_3d_sample_valid else (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"SYNC {self.last_sync_spread_ms:.0f} ms | "
                f"REPROJ {self.last_reprojection_rms_px:.1f} px | "
                f"DEPTH {self.last_depth_fused_count}/9 | DROPS {self.sync_drop_count}"
            ),
            (12, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )
        if self.last_result_display is None:
            text = "No punch yet"
            color = (255, 255, 255)
        else:
            result = self.last_result_display
            text = (
                f"#{result['punch_id']} {result['side']} "
                f"{str(result['punch_type']).upper()} score={result['score']:.1f} "
                f"image={result['image_published']} saved={result['image_saved']}"
            )
            color = PUNCH_COLOR_BGR.get(result["punch_type"], (255, 255, 255))
        cv2.putText(
            frame,
            text,
            (12, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            2 if self.last_result_display else 1,
            cv2.LINE_AA,
        )

    def draw_cooldown_effect(self, frame: np.ndarray) -> None:
        if self.detector.state != "COOLDOWN" or self.impact_overlay is None:
            return
        overlay = self.impact_overlay
        height, width = frame.shape[:2]
        x_norm = float(overlay["x"])
        if self.mirror_display:
            x_norm = 1.0 - x_norm
        x = int(np.clip(x_norm, 0.0, 1.0) * (width - 1))
        y = int(np.clip(float(overlay["y"]), 0.0, 1.0) * (height - 1))
        color = PUNCH_COLOR_BGR.get(overlay["punch_type"], (0, 255, 255))
        duration = max(float(self.detector.cooldown_s), 0.1)
        progress = float(
            np.clip((time.monotonic() - overlay["started_at"]) / duration, 0.0, 1.0)
        )
        radius = 20 + int(32 * progress)
        cv2.circle(frame, (x, y), radius, color, 4, cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(7, radius // 3), (255, 255, 255), 3)
        label = "GOOD!" if overlay["passed"] else "BAD.."
        label_color = (80, 255, 80) if overlay["passed"] else (40, 40, 255)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        text_x = int(np.clip(x - text_width // 2, 8, max(8, width - text_width - 8)))
        text_y = min(y + radius + text_height + 14, height - baseline - 8)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            label_color,
            2,
            cv2.LINE_AA,
        )

    def _render_3d_evidence(
        self,
        classified: ClassifiedPunch3D,
        result: ScoreResult,
    ) -> np.ndarray:
        if classified.key_sample.front_pose is None:
            raise ValueError("3D key sample has no front-camera pose")
        display_classification = ClassifiedPunch(
            punch_type=classified.punch_type,
            side=classified.side,
            confidence=classified.confidence,
            key_sample=classified.key_sample.front_pose,
            motion_features=classified.motion_features,
            classification_reason=classified.classification_reason,
        )
        return render_evidence(
            display_classification,
            result,
            mirror=self.mirror_display,
            reference_label="임시 3카메라 3D 정자세 기준",
        )

    def handle_punch(self, event: PunchEvent3D) -> None:
        classified = classify_punch_3d(event, self.reference["classification"])
        result = score_punch_3d(
            classified, self.reference["profiles"], self.reference["feedback"]
        )
        self.punch_id += 1
        now = self.get_clock().now().to_msg()
        stamp_ns = int(now.sec) * 1_000_000_000 + int(now.nanosec)
        payload = {
            "stamp_ns": stamp_ns,
            "punch_id": self.punch_id,
            "mode": "three_camera_3d",
            "coordinate_frame": "front_realsense_color_optical_frame",
            "punch_type": classified.punch_type,
            "punch_side": classified.side,
            "classification_confidence": round(classified.confidence, 4),
            "classification_reason": classified.classification_reason,
            "candidate_scores": {
                key: round(value, 4)
                for key, value in classified.candidate_scores.items()
            },
            "total_score": round(result.total_score, 2),
            "passed": result.total_score >= self.pass_score_threshold,
            "feedback_required": result.feedback_required,
            "image_save_required": result.total_score < self.save_score_threshold,
            "motion_features": {
                key: round(value, 4)
                for key, value in classified.motion_features.items()
            },
            "quality": {
                "impact_sync_spread_ms": round(
                    classified.key_sample.sync_spread_ms, 2
                ),
                "impact_depth_fused_joints": classified.key_sample.depth_fused_count,
                "impact_mean_reprojection_error_px": round(
                    float(
                        np.mean(
                            [
                                point.reprojection_error_px
                                for point in classified.key_sample.landmarks.values()
                            ]
                        )
                    ),
                    3,
                ),
                "impact_min_camera_count": min(
                    point.camera_count
                    for point in classified.key_sample.landmarks.values()
                ),
            },
            "violations": [feature_error_dict(item) for item in result.violations],
            "all_feature_errors": [
                feature_error_dict(item) for item in result.feature_errors
            ],
        }
        impact_xyz_front = np.asarray(
            classified.key_sample.landmarks[f"{classified.side}_wrist"].xyz,
            dtype=np.float64,
        )
        payload["impact_point"] = {
            "front_camera_optical_mm": {
                "x": round(float(impact_xyz_front[0]), 2),
                "y": round(float(impact_xyz_front[1]), 2),
                "z": round(float(impact_xyz_front[2]), 2),
            }
        }
        if self.calibration.T_base_front_mm is not None:
            impact_xyz_base = self.calibration.front_point_to_base(impact_xyz_front)
            payload["impact_point"]["robot_base_mm"] = {
                "x": round(float(impact_xyz_base[0]), 2),
                "y": round(float(impact_xyz_base[1]), 2),
                "z": round(float(impact_xyz_base[2]), 2),
            }
        score_message = String()
        score_message.data = json.dumps(payload, ensure_ascii=False)
        self.score_publisher.publish(score_message)

        image_published = False
        image_saved = False
        image_save_required = result.total_score < self.save_score_threshold
        if result.feedback_required or image_save_required:
            evidence = self._render_3d_evidence(classified, result)
            encoded, jpeg = cv2.imencode(
                ".jpg", evidence, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if encoded:
                jpeg_bytes = jpeg.tobytes()
                if result.feedback_required:
                    image_message = CompressedImage()
                    image_message.header.stamp = now
                    image_message.header.frame_id = (
                        "front_realsense_color_optical_frame"
                    )
                    image_message.format = "jpeg"
                    image_message.data = jpeg_bytes
                    self.image_publisher.publish(image_message)
                    image_published = True
                if image_save_required:
                    image_saved = self.save_feedback_image(
                        jpeg_bytes, classified, result
                    ) is not None
            else:
                self.get_logger().error("Failed to encode 3D evidence JPEG")

        front_pose = classified.key_sample.front_pose
        impact_wrist = front_pose.landmarks[f"{classified.side}_wrist"]
        self.impact_overlay = {
            "x": impact_wrist.x,
            "y": impact_wrist.y,
            "punch_type": classified.punch_type,
            "passed": result.total_score >= self.pass_score_threshold,
            "started_at": time.monotonic(),
        }
        self.last_result_display = {
            "punch_id": self.punch_id,
            "side": classified.side,
            "punch_type": classified.punch_type,
            "score": result.total_score,
            "image_published": image_published,
            "image_saved": image_saved,
        }
        motion = classified.motion_features
        self.get_logger().info(
            f"#{self.punch_id} {classified.side} {classified.punch_type} "
            f"score={result.total_score:.1f} image={image_published} "
            f"saved={image_saved} forward={motion['forward_component_ratio']:.2f} "
            f"lateral={motion['lateral_component_ratio']:.2f} "
            f"up={motion['upward_component_ratio']:.2f} "
            f"curve={motion['path_curvature_ratio']:.2f}"
        )

    def save_feedback_image(
        self,
        jpeg_bytes: bytes,
        classified: ClassifiedPunch3D,
        result: ScoreResult,
    ) -> Path | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.feedback_image_dir / (
            f"{timestamp}_punch_{self.punch_id:04d}_{classified.side}_"
            f"{classified.punch_type}_score_{result.total_score:.1f}_3d.jpg"
        )
        try:
            self.feedback_image_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jpeg_bytes)
        except OSError as error:
            self.get_logger().error(f"Failed to save 3D feedback image: {error}")
            return None
        self.get_logger().info(f"Saved low-score 3D feedback image: {path}")
        return path

    def destroy_node(self) -> bool:
        if hasattr(self, "timer"):
            self.timer.cancel()
        self._close_cameras()
        if hasattr(self, "pose_backend"):
            self.pose_backend.close()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: ThreeCameraPunchFeedbackNode | None = None
    try:
        node = ThreeCameraPunchFeedbackNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"[FATAL] {error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

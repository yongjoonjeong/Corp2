from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    from rclpy.exceptions import RCLError
except ImportError:  # ROS 2 Humble exports RCLError from the native module.
    from rclpy._rclpy_pybind11 import RCLError

from .calibration import load_robot_world
from .capture import CameraHub
from .ekf import ConstantVelocityEkf
from .impact import FrontImpactDetector2D
from .mitt import MittTrack, RedMittTracker
from .pose import (
    LatestPoseWorker,
    draw_pose,
    expand_and_clip_roi,
    map_original_bbox_to_rotated,
    rotate_image,
)
from .snapshot import ImpactSnapshotWriter, SnapshotResult
from .target import LowRateBoxerWorker
from .triangulation import (
    PoseHistory,
    align_pose_histories,
    fuse_front_depth,
    median_depth_point_base,
    triangulate_robust,
)
from .types import CAMERAS, FistState, ImpactEvent, Landmark2D, SIDES, TriangulationResult


FRONT_READY_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


def _front_pose_ready_status(
    sample: Any,
    now_ns: int,
    minimum_confidence: float,
    maximum_age_ms: float,
) -> tuple[bool, float, float | None]:
    """Return whether the fresh FRONT sample contains every training joint."""
    if sample is None:
        return False, 0.0, None
    age_ms = max(0.0, (int(now_ns) - int(sample.stamp_ns)) / 1e6)
    confidences = [
        float(sample.landmarks[name].confidence)
        for name in FRONT_READY_LANDMARKS
        if name in sample.landmarks
    ]
    confidence = min(confidences) if len(confidences) == len(FRONT_READY_LANDMARKS) else 0.0
    ready = age_ms <= float(maximum_age_ms) and confidence >= float(minimum_confidence)
    return ready, confidence, age_ms


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _stamp_message(header: Any, stamp_ns: int, frame_id: str = "base") -> None:
    header.stamp.sec = int(stamp_ns // 1_000_000_000)
    header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    header.frame_id = frame_id


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _fist_state_overlay_line(
    side: str,
    state: FistState | None,
) -> tuple[str, tuple[int, int, int]]:
    label = side.upper()
    if state is None:
        return (
            f"{label:<5} WAITING | target lock and wrist landmarks from 2+ cameras required",
            (170, 170, 170),
        )
    position = ",".join(f"{float(value):7.0f}" for value in state.position_base_mm)
    velocity = ",".join(f"{float(value):7.0f}" for value in state.velocity_base_mm_s)
    status = "VALID" if state.valid else "STALE"
    color = (80, 230, 80) if state.valid else (0, 180, 255)
    return (
        f"{label:<5} {status:<5} | P[{position}] mm | V[{velocity}] mm/s | "
        f"age {state.measurement_age_ms:5.0f} ms | std {state.position_std_mm:4.0f} mm | "
        f"cams {state.camera_count} | conf {state.confidence:.2f}",
        color,
    )


def _append_fist_state_panel(canvas: np.ndarray, states: dict[str, FistState]) -> np.ndarray:
    panel = np.full((104, canvas.shape[1], 3), 18, dtype=np.uint8)
    cv2.line(panel, (0, 0), (panel.shape[1] - 1, 0), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "6-STATE EKF / BASE FRAME   P=(X,Y,Z) mm   V=(VX,VY,VZ) mm/s",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    for y, side in zip((55, 87), SIDES):
        text, color = _fist_state_overlay_line(side, states.get(side))
        cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return np.vstack((canvas, panel))


def _draw_runtime_status(
    canvas: np.ndarray,
    target_state: str,
    target_people: str,
    impact_state: str,
    guard: int,
    guard_goal: int,
    yolo_device: str,
    yolo_inference_ms: float,
) -> np.ndarray:
    height, width = canvas.shape[:2]
    bar_top = max(0, height - 38)
    cv2.rectangle(canvas, (0, bar_top), (width, height), (0, 0, 0), -1)
    baseline_y = height - 13
    prefix = f"TARGET {target_state} PPL {target_people} | "
    cv2.putText(
        canvas,
        prefix,
        (14, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    prefix_width = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
    state_colors = {
        "READY": (0, 255, 0),
        "ACTIVE": (0, 200, 255),
        "IMPACT": (0, 0, 255),
        "COOLDOWN": (255, 190, 80),
    }
    state_color = state_colors.get(impact_state, (0, 255, 255))
    cv2.putText(
        canvas,
        impact_state,
        (14 + prefix_width, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        state_color,
        2,
        cv2.LINE_AA,
    )
    impact_width = cv2.getTextSize(impact_state, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
    device_label = "CPU" if yolo_device.lower() == "cpu" else f"GPU:{yolo_device}"
    device_text = f" | YOLO {device_label} {yolo_inference_ms:.1f}ms"
    cv2.putText(
        canvas,
        device_text,
        (14 + prefix_width + impact_width, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    gauge_width = 250
    gauge_height = 18
    gauge_x = width - gauge_width - 54
    gauge_y = height - 28
    cv2.putText(
        canvas,
        "GUARD",
        (gauge_x - 88, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (gauge_x, gauge_y),
        (gauge_x + gauge_width, gauge_y + gauge_height),
        (110, 110, 110),
        1,
    )
    goal = max(int(guard_goal), 1)
    progress = float(np.clip(int(guard) / goal, 0.0, 1.0))
    fill_width = int(round((gauge_width - 4) * progress))
    if fill_width > 0:
        cv2.rectangle(
            canvas,
            (gauge_x + 2, gauge_y + 2),
            (gauge_x + 2 + fill_width, gauge_y + gauge_height - 2),
            (50, 210, 50),
            -1,
        )
    gauge_text = f"{min(max(int(guard), 0), goal)}/{goal}"
    text_size = cv2.getTextSize(gauge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0]
    cv2.putText(
        canvas,
        gauge_text,
        (gauge_x + (gauge_width - text_size[0]) // 2, gauge_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


class SandbagVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("sandbag_vision_realtime")
        default_config = Path(__file__).resolve().parents[1] / "config" / "runtime.yaml"
        self.declare_parameter("config_file", str(default_config))
        config_path = Path(str(self.get_parameter("config_file").value)).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.config_path = config_path
        calibration_path = _resolve_path(config_path, self.config["calibration"]["robot_world"])
        self.calibration = load_robot_world(calibration_path)
        self.workspace = self.config["target"]["workspace_base_mm"]

        camera_config = self.config["camera"]
        self.camera_hub = CameraHub(camera_config, config_path.parent)
        target_config = self.config["target"]
        target_model = _resolve_path(config_path, target_config["model"])
        pose_config = self.config["pose"]
        self.pose_rotations = dict(pose_config.get("rotations", {}))
        self.target_workers = {
            camera: LowRateBoxerWorker(
                target_model,
                self.calibration.cameras[camera],
                target_config,
                self.workspace,
                camera_name=camera,
                rotation=self.pose_rotations.get(camera, "none"),
            )
            for camera in CAMERAS
        }
        # Keep the existing name for the authoritative front-camera target.
        self.target_worker = self.target_workers["front"]
        pose_model = _resolve_path(config_path, pose_config["model"])
        self.pose_workers = {
            camera: LatestPoseWorker(
                camera,
                pose_model,
                pose_config,
                self.pose_rotations.get(camera, "none"),
            )
            for camera in CAMERAS
        }
        self.histories = {camera: PoseHistory(40) for camera in CAMERAS}
        self._last_submitted_sequence = {camera: -1 for camera in CAMERAS}
        self._last_pose_stamp = {camera: -1 for camera in CAMERAS}
        self._last_triangulation_stamp = -1
        self._last_target_id: int | None = None
        self._target_epoch_ns = 0
        self._side_rois = {
            camera: self.calibration.project_workspace(camera, self.workspace, margin_ratio=0.08)
            for camera in ("left", "right")
        }

        filter_config = self.config["filter"]
        self.filters = {
            side: ConstantVelocityEkf(
                filter_config["process_acceleration_std_mm_s2"],
                filter_config["initial_position_std_mm"],
                filter_config["initial_velocity_std_mm_s"],
                filter_config["maximum_innovation_sigma"],
            )
            for side in SIDES
        }
        self._last_quality: dict[str, TriangulationResult | None] = {side: None for side in SIDES}
        self.impact_detector = FrontImpactDetector2D(self.config["impact"])
        mitt_tracking_config = self.config["impact"].get("mitt_tracking", {})
        self.mitt_tracker = RedMittTracker(
            mitt_tracking_config if isinstance(mitt_tracking_config, dict) else {},
            self.config["impact"].get("mitt_roi_normalized"),
        )
        self._last_mitt_sequence = -1
        self._mitt_track = MittTrack(0, None, "STARTING", 0.0, 0.0)
        snapshot_config = self.config["snapshot"]
        output_root = Path(str(snapshot_config["output_directory"]))
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[1] / output_root
        self.snapshot_writer = ImpactSnapshotWriter(
            output_root,
            int(snapshot_config["jpeg_quality"]),
            list(snapshot_config["candidate_offsets_ms"]),
            self.pose_rotations,
            self.config["impact"].get("mitt_roi_normalized"),
        )
        self._snapshot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="impact-snapshot")
        self._snapshot_jobs: list[tuple[ImpactEvent, Future[SnapshotResult | None]]] = []

        ros_config = self.config["ros"]
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.fist_publisher = self.create_publisher(String, ros_config["fist_state_topic"], sensor_qos)
        self.position_publishers = {
            side: self.create_publisher(PointStamped, f"/sandbag/fist_position_base_mm/{side}", sensor_qos)
            for side in SIDES
        }
        self.velocity_publishers = {
            side: self.create_publisher(Vector3Stamped, f"/sandbag/fist_velocity_base_mm_s/{side}", sensor_qos)
            for side in SIDES
        }
        self.impact_publisher = self.create_publisher(String, ros_config["impact_event_topic"], event_qos)
        self.image_publisher = self.create_publisher(CompressedImage, ros_config["impact_image_topic"], event_qos)
        self.preview_publisher = self.create_publisher(
            CompressedImage,
            ros_config.get("preview_image_topic", "/sandbag/vision/preview/compressed"),
            sensor_qos,
        )
        self.front_preview_publisher = self.create_publisher(
            CompressedImage,
            ros_config.get("front_preview_image_topic", "/sandbag/vision/front/compressed"),
            sensor_qos,
        )
        self.status_publisher = self.create_publisher(String, ros_config["status_topic"], sensor_qos)
        self.show_preview = bool(ros_config.get("show_preview", True))
        self.show_fist_overlay = bool(ros_config.get("show_fist_overlay", True))
        self.publish_preview = bool(ros_config.get("publish_preview", True))
        self.publish_front_preview = bool(ros_config.get("publish_front_preview", True))
        self.preview_publish_period_ns = int(
            1e9 / max(float(ros_config.get("preview_publish_hz", 8.0)), 0.5)
        )
        self.front_preview_publish_period_ns = int(
            1e9 / max(float(ros_config.get("front_preview_publish_hz", 10.0)), 0.5)
        )
        self.status_publish_period_ns = int(
            1e9 / max(float(ros_config.get("status_publish_hz", 10.0)), 1.0)
        )
        self.preview_jpeg_quality = int(np.clip(ros_config.get("preview_jpeg_quality", 82), 40, 100))
        self.front_preview_jpeg_quality = int(
            np.clip(ros_config.get("front_preview_jpeg_quality", 82), 40, 100)
        )
        self.preview_rotations = dict(ros_config.get("preview_rotations", {}))
        self.publish_period_ns = int(1e9 / max(float(ros_config.get("publish_hz", 30.0)), 1.0))
        self._last_publish_ns = 0
        self._last_preview_publish_ns = 0
        self._last_front_preview_publish_ns = 0
        self._last_front_preview_sequence = -1
        self._last_status_ns = 0
        self._latest_states: dict[str, FistState] = {}
        self._closing = False
        self._reported_target_errors: dict[str, str] = {}

        self.camera_hub.start()
        self.timer = self.create_timer(1.0 / 90.0, self._tick)
        self.get_logger().info(
            f"new realtime pipeline started | calibration={calibration_path} | "
            f"left={self.camera_hub.devices['left']} right={self.camera_hub.devices['right']} | "
            f"yolo_device={self.target_worker.device} | botsort=left/front/right"
        )

    def _reset_target_dependent_state(self, new_target_id: int | None, epoch_ns: int) -> None:
        if new_target_id == self._last_target_id:
            return
        self._last_target_id = new_target_id
        self._target_epoch_ns = int(epoch_ns)
        for history in self.histories.values():
            history.clear()
        for state_filter in self.filters.values():
            state_filter.reset()
        self._last_quality = {side: None for side in SIDES}
        self._last_triangulation_stamp = -1
        self._latest_states.clear()
        self.impact_detector = FrontImpactDetector2D(self.config["impact"])
        if self._mitt_track.stamp_ns > 0:
            self.impact_detector.set_mitt_roi(
                self._mitt_track.stamp_ns,
                self._mitt_track.roi_normalized,
            )

    def _submit_frames(self, frames: dict) -> None:
        front = frames.get("front")
        if front is not None:
            if front.sequence != self._last_mitt_sequence:
                self._mitt_track = self.mitt_tracker.update(front.stamp_ns, front.color_bgr)
                self._last_mitt_sequence = front.sequence
                self.impact_detector.set_mitt_roi(
                    self._mitt_track.stamp_ns,
                    self._mitt_track.roi_normalized,
                )
            self.target_workers["front"].submit(front)
        target = self.target_worker.latest()
        preferred_torso = (
            target.torso_base_mm
            if target is not None
            and target.state in ("LOCKED", "TEMPORARILY_LOST")
            and target.torso_base_mm is not None
            else None
        )
        for camera in ("left", "right"):
            packet = frames.get(camera)
            if packet is not None:
                self.target_workers[camera].submit(packet, preferred_torso)
        for camera, worker in self.target_workers.items():
            if worker.error and worker.error != self._reported_target_errors.get(camera):
                self._reported_target_errors[camera] = worker.error
                self.get_logger().error(f"{camera} boxer target worker failed: {worker.error}")
        target_id = target.local_track_id if target is not None and target.state in ("LOCKED", "TEMPORARILY_LOST") else None
        self._reset_target_dependent_state(
            target_id,
            front.stamp_ns if front is not None else time.time_ns(),
        )
        if target_id is None or target is None or target.bbox_xyxy is None:
            return
        for camera in CAMERAS:
            packet = frames.get(camera)
            if packet is None or packet.sequence == self._last_submitted_sequence[camera]:
                continue
            if camera == "front":
                roi = expand_and_clip_roi(
                    target.bbox_xyxy,
                    packet.color_bgr.shape[1],
                    packet.color_bgr.shape[0],
                    float(self.config["target"].get("bbox_margin_ratio", 0.18)),
                )
            else:
                roi = None
                side_target = self.target_workers[camera].latest()
                if (
                    side_target is not None
                    and side_target.state in ("LOCKED", "TEMPORARILY_LOST")
                    and side_target.aligned_to_front
                    and side_target.bbox_xyxy is not None
                ):
                    roi = expand_and_clip_roi(
                        side_target.bbox_xyxy,
                        packet.color_bgr.shape[1],
                        packet.color_bgr.shape[0],
                        float(self.config["target"].get("bbox_margin_ratio", 0.18)),
                    )
                if roi is None and target.torso_base_mm is not None:
                    center = self.calibration.cameras[camera].project_base(target.torso_base_mm)
                    if np.all(np.isfinite(center)):
                        half_width = int(self.config["target"].get("side_roi_half_width_px", 280))
                        half_height = int(self.config["target"].get("side_roi_half_height_px", 360))
                        raw = (
                            int(round(center[0])) - half_width,
                            int(round(center[1])) - half_height,
                            int(round(center[0])) + half_width,
                            int(round(center[1])) + half_height,
                        )
                        if raw[2] > 0 and raw[3] > 0 and raw[0] < packet.color_bgr.shape[1] and raw[1] < packet.color_bgr.shape[0]:
                            roi = expand_and_clip_roi(
                                raw,
                                packet.color_bgr.shape[1],
                                packet.color_bgr.shape[0],
                                0.0,
                            )
                roi = roi or self._side_rois[camera] or (0, 0, packet.color_bgr.shape[1], packet.color_bgr.shape[0])
            self.pose_workers[camera].submit(packet, roi)
            self._last_submitted_sequence[camera] = packet.sequence

    def _collect_pose_results(self) -> None:
        for camera, worker in self.pose_workers.items():
            sample = worker.latest()
            if sample is None or sample.stamp_ns <= self._last_pose_stamp[camera]:
                continue
            self._last_pose_stamp[camera] = sample.stamp_ns
            if self._last_target_id is None or sample.stamp_ns < self._target_epoch_ns:
                continue
            self.histories[camera].add(sample)
            if camera == "front":
                event = self.impact_detector.update(sample)
                if event is not None:
                    future = self._snapshot_executor.submit(
                        self.snapshot_writer.save,
                        event,
                        self.camera_hub.rings,
                        self.histories,
                    )
                    self._snapshot_jobs.append((event, future))

    def _front_depth_point(self, stamp_ns: int, pixel: tuple[float, float]) -> np.ndarray | None:
        packet = self.camera_hub.nearest("front", stamp_ns, maximum_delta_ns=55_000_000)
        if packet is None:
            return None
        return median_depth_point_base(
            self.calibration,
            packet.depth_raw,
            packet.depth_scale_mm,
            pixel,
            radius=3,
        )

    def _triangulate(self, now_ns: int) -> None:
        pose_config = self.config["pose"]
        tri_config = self.config["triangulation"]
        aligned_result = align_pose_histories(
            self.histories,
            now_ns,
            int(float(pose_config["maximum_result_age_ms"]) * 1e6),
            int(float(tri_config["maximum_interpolation_gap_ms"]) * 1e6),
            int(float(tri_config["maximum_time_skew_ms"]) * 1e6),
        )
        if aligned_result is None:
            return
        reference_ns, samples = aligned_result
        if reference_ns <= self._last_triangulation_stamp:
            return
        self._last_triangulation_stamp = reference_ns
        minimum_confidence = float(self.config["pose"]["minimum_landmark_confidence"])
        for side in SIDES:
            wrist_observations: dict[str, Landmark2D] = {}
            elbow_observations: dict[str, Landmark2D] = {}
            for camera, sample in samples.items():
                wrist = sample.landmarks.get(f"{side}_wrist")
                elbow = sample.landmarks.get(f"{side}_elbow")
                if wrist is not None and wrist.confidence >= minimum_confidence:
                    wrist_observations[camera] = wrist
                if elbow is not None and elbow.confidence >= minimum_confidence:
                    elbow_observations[camera] = elbow
            if len(wrist_observations) < int(tri_config.get("minimum_cameras", 2)):
                continue
            wrist_result = triangulate_robust(
                self.calibration,
                wrist_observations,
                float(tri_config["maximum_reprojection_error_px"]),
                float(tri_config["minimum_ray_angle_deg"]),
            )
            if wrist_result is None:
                continue
            front_wrist = wrist_observations.get("front")
            if front_wrist is not None:
                depth_point = self._front_depth_point(reference_ns, front_wrist.pixel)
                wrist_result = fuse_front_depth(
                    wrist_result,
                    depth_point,
                    float(tri_config["depth_fusion_weight"]),
                    float(tri_config["maximum_depth_disagreement_mm"]),
                )
            point = wrist_result.point_base_mm.copy()
            elbow_result = None
            if len(elbow_observations) >= 2:
                elbow_result = triangulate_robust(
                    self.calibration,
                    elbow_observations,
                    float(tri_config["maximum_reprojection_error_px"]),
                    float(tri_config["minimum_ray_angle_deg"]),
                )
            if elbow_result is not None:
                direction = point - elbow_result.point_base_mm
                norm = float(np.linalg.norm(direction))
                if norm > 30.0:
                    point += float(tri_config["hand_extension_mm"]) * direction / norm
            accepted = self.filters[side].update(
                point,
                reference_ns,
                wrist_result.position_std_mm,
            )
            if accepted:
                self._last_quality[side] = wrist_result

    def _make_state(self, side: str, now_ns: int) -> FistState | None:
        estimate = self.filters[side].estimate(now_ns, float(self.config["filter"]["maximum_measurement_age_ms"]))
        if estimate is None:
            return None
        quality = self._last_quality[side]
        return FistState(
            side=side,
            stamp_ns=estimate.stamp_ns,
            position_base_mm=estimate.position_mm,
            velocity_base_mm_s=estimate.velocity_mm_s,
            position_std_mm=max(estimate.position_std_mm, quality.position_std_mm if quality else 999.0),
            measurement_age_ms=estimate.measurement_age_ms,
            reprojection_error_px=quality.reprojection_rms_px if quality else 999.0,
            confidence=quality.confidence if quality and estimate.valid else 0.0,
            camera_count=len(quality.cameras) if quality else 0,
            camera_mask=quality.camera_mask if quality else 0,
            minimum_ray_angle_deg=quality.minimum_ray_angle_deg if quality else 0.0,
            depth_used=bool(quality.depth_used) if quality else False,
            valid=estimate.valid and quality is not None,
        )

    def _publish_states(self, now_ns: int) -> None:
        if now_ns - self._last_publish_ns < self.publish_period_ns:
            return
        self._last_publish_ns = now_ns
        for side in SIDES:
            state = self._make_state(side, now_ns)
            if state is None:
                continue
            self._latest_states[side] = state
            payload = {
                "stamp_ns": state.stamp_ns,
                "frame_id": "base",
                "side": state.side,
                "position_base_mm": state.position_base_mm.round(3).tolist(),
                "velocity_base_mm_s": state.velocity_base_mm_s.round(3).tolist(),
                "position_std_mm": round(state.position_std_mm, 3),
                "measurement_age_ms": round(state.measurement_age_ms, 3),
                "reprojection_error_px": round(state.reprojection_error_px, 3),
                "confidence": round(state.confidence, 4),
                "camera_count": state.camera_count,
                "camera_mask": state.camera_mask,
                "minimum_ray_angle_deg": round(state.minimum_ray_angle_deg, 3),
                "depth_used": state.depth_used,
                "valid": state.valid,
            }
            message = String()
            message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.fist_publisher.publish(message)
            if not state.valid:
                continue
            point = PointStamped()
            _stamp_message(point.header, state.stamp_ns)
            point.point.x, point.point.y, point.point.z = (float(value) for value in state.position_base_mm)
            self.position_publishers[side].publish(point)
            velocity = Vector3Stamped()
            _stamp_message(velocity.header, state.stamp_ns)
            velocity.vector.x, velocity.vector.y, velocity.vector.z = (float(value) for value in state.velocity_base_mm_s)
            self.velocity_publishers[side].publish(velocity)

    def _publish_finished_snapshots(self) -> None:
        remaining: list[tuple[ImpactEvent, Future[SnapshotResult | None]]] = []
        for event, future in self._snapshot_jobs:
            if not future.done():
                remaining.append((event, future))
                continue
            try:
                result = future.result()
            except Exception as error:
                self.get_logger().error(f"impact snapshot failed: {error}")
                result = None
            payload = {
                "impact_id": event.impact_id,
                "impact_stamp_ns": event.stamp_ns,
                "side": event.side,
                "source": event.source,
                "confidence": event.confidence,
                "image_saved": result is not None,
                "directory": str(result.directory) if result is not None else "",
                **dict(event.metadata),
            }
            event_message = String()
            event_message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.impact_publisher.publish(event_message)
            if result is not None:
                image_message = CompressedImage()
                _stamp_message(image_message.header, event.stamp_ns, "impact_triptych")
                image_message.format = "jpeg"
                image_message.data = result.jpeg_bytes
                self.image_publisher.publish(image_message)
                self.get_logger().info(f"impact #{event.impact_id} saved: {result.directory}")
        self._snapshot_jobs = remaining

    def _publish_status(self, now_ns: int) -> None:
        if now_ns - self._last_status_ns < self.status_publish_period_ns:
            return
        self._last_status_ns = now_ns
        target = self.target_worker.latest()
        camera_targets = {
            camera: worker.latest()
            for camera, worker in self.target_workers.items()
        }
        pose_config = self.config["pose"]
        front_pose_ready, front_pose_confidence, front_pose_age_ms = _front_pose_ready_status(
            # Alignment is a FRONT-camera check, not a multi-camera tracking
            # check.  Use the worker's fresh result directly so a BoT-SORT ID
            # change cannot clear the sample that allows training to start.
            self.pose_workers["front"].latest(),
            now_ns,
            float(pose_config.get("front_ready_minimum_landmark_confidence", 0.50)),
            float(pose_config.get("front_ready_maximum_result_age_ms", 250.0)),
        )
        guard, guard_goal = self.impact_detector.guard_progress
        payload = {
            "stamp_ns": now_ns,
            "target_state": target.state if target is not None else "STARTING",
            "target_id": target.local_track_id if target is not None else None,
            "target_detected_people": target.detected_people_count if target is not None else 0,
            "target_valid_people": target.people_count if target is not None else 0,
            "target_device": self.target_worker.device,
            "target_inference_ms": round(self.target_worker.inference_ms, 3),
            "front_pose_detected": front_pose_ready,
            "front_pose_confidence": round(front_pose_confidence, 4),
            "front_pose_age_ms": (
                round(front_pose_age_ms, 2) if front_pose_age_ms is not None else None
            ),
            "targets": {
                camera: {
                    "state": snapshot.state if snapshot is not None else "STARTING",
                    "track_id": snapshot.local_track_id if snapshot is not None else None,
                    "aligned_to_front": snapshot.aligned_to_front if snapshot is not None else False,
                    "alignment_distance_px": (
                        round(snapshot.alignment_distance_px, 2)
                        if snapshot is not None and snapshot.alignment_distance_px is not None
                        else None
                    ),
                    "detected_people": snapshot.detected_people_count if snapshot is not None else 0,
                    "inference_ms": round(self.target_workers[camera].inference_ms, 3),
                    "error": self.target_workers[camera].error,
                }
                for camera, snapshot in camera_targets.items()
            },
            "impact_state": self.impact_detector.state,
            "mitt_tracker": {
                "state": self._mitt_track.state,
                "roi_normalized": self._mitt_track.roi_normalized,
                "processing_ms": round(self._mitt_track.processing_ms, 3),
                "contour_area_px": round(self._mitt_track.contour_area_px, 1),
            },
            "preview_layout": {
                "canvas_width": 1440,
                "canvas_height": 464 if self.show_fist_overlay else 360,
                "front_tile_xywh": [480, 0, 480, 360],
                "front_frame_size": [
                    int(self.config["camera"].get("width", 640)),
                    int(self.config["camera"].get("height", 480)),
                ],
            },
            "guard": [guard, guard_goal],
            "cameras": self.camera_hub.status(),
            "target_error": self.target_worker.error,
            "target_errors": {
                camera: worker.error
                for camera, worker in self.target_workers.items()
                if worker.error
            },
            "pose": {
                camera: {
                    "completed": worker.completed,
                    "dropped": worker.dropped,
                    "error": worker.error,
                }
                for camera, worker in self.pose_workers.items()
            },
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.status_publisher.publish(message)

    def _preview(self, frames: dict, now_ns: int) -> None:
        if (not self.show_preview and not self.publish_preview) or any(
            frames.get(camera) is None for camera in CAMERAS
        ):
            return
        targets = {camera: worker.latest() for camera, worker in self.target_workers.items()}
        target = targets["front"]
        tiles = []
        for camera in CAMERAS:
            image = draw_pose(frames[camera].color_bgr, self.histories[camera].latest)
            camera_target = targets[camera]
            if camera == "front":
                mitt_roi = self._mitt_track.roi_normalized
                if isinstance(mitt_roi, (list, tuple)) and len(mitt_roi) == 4:
                    height, width = image.shape[:2]
                    x1, y1, x2, y2 = (
                        int(round(float(mitt_roi[0]) * width)),
                        int(round(float(mitt_roi[1]) * height)),
                        int(round(float(mitt_roi[2]) * width)),
                        int(round(float(mitt_roi[3]) * height)),
                    )
                    cv2.rectangle(image, (x1, y1), (x2, y2), (80, 255, 80), 2)
                    cv2.putText(
                        image,
                        "IMPACT ZONE",
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (80, 255, 80),
                        2,
                        cv2.LINE_AA,
                    )
                elif self.mitt_tracker.enabled:
                    cv2.putText(
                        image,
                        "MITT LOST - IMPACT DISABLED",
                        (14, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            preview_rotation = self.preview_rotations.get(camera, "none")
            raw_height, raw_width = image.shape[:2]
            image = rotate_image(image, preview_rotation)
            if camera_target is not None and camera_target.bbox_xyxy is not None:
                x1, y1, x2, y2 = map_original_bbox_to_rotated(
                    camera_target.bbox_xyxy,
                    raw_width,
                    raw_height,
                    preview_rotation,
                )
                aligned = camera == "front" or camera_target.aligned_to_front
                color = (0, 255, 255) if aligned else (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
                track_id = camera_target.local_track_id
                label = f"BOT-SORT #{track_id if track_id is not None else '-'}"
                if camera != "front" and aligned:
                    label += " FRONT ALIGN"
                text_x = max(6, x1 + 6)
                text_y = min(image.shape[0] - 10, max(84, y1 + 26))
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
                cv2.rectangle(
                    image,
                    (text_x - 4, text_y - text_size[1] - 8),
                    (min(image.shape[1] - 1, text_x + text_size[0] + 4), text_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            tile = _letterbox(image, 480, 360)
            cv2.rectangle(tile, (0, 0), (480, 32), (0, 0, 0), -1)
            cv2.putText(tile, camera.upper(), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        canvas = np.hstack(tiles)
        target_state = target.state if target is not None else "STARTING"
        target_people = (
            f"{target.detected_people_count}/{target.people_count}"
            if target is not None
            else "0/0"
        )
        guard, goal = self.impact_detector.guard_progress
        _draw_runtime_status(
            canvas,
            target_state,
            target_people,
            self.impact_detector.state,
            guard,
            goal,
            self.target_worker.device,
            self.target_worker.inference_ms,
        )
        target_errors = [
            f"{camera.upper()}: {worker.error}"
            for camera, worker in self.target_workers.items()
            if worker.error
        ]
        if target_errors:
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 180), -1)
            cv2.putText(
                canvas,
                f"TARGET ERROR: {' | '.join(target_errors)[:150]}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if self.show_fist_overlay:
            canvas = _append_fist_state_panel(canvas, self._latest_states)
        if (
            self.publish_preview
            and now_ns - self._last_preview_publish_ns >= self.preview_publish_period_ns
        ):
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, self.preview_jpeg_quality],
            )
            if encoded_ok:
                preview_message = CompressedImage()
                _stamp_message(preview_message.header, now_ns, "sandbag_vision_preview")
                preview_message.format = "jpeg"
                preview_message.data = encoded.tobytes()
                self.preview_publisher.publish(preview_message)
                self._last_preview_publish_ns = now_ns
        if self.show_preview:
            cv2.imshow("Sandbag Vision Realtime", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

    def _publish_front_preview(self, frames: dict, now_ns: int) -> None:
        """Publish one clean RealSense color frame for browser-side measurement."""
        if not self.publish_front_preview:
            return
        packet = frames.get("front")
        if packet is None or packet.sequence == self._last_front_preview_sequence:
            return
        if now_ns - self._last_front_preview_publish_ns < self.front_preview_publish_period_ns:
            return
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            packet.color_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.front_preview_jpeg_quality],
        )
        if not encoded_ok:
            return
        message = CompressedImage()
        _stamp_message(message.header, packet.stamp_ns, "front_realsense_color_optical_frame")
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.front_preview_publisher.publish(message)
        self._last_front_preview_sequence = packet.sequence
        self._last_front_preview_publish_ns = now_ns

    def _tick(self) -> None:
        now_ns = time.time_ns()
        frames = self.camera_hub.latest()
        self._submit_frames(frames)
        self._collect_pose_results()
        self._triangulate(now_ns)
        self._publish_states(now_ns)
        self._publish_finished_snapshots()
        self._publish_status(now_ns)
        self._publish_front_preview(frames, now_ns)
        self._preview(frames, now_ns)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.camera_hub.close()
        for worker in self.target_workers.values():
            worker.close()
        for worker in self.pose_workers.values():
            worker.close()
        self._snapshot_executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()

    def destroy_node(self) -> bool:
        self.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SandbagVisionNode | None = None
    try:
        node = SandbagVisionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        # SIGTERM can invalidate the ROS context while a timer callback is in
        # publish(). Treat that shutdown race as clean, but keep real errors.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

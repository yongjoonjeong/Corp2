#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import rclpy
import yaml
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from punch_feedback_core import (
    LANDMARK_NAMES,
    ClassifiedPunch,
    FeatureError,
    Landmark2D,
    PoseSample,
    PunchDetector,
    ScoreResult,
    classify_punch,
    joint_landmark_names,
    score_punch,
)


try:
    mp_pose = mp.solutions.pose
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose


LANDMARK_INDEX = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
}

SKELETON_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)

SIDE_LABEL_KO = {
    "left": "왼손",
    "right": "오른손",
}

PUNCH_LABEL_KO = {
    "straight": "스트레이트",
    "hook": "훅",
    "uppercut": "어퍼컷",
}

# OpenCV BGR colors: straight=red, hook=green, uppercut=blue.
PUNCH_COLOR_BGR = {
    "straight": (40, 40, 255),
    "hook": (40, 230, 40),
    "uppercut": (255, 120, 40),
}

VIOLATION_LABEL_KO = {
    "elbow_not_extended": "팔꿈치를 충분히 펴세요",
    "elbow_flared": "팔꿈치가 바깥으로 벌어졌어요",
    "guard_dropped": "반대 손 가드를 얼굴 가까이 올리세요",
    "torso_overlean": "상체가 너무 많이 기울어졌어요",
    "wrist_height_off": "주먹 높이를 기준 자세에 맞추세요",
    "hook_elbow_angle_off": "훅의 팔꿈치 각도를 조정하세요",
    "hook_elbow_path_off": "훅 팔꿈치 궤적을 조정하세요",
    "hook_wrist_elbow_misaligned": "훅에서 손목과 팔꿈치 높이를 맞추세요",
    "uppercut_elbow_angle_off": "어퍼컷 팔꿈치 각도를 조정하세요",
    "uppercut_wrist_path_off": "어퍼컷 손목을 위쪽으로 움직이세요",
    "uppercut_height_off": "어퍼컷 주먹 높이를 조정하세요",
}

KOREAN_FONT_REGULAR = Path(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
)
KOREAN_FONT_BOLD = Path(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
)


def load_reference(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    required_sections = ("detector", "classification", "feedback", "profiles")
    missing = [name for name in required_sections if name not in data]
    if missing:
        raise ValueError(f"Missing YAML sections: {', '.join(missing)}")
    return data


def load_korean_font(size: int, bold: bool = False):
    preferred = KOREAN_FONT_BOLD if bold else KOREAN_FONT_REGULAR
    candidates = (
        preferred,
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def to_pose_sample(results, stamp_s: float, image) -> PoseSample | None:
    if not results.pose_landmarks:
        return None
    source = results.pose_landmarks.landmark
    landmarks = {
        name: Landmark2D(
            x=float(source[index].x),
            y=float(source[index].y),
            visibility=float(source[index].visibility),
            z=float(source[index].z),
        )
        for name, index in LANDMARK_INDEX.items()
    }
    return PoseSample(
        stamp_s=stamp_s,
        landmarks=landmarks,
        image=image.copy(),
    )


def landmark_pixel(
    sample: PoseSample,
    name: str,
    width: int,
    height: int,
    mirror: bool,
) -> tuple[int, int]:
    landmark = sample.landmarks[name]
    x = 1.0 - landmark.x if mirror else landmark.x
    return (
        int(max(0.0, min(1.0, x)) * (width - 1)),
        int(max(0.0, min(1.0, landmark.y)) * (height - 1)),
    )


def draw_pose(
    image,
    sample: PoseSample,
    mirror: bool,
    highlighted_landmarks: Iterable[str] = (),
) -> None:
    height, width = image.shape[:2]
    highlighted = set(highlighted_landmarks)

    for first, second in SKELETON_CONNECTIONS:
        cv2.line(
            image,
            landmark_pixel(sample, first, width, height, mirror),
            landmark_pixel(sample, second, width, height, mirror),
            (70, 210, 70),
            2,
            cv2.LINE_AA,
        )

    for name in LANDMARK_NAMES:
        point = landmark_pixel(sample, name, width, height, mirror)
        if name in highlighted:
            cv2.circle(image, point, 14, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(
                image,
                name.upper(),
                (point[0] + 10, max(20, point[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(image, point, 5, (0, 220, 0), -1, cv2.LINE_AA)


def render_evidence(
    classified: ClassifiedPunch,
    score: ScoreResult,
    mirror: bool,
):
    source = classified.key_sample.image
    if source is None:
        raise ValueError("Key pose sample has no image")
    output = cv2.flip(source, 1) if mirror else source.copy()

    highlighted: set[str] = set()
    for violation in score.violations:
        highlighted.update(joint_landmark_names(violation.joint, classified.side))
    draw_pose(output, classified.key_sample, mirror, highlighted)

    cv2.rectangle(output, (0, 0), (output.shape[1], 126), (0, 0, 0), -1)

    pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    header_font = load_korean_font(22, bold=True)
    feedback_font = load_korean_font(18, bold=True)
    footer_font = load_korean_font(14)

    side_label = SIDE_LABEL_KO.get(classified.side, classified.side)
    punch_label = PUNCH_LABEL_KO.get(
        classified.punch_type,
        classified.punch_type,
    )
    draw.text(
        (14, 5),
        f"{side_label} {punch_label} | 점수 {score.total_score:.1f}",
        font=header_font,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )

    feedback_messages = [
        VIOLATION_LABEL_KO.get(violation.code, violation.code)
        for violation in score.violations[:2]
    ]
    if not feedback_messages:
        feedback_messages = ["전체 자세 균형을 확인하세요"]
    for index, message in enumerate(feedback_messages):
        prefix = "피드백: " if index == 0 else "          "
        draw.text(
            (14, 38 + index * 27),
            prefix + message,
            font=feedback_font,
            fill=(255, 105, 20),
            stroke_width=1,
            stroke_fill=(35, 20, 0),
        )

    draw.text(
        (14, 103),
        "임시 2D 웹캠 정자세 기준",
        font=footer_font,
        fill=(185, 185, 185),
    )
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def feature_error_dict(error: FeatureError) -> dict:
    return {
        "feature": error.feature,
        "joint": error.joint,
        "code": error.code,
        "value": round(error.value, 4),
        "target": round(error.target, 4),
        "tolerance": round(error.tolerance, 4),
        "error_ratio": round(error.error_ratio, 4),
    }


def draw_guard_gauge(
    frame,
    state: str,
    guard_frames: int,
    ready_frames: int,
) -> None:
    height, width = frame.shape[:2]
    if height < 36 or width < 280:
        return

    gauge_left = max(230, int(width * 0.43))
    gauge_right = width - 12
    gauge_top = 7
    gauge_bottom = 32
    inner_left = gauge_left + 2
    inner_right = gauge_right - 2
    inner_top = gauge_top + 2
    inner_bottom = gauge_bottom - 2

    ready_frames = max(ready_frames, 1)
    progress = min(max(guard_frames / ready_frames, 0.0), 1.0)
    label = f"GUARD {guard_frames}/{ready_frames}"
    fill_color = (0, 170, 255)

    if state == "READY" or state.startswith("ACTIVE_"):
        progress = 1.0
        label = "GUARD READY"
        fill_color = (40, 190, 40)
    elif state == "COOLDOWN":
        progress = 1.0
        label = "COOLDOWN"
        fill_color = (255, 150, 30)
    elif progress >= 0.75:
        fill_color = (0, 220, 220)

    cv2.rectangle(
        frame,
        (gauge_left, gauge_top),
        (gauge_right, gauge_bottom),
        (75, 75, 75),
        -1,
    )
    fill_width = int((inner_right - inner_left) * progress)
    if fill_width > 0:
        cv2.rectangle(
            frame,
            (inner_left, inner_top),
            (inner_left + fill_width, inner_bottom),
            fill_color,
            -1,
        )
    cv2.rectangle(
        frame,
        (gauge_left, gauge_top),
        (gauge_right, gauge_bottom),
        (215, 215, 215),
        1,
    )

    font_scale = 0.46
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    text_x = gauge_left + max((gauge_right - gauge_left - text_width) // 2, 2)
    text_y = gauge_top + (gauge_bottom - gauge_top + text_height) // 2 - 1
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


class WebcamPunchFeedbackNode(Node):
    def __init__(self) -> None:
        super().__init__("webcam_punch_feedback")

        default_reference = (
            Path(__file__).resolve().parent
            / "config"
            / "temporary_form_reference.yaml"
        )
        default_feedback_dir = Path(__file__).resolve().parent / "feedback_images"
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("display", True)
        self.declare_parameter("mirror_display", True)
        self.declare_parameter("display_width", 960)
        self.declare_parameter("display_height", 720)
        self.declare_parameter("status_panel_height", 138)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("preview_publish_fps", 8.0)
        self.declare_parameter("status_publish_fps", 5.0)
        self.declare_parameter("reference_path", str(default_reference))
        self.declare_parameter("feedback_image_dir", str(default_feedback_dir))

        self.camera_index = int(self.get_parameter("camera_index").value)
        self.frame_width = int(self.get_parameter("frame_width").value)
        self.frame_height = int(self.get_parameter("frame_height").value)
        self.fps = float(self.get_parameter("fps").value)
        self.display = bool(self.get_parameter("display").value)
        self.mirror_display = bool(self.get_parameter("mirror_display").value)
        self.display_width = max(int(self.get_parameter("display_width").value), 320)
        self.display_height = max(int(self.get_parameter("display_height").value), 240)
        self.status_panel_height = max(
            int(self.get_parameter("status_panel_height").value),
            138,
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.preview_publish_fps = max(float(self.get_parameter("preview_publish_fps").value), 0.0)
        self.status_publish_fps = max(float(self.get_parameter("status_publish_fps").value), 0.0)
        reference_path = Path(str(self.get_parameter("reference_path").value))
        self.feedback_image_dir = Path(
            str(self.get_parameter("feedback_image_dir").value)
        ).expanduser()

        self.reference = load_reference(reference_path)
        self.save_score_threshold = float(
            self.reference["feedback"].get("save_score_threshold", 30.0)
        )
        self.pass_score_threshold = float(
            self.reference["feedback"].get("pass_score_threshold", 30.0)
        )
        self.detector = PunchDetector(self.reference["detector"])
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
        )

        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.camera_index}")

        self.window_name = "Sandbag Punch Feedback MVP"
        if self.display:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
            cv2.namedWindow(self.window_name, window_flags)
            cv2.resizeWindow(
                self.window_name,
                self.display_width,
                self.display_height + self.status_panel_height,
            )

        event_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        evidence_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        preview_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.score_publisher = self.create_publisher(
            String,
            "/sandbag/form/score",
            event_qos,
        )
        self.image_publisher = self.create_publisher(
            CompressedImage,
            "/sandbag/form/joint_evidence/compressed",
            evidence_qos,
        )
        self.preview_publisher = self.create_publisher(
            CompressedImage,
            "/sandbag/form/preview/compressed",
            preview_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            "/sandbag/form/status",
            event_qos,
        )

        self.punch_id = 0
        self.missing_pose_frames = 0
        self.last_result = "No punch yet"
        self.last_result_display: dict | None = None
        self.impact_overlay: dict | None = None
        self._last_preview_publish = 0.0
        self._last_status_publish = 0.0
        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self.on_frame)
        self.get_logger().info(
            f"Webcam MVP ready: camera={self.camera_index}, reference={reference_path}"
        )
        self.get_logger().info(
            "Topics: /sandbag/form/score, "
            "/sandbag/form/joint_evidence/compressed, "
            "/sandbag/form/preview/compressed, "
            "/sandbag/form/status"
        )

    def on_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.get_logger().warning("Webcam frame read failed")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        sample = to_pose_sample(results, time.monotonic(), frame)

        if sample is None:
            self.missing_pose_frames += 1
            if self.missing_pose_frames >= 10:
                self.detector.reset()
        else:
            self.missing_pose_frames = 0
            event = self.detector.update(sample)
            if event is not None:
                self.handle_punch(event)

        self.publish_status(sample)

        camera_view = cv2.flip(frame, 1) if self.mirror_display else frame.copy()
        if sample is not None:
            draw_pose(camera_view, sample, self.mirror_display)
        self.draw_cooldown_effect(camera_view)
        self.publish_preview(camera_view)

        if self.display:
            preview = self.build_display_frame(camera_view)
            cv2.imshow(self.window_name, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                rclpy.shutdown()


    def publish_status(self, sample: PoseSample | None) -> None:
        if self.status_publish_fps <= 0:
            return
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_status_publish < 1.0 / self.status_publish_fps:
            return
        centered = False
        if sample is not None:
            shoulder_center_x = (
                sample.landmarks["left_shoulder"].x
                + sample.landmarks["right_shoulder"].x
            ) / 2.0
            centered = 0.30 <= shoulder_center_x <= 0.70
        telemetry = self.detector.telemetry
        payload = {
            "pose_detected": sample is not None,
            "centered": centered,
            "detector_state": self.detector.state,
            "guard_frames": int(telemetry.get("guard_frames", 0)),
            "ready_frames": int(telemetry.get("ready_frames", 0)),
            "camera_index": self.camera_index,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.status_publisher.publish(message)
        self._last_status_publish = now_monotonic

    def publish_preview(self, frame) -> None:
        if self.preview_publish_fps <= 0:
            return
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_preview_publish < 1.0 / self.preview_publish_fps:
            return
        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, min(self.jpeg_quality, 82)],
        )
        if not encoded:
            return
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"webcam_{self.camera_index}"
        message.format = "jpeg"
        message.data = jpeg.tobytes()
        self.preview_publisher.publish(message)
        self._last_preview_publish = now_monotonic

    def build_display_frame(self, camera_view):
        resized_camera = cv2.resize(
            camera_view,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_LINEAR,
        )
        status_panel = np.zeros(
            (self.status_panel_height, self.display_width, 3),
            dtype=np.uint8,
        )
        self.draw_preview_status(status_panel)
        return np.vstack((status_panel, resized_camera))

    def draw_preview_status(self, frame) -> None:
        telemetry = self.detector.telemetry
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], frame.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            f"STATE: {self.detector.state}",
            (12, 26),
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

        for line_index, side in enumerate(("left", "right")):
            side_key = side.lower()
            text = (
                f"{side.upper()[0]} SPD {telemetry[f'{side_key}_speed']:.2f} | "
                f"MOVE {telemetry[f'{side_key}_displacement']:.2f} | "
                f"ELB+ {telemetry[f'{side_key}_elbow_delta']:.1f} | "
                f"START {int(telemetry[f'{side_key}_start_confirm'])}/"
                f"{int(telemetry['start_confirm_frames'])}"
            )
            cv2.putText(
                frame,
                text,
                (12, 54 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        self.draw_last_result(frame)

    def draw_last_result(self, frame) -> None:
        if self.last_result_display is None:
            cv2.putText(
                frame,
                "No punch yet",
                (12, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return

        result = self.last_result_display
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        thickness = 1
        x = 12
        y = 124
        prefix = f"#{result['punch_id']} {result['side']} "
        punch_text = str(result["punch_type"]).upper()
        suffix = (
            f" score={result['score']:.1f} image={result['image_published']} "
            f"saved={result['image_saved']}"
        )

        cv2.putText(
            frame,
            prefix,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        prefix_width = cv2.getTextSize(
            prefix,
            font,
            font_scale,
            thickness,
        )[0][0]
        x += prefix_width
        punch_color = PUNCH_COLOR_BGR.get(
            str(result["punch_type"]),
            (255, 255, 255),
        )
        cv2.putText(
            frame,
            punch_text,
            (x, y),
            font,
            font_scale,
            punch_color,
            2,
            cv2.LINE_AA,
        )
        punch_width = cv2.getTextSize(
            punch_text,
            font,
            font_scale,
            2,
        )[0][0]
        x += punch_width
        cv2.putText(
            frame,
            suffix,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def draw_cooldown_effect(self, frame) -> None:
        if self.detector.state != "COOLDOWN" or self.impact_overlay is None:
            return

        overlay = self.impact_overlay
        height, width = frame.shape[:2]
        x_norm = float(overlay["x"])
        if self.mirror_display:
            x_norm = 1.0 - x_norm
        x = int(np.clip(x_norm, 0.0, 1.0) * (width - 1))
        y = int(np.clip(float(overlay["y"]), 0.0, 1.0) * (height - 1))
        color = PUNCH_COLOR_BGR.get(
            str(overlay["punch_type"]),
            (0, 255, 255),
        )

        duration = max(float(self.detector.cooldown_s), 0.1)
        progress = float(
            np.clip((time.monotonic() - float(overlay["started_at"])) / duration, 0.0, 1.0)
        )
        radius = 20 + int(32 * progress)
        cv2.circle(frame, (x, y), radius, color, 4, cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(7, radius // 3), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, (x - radius - 12, y), (x + radius + 12, y), color, 2)
        cv2.line(frame, (x, y - radius - 12), (x, y + radius + 12), color, 2)
        cv2.putText(
            frame,
            str(overlay["punch_type"]).upper(),
            (max(8, x + radius + 10), max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        passed = bool(overlay["passed"])
        feedback_text = "GOOD!" if passed else "BAD.."
        feedback_color = (80, 255, 80) if passed else (40, 40, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.80
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            feedback_text,
            font,
            font_scale,
            thickness,
        )
        text_x = int(np.clip(
            x - text_width // 2,
            8,
            max(8, width - text_width - 8),
        ))
        text_y = min(
            y + radius + text_height + 14,
            height - baseline - 8,
        )
        cv2.putText(
            frame,
            feedback_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            feedback_text,
            (text_x, text_y),
            font,
            font_scale,
            feedback_color,
            thickness,
            cv2.LINE_AA,
        )

    def handle_punch(self, event) -> None:
        classified = classify_punch(event, self.reference["classification"])
        result = score_punch(
            classified,
            self.reference["profiles"],
            self.reference["feedback"],
        )
        self.punch_id += 1
        now = self.get_clock().now().to_msg()
        stamp_ns = int(now.sec) * 1_000_000_000 + int(now.nanosec)

        payload = {
            "stamp_ns": stamp_ns,
            "punch_id": self.punch_id,
            "punch_type": classified.punch_type,
            "punch_side": classified.side,
            "classification_confidence": round(classified.confidence, 4),
            "classification_reason": classified.classification_reason,
            "total_score": round(result.total_score, 2),
            "passed": result.total_score >= self.pass_score_threshold,
            "feedback_required": result.feedback_required,
            "image_save_required": result.total_score < self.save_score_threshold,
            "motion_features": {
                key: round(value, 4)
                for key, value in classified.motion_features.items()
            },
            "violations": [
                feature_error_dict(error) for error in result.violations
            ],
            "all_feature_errors": [
                feature_error_dict(error) for error in result.feature_errors
            ],
        }
        score_message = String()
        score_message.data = json.dumps(payload, ensure_ascii=False)
        self.score_publisher.publish(score_message)

        image_published = False
        image_saved = False
        image_save_required = result.total_score < self.save_score_threshold
        if result.feedback_required or image_save_required:
            evidence = render_evidence(
                classified,
                result,
                mirror=self.mirror_display,
            )
            encoded, jpeg = cv2.imencode(
                ".jpg",
                evidence,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if encoded:
                jpeg_bytes = jpeg.tobytes()
                if result.feedback_required:
                    image_message = CompressedImage()
                    image_message.header.stamp = now
                    image_message.header.frame_id = f"webcam_{self.camera_index}"
                    image_message.format = "jpeg"
                    image_message.data = jpeg_bytes
                    self.image_publisher.publish(image_message)
                    image_published = True

                if image_save_required:
                    saved_path = self.save_feedback_image(
                        jpeg_bytes,
                        classified,
                        result,
                    )
                    image_saved = saved_path is not None
            else:
                self.get_logger().error("Failed to encode evidence JPEG")

        self.last_result = (
            f"#{self.punch_id} {classified.side} {classified.punch_type} "
            f"score={result.total_score:.1f} image={image_published} "
            f"saved={image_saved}"
        )
        impact_wrist = classified.key_sample.landmarks[
            f"{classified.side}_wrist"
        ]
        self.impact_overlay = {
            "x": impact_wrist.x,
            "y": impact_wrist.y,
            "punch_type": classified.punch_type,
            "score": result.total_score,
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
            f"{self.last_result} reason={classified.classification_reason} "
            f"linearity={motion.get('path_linearity', 0.0):.2f} "
            f"curvature={motion.get('path_curvature_ratio', 0.0):.2f} "
            f"ignored={motion.get('recovery_frames_ignored', 0.0):.0f}"
        )

    def save_feedback_image(
        self,
        jpeg_bytes: bytes,
        classified: ClassifiedPunch,
        result: ScoreResult,
    ) -> Path | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = (
            f"{timestamp}_punch_{self.punch_id:04d}_"
            f"{classified.side}_{classified.punch_type}_"
            f"score_{result.total_score:.1f}.jpg"
        )
        path = self.feedback_image_dir / filename
        try:
            self.feedback_image_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jpeg_bytes)
        except OSError as error:
            self.get_logger().error(f"Failed to save feedback image: {error}")
            return None

        self.get_logger().info(f"Saved low-score feedback image: {path}")
        return path

    def destroy_node(self) -> bool:
        if hasattr(self, "timer"):
            self.timer.cancel()
        if hasattr(self, "capture"):
            self.capture.release()
        if hasattr(self, "pose"):
            self.pose.close()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WebcamPunchFeedbackNode()
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

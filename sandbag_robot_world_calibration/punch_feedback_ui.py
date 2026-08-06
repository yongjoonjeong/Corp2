from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from punch_feedback_core import (
    LANDMARK_NAMES,
    ClassifiedPunch,
    FeatureError,
    PoseSample,
    ScoreResult,
    joint_landmark_names,
)


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
SIDE_LABEL_KO = {"left": "왼손", "right": "오른손"}
PUNCH_LABEL_KO = {
    "straight": "스트레이트",
    "hook": "훅",
    "uppercut": "어퍼컷",
}
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
    "straight_forward_path_off": "주먹을 목표 방향으로 곧게 뻗으세요",
    "straight_path_not_linear": "스트레이트 손목 경로를 더 곧게 만드세요",
    "hook_lateral_path_off": "훅 손목을 옆에서 안쪽으로 이동하세요",
    "hook_curve_off": "훅 손목 궤적을 더 둥글게 만드세요",
    "uppercut_upward_path_off": "어퍼컷 주먹을 아래에서 위로 올리세요",
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
    image: np.ndarray,
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
    reference_label: str = "임시 2D 웹캠 정자세 기준",
) -> np.ndarray:
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
    punch_label = PUNCH_LABEL_KO.get(classified.punch_type, classified.punch_type)
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
    ] or ["전체 자세 균형을 확인하세요"]
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
        reference_label,
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
    frame: np.ndarray,
    state: str,
    guard_frames: int,
    ready_frames: int,
) -> None:
    height, width = frame.shape[:2]
    if height < 36 or width < 280:
        return
    gauge_left = max(230, int(width * 0.43))
    gauge_right = width - 12
    gauge_top, gauge_bottom = 7, 32
    inner_left, inner_right = gauge_left + 2, gauge_right - 2
    inner_top, inner_bottom = gauge_top + 2, gauge_bottom - 2
    ready_frames = max(ready_frames, 1)
    progress = min(max(guard_frames / ready_frames, 0.0), 1.0)
    label = f"GUARD {guard_frames}/{ready_frames}"
    fill_color = (0, 170, 255)
    if state == "READY" or state.startswith("ACTIVE_"):
        progress, label, fill_color = 1.0, "GUARD READY", (40, 190, 40)
    elif state == "COOLDOWN":
        progress, label, fill_color = 1.0, "COOLDOWN", (255, 150, 30)
    elif progress >= 0.75:
        fill_color = (0, 220, 220)
    cv2.rectangle(
        frame, (gauge_left, gauge_top), (gauge_right, gauge_bottom), (75, 75, 75), -1
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
    font_scale, thickness = 0.46, 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
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

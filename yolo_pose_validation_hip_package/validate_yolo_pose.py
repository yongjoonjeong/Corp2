#!/usr/bin/env python3
"""
YOLO Pose 복싱 영상 비교 검증기

주요 기능
- 여러 YOLO Pose 모델을 동일 영상/동일 조건으로 순차 비교
- BoT-SORT 기반 Track ID 유지
- 최초 중앙 사람을 TARGET_ID로 고정
- 평균 FPS / 추론시간 / 대상 유지율 측정
- 손목·팔꿈치·골반 키포인트 confidence와 누락률 측정
- 손목 위치의 프레임 간 흔들림(jitter) 측정
- 모델별 주석 영상, 프레임 CSV, 요약 CSV/JSON 저장

주의
- 이 스크립트의 '누락률/유지율'은 정답 라벨 없이 산출하는 실사용 지표입니다.
- 논문식 pose mAP/OKS를 측정하려면 별도의 정답 키포인트 라벨이 필요합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "OpenCV 또는 NumPy가 없습니다.\n"
        "설치: pip install opencv-python numpy"
    ) from exc

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch가 없습니다.\n"
        "환경에 맞는 PyTorch를 설치하세요: https://pytorch.org/"
    ) from exc

try:
    from ultralytics import YOLO
    import ultralytics
except ImportError as exc:
    raise SystemExit(
        "Ultralytics가 없습니다.\n"
        "설치: pip install -U ultralytics"
    ) from exc


# COCO Pose 17-keypoint index
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

KEYPOINT_NAMES = {
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
}

SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
]


@dataclass
class ModelSummary:
    model: str
    video: str
    tracker: str
    input_width: int
    input_height: int
    source_fps: float
    total_frames: int
    processed_frames: int
    duration_sec: float
    avg_processing_fps: float
    avg_inference_ms: float
    p95_inference_ms: float
    avg_people_detected: float
    frames_with_person_pct: float
    target_present_pct: float
    target_id_switches: int
    target_relocks: int
    left_wrist_valid_pct: float
    right_wrist_valid_pct: float
    both_wrists_valid_pct: float
    left_elbow_valid_pct: float
    right_elbow_valid_pct: float
    left_hip_valid_pct: float
    right_hip_valid_pct: float
    left_wrist_avg_conf: float
    right_wrist_avg_conf: float
    left_elbow_avg_conf: float
    right_elbow_avg_conf: float
    left_hip_avg_conf: float
    right_hip_avg_conf: float
    left_wrist_jitter_norm_avg: float
    right_wrist_jitter_norm_avg: float
    output_video: str
    frame_csv: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO Pose 모델별 복싱 영상 성능 비교"
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=Path("realsense_20260805_163743.mp4"),
        help="입력 영상 경로",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n-pose.pt", "yolo11s-pose.pt"],
        help="비교할 Pose 모델 목록",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo_pose_results"),
        help="결과 저장 폴더",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, 0, 1 등",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 입력 이미지 크기",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="사람 검출 confidence threshold",
    )
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.35,
        help="유효 키포인트 confidence threshold",
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="botsort.yaml 또는 bytetrack.yaml",
    )
    parser.add_argument(
        "--target-timeout",
        type=int,
        default=30,
        help="TARGET_ID가 이 프레임 수만큼 없으면 중앙 사람으로 재선택",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="테스트용 최대 프레임 수, 0은 전체",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="N프레임마다 1회 처리. 정식 비교는 1 권장",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="결과 영상 저장 생략",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="처리 화면 실시간 표시",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def sanitize_model_name(model_name: str) -> str:
    return Path(model_name).stem.replace(" ", "_").replace("/", "_")


def choose_center_detection(
    boxes_xyxy: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> Optional[int]:
    if boxes_xyxy.size == 0:
        return None

    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    best_idx = None
    best_score = float("inf")

    for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        distance = math.hypot(
            (box_center_x - center_x) / max(frame_width, 1),
            (box_center_y - center_y) / max(frame_height, 1),
        )

        # 너무 작은 주변 인물보다 실제 운동자 박스를 우선하도록 약한 면적 보정
        area_ratio = max((x2 - x1) * (y2 - y1), 1.0) / (
            frame_width * frame_height
        )
        score = distance - 0.10 * math.sqrt(area_ratio)

        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def torso_scale(kpts_xy: np.ndarray, kpts_conf: np.ndarray, threshold: float) -> float:
    valid = (
        kpts_conf[LEFT_SHOULDER] >= threshold
        and kpts_conf[RIGHT_SHOULDER] >= threshold
    )
    if valid:
        return max(
            float(
                np.linalg.norm(
                    kpts_xy[LEFT_SHOULDER] - kpts_xy[RIGHT_SHOULDER]
                )
            ),
            1.0,
        )
    return 1.0


def valid_keypoint(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    index: int,
    threshold: float,
) -> bool:
    if index >= len(kpts_xy) or index >= len(kpts_conf):
        return False
    x, y = kpts_xy[index]
    return (
        float(kpts_conf[index]) >= threshold
        and math.isfinite(float(x))
        and math.isfinite(float(y))
        and float(x) > 0
        and float(y) > 0
    )


def draw_target_pose(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    threshold: float,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        frame,
        f"TARGET ID {track_id}",
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for a, b in SKELETON:
        if valid_keypoint(kpts_xy, kpts_conf, a, threshold) and valid_keypoint(
            kpts_xy, kpts_conf, b, threshold
        ):
            pa = tuple(map(int, kpts_xy[a]))
            pb = tuple(map(int, kpts_xy[b]))
            cv2.line(frame, pa, pb, (255, 255, 0), 3, cv2.LINE_AA)

    for idx, name in KEYPOINT_NAMES.items():
        if valid_keypoint(kpts_xy, kpts_conf, idx, threshold):
            point = tuple(map(int, kpts_xy[idx]))
            cv2.circle(frame, point, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{name}:{kpts_conf[idx]:.2f}",
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            # 누락 관절은 박스 내부 좌측에 표시
            pass


def create_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"결과 영상 저장기를 열 수 없습니다: {output_path}")
    return writer


def run_model(
    model_path: str,
    video_path: Path,
    output_dir: Path,
    device: str,
    imgsz: int,
    conf: float,
    kpt_conf_threshold: float,
    tracker: str,
    target_timeout: int,
    max_frames: int,
    frame_stride: int,
    save_video: bool,
    show: bool,
) -> ModelSummary:
    model_tag = sanitize_model_name(model_path)
    model_dir = output_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = model_dir / f"{model_tag}_annotated.mp4"
    frame_csv_path = model_dir / f"{model_tag}_frames.csv"

    writer: Optional[cv2.VideoWriter] = None
    if save_video:
        writer = create_writer(
            output_video_path,
            source_fps / max(frame_stride, 1),
            source_width,
            source_height,
        )

    print("\n" + "=" * 76)
    print(f"[모델 로드] {model_path}")
    print(f"[영상] {video_path}")
    print(f"[장치] {device} | imgsz={imgsz} | conf={conf}")
    print("=" * 76)

    model = YOLO(model_path)

    processed_frames = 0
    raw_frame_index = -1
    frames_with_person = 0
    people_counts: List[int] = []
    inference_ms_values: List[float] = []

    target_id: Optional[int] = None
    target_missing_frames = 0
    target_present_frames = 0
    target_id_switches = 0
    target_relocks = 0
    previous_target_id: Optional[int] = None

    valid_counts = {name: 0 for name in KEYPOINT_NAMES.values()}
    conf_values: Dict[str, List[float]] = {
        name: [] for name in KEYPOINT_NAMES.values()
    }

    both_wrists_valid_frames = 0

    previous_wrist_position: Dict[str, Optional[np.ndarray]] = {
        "left_wrist": None,
        "right_wrist": None,
    }
    wrist_jitter_values: Dict[str, List[float]] = {
        "left_wrist": [],
        "right_wrist": [],
    }

    processing_start = time.perf_counter()

    csv_fields = [
        "raw_frame",
        "processed_frame",
        "timestamp_sec",
        "people_count",
        "target_id",
        "target_present",
        "target_relocked",
        "inference_ms",
        "left_shoulder_conf",
        "right_shoulder_conf",
        "left_elbow_conf",
        "right_elbow_conf",
        "left_wrist_conf",
        "right_wrist_conf",
        "left_hip_conf",
        "right_hip_conf",
        "left_wrist_valid",
        "right_wrist_valid",
        "both_wrists_valid",
        "left_hip_valid",
        "right_hip_valid",
        "both_hips_valid",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_hip_x",
        "left_hip_y",
        "right_hip_x",
        "right_hip_y",
        "left_wrist_jitter_norm",
        "right_wrist_jitter_norm",
    ]

    try:
        with frame_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            csv_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                raw_frame_index += 1

                if raw_frame_index % max(frame_stride, 1) != 0:
                    continue

                if max_frames > 0 and processed_frames >= max_frames:
                    break

                processed_frames += 1
                timestamp_sec = raw_frame_index / source_fps if source_fps else 0.0

                inference_start = time.perf_counter()
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker=tracker,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )[0]
                wall_inference_ms = (time.perf_counter() - inference_start) * 1000.0

                # Ultralytics 내부 speed 값이 있으면 preprocess+inference+postprocess 합 사용
                if getattr(result, "speed", None):
                    speed = result.speed
                    inference_ms = float(
                        speed.get("preprocess", 0.0)
                        + speed.get("inference", 0.0)
                        + speed.get("postprocess", 0.0)
                    )
                    if inference_ms <= 0:
                        inference_ms = wall_inference_ms
                else:
                    inference_ms = wall_inference_ms

                inference_ms_values.append(inference_ms)

                annotated = frame.copy()
                people_count = 0
                target_idx: Optional[int] = None
                target_relocked_this_frame = False

                boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                track_ids = np.empty((0,), dtype=np.int64)
                keypoints_xy = np.empty((0, 17, 2), dtype=np.float32)
                keypoints_conf = np.empty((0, 17), dtype=np.float32)

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    people_count = len(boxes_xyxy)

                    if result.boxes.id is not None:
                        track_ids = (
                            result.boxes.id.detach().cpu().numpy().astype(np.int64)
                        )
                    else:
                        # 추적 ID가 없는 경우 프레임 내부 임시 ID
                        track_ids = np.arange(people_count, dtype=np.int64)

                    if result.keypoints is not None:
                        keypoints_xy = (
                            result.keypoints.xy.detach().cpu().numpy()
                        )
                        if result.keypoints.conf is not None:
                            keypoints_conf = (
                                result.keypoints.conf.detach().cpu().numpy()
                            )
                        else:
                            keypoints_conf = np.ones(
                                (people_count, keypoints_xy.shape[1]),
                                dtype=np.float32,
                            )

                people_counts.append(people_count)
                if people_count > 0:
                    frames_with_person += 1

                # 현재 TARGET_ID 찾기
                if target_id is not None and len(track_ids) > 0:
                    matching = np.where(track_ids == target_id)[0]
                    if matching.size > 0:
                        target_idx = int(matching[0])
                        target_missing_frames = 0
                    else:
                        target_missing_frames += 1
                elif target_id is not None:
                    target_missing_frames += 1

                # 초기 선택 또는 timeout 후 중앙 인물 재선택
                if people_count > 0 and (
                    target_id is None or target_missing_frames >= target_timeout
                ):
                    center_idx = choose_center_detection(
                        boxes_xyxy, source_width, source_height
                    )
                    if center_idx is not None:
                        new_target_id = int(track_ids[center_idx])

                        if target_id is not None:
                            target_relocks += 1
                            target_relocked_this_frame = True
                            if new_target_id != target_id:
                                target_id_switches += 1

                        previous_target_id = target_id
                        target_id = new_target_id
                        target_idx = center_idx
                        target_missing_frames = 0

                        # 재연결 시 이전 위치와의 jitter 연결을 끊음
                        previous_wrist_position["left_wrist"] = None
                        previous_wrist_position["right_wrist"] = None

                row = {
                    "raw_frame": raw_frame_index,
                    "processed_frame": processed_frames,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "people_count": people_count,
                    "target_id": "" if target_id is None else target_id,
                    "target_present": int(target_idx is not None),
                    "target_relocked": int(target_relocked_this_frame),
                    "inference_ms": round(inference_ms, 4),
                    "left_shoulder_conf": 0.0,
                    "right_shoulder_conf": 0.0,
                    "left_elbow_conf": 0.0,
                    "right_elbow_conf": 0.0,
                    "left_wrist_conf": 0.0,
                    "right_wrist_conf": 0.0,
                    "left_hip_conf": 0.0,
                    "right_hip_conf": 0.0,
                    "left_wrist_valid": 0,
                    "right_wrist_valid": 0,
                    "both_wrists_valid": 0,
                    "left_hip_valid": 0,
                    "right_hip_valid": 0,
                    "both_hips_valid": 0,
                    "left_wrist_x": "",
                    "left_wrist_y": "",
                    "right_wrist_x": "",
                    "right_wrist_y": "",
                    "left_hip_x": "",
                    "left_hip_y": "",
                    "right_hip_x": "",
                    "right_hip_y": "",
                    "left_wrist_jitter_norm": "",
                    "right_wrist_jitter_norm": "",
                }

                if target_idx is not None and target_idx < len(keypoints_xy):
                    target_present_frames += 1
                    kpts_xy = keypoints_xy[target_idx]
                    kpts_conf = keypoints_conf[target_idx]
                    box = boxes_xyxy[target_idx]

                    draw_target_pose(
                        annotated,
                        box,
                        int(track_ids[target_idx]),
                        kpts_xy,
                        kpts_conf,
                        kpt_conf_threshold,
                    )

                    scale = torso_scale(
                        kpts_xy, kpts_conf, kpt_conf_threshold
                    )

                    for index, name in KEYPOINT_NAMES.items():
                        confidence = (
                            float(kpts_conf[index])
                            if index < len(kpts_conf)
                            else 0.0
                        )
                        row[f"{name}_conf"] = round(confidence, 4)
                        conf_values[name].append(confidence)

                        if valid_keypoint(
                            kpts_xy, kpts_conf, index, kpt_conf_threshold
                        ):
                            valid_counts[name] += 1

                    left_wrist_valid = valid_keypoint(
                        kpts_xy,
                        kpts_conf,
                        LEFT_WRIST,
                        kpt_conf_threshold,
                    )
                    right_wrist_valid = valid_keypoint(
                        kpts_xy,
                        kpts_conf,
                        RIGHT_WRIST,
                        kpt_conf_threshold,
                    )

                    row["left_wrist_valid"] = int(left_wrist_valid)
                    row["right_wrist_valid"] = int(right_wrist_valid)
                    row["both_wrists_valid"] = int(
                        left_wrist_valid and right_wrist_valid
                    )

                    if left_wrist_valid and right_wrist_valid:
                        both_wrists_valid_frames += 1

                    left_hip_valid = valid_keypoint(
                        kpts_xy, kpts_conf, LEFT_HIP, kpt_conf_threshold
                    )
                    right_hip_valid = valid_keypoint(
                        kpts_xy, kpts_conf, RIGHT_HIP, kpt_conf_threshold
                    )
                    row["left_hip_valid"] = int(left_hip_valid)
                    row["right_hip_valid"] = int(right_hip_valid)
                    row["both_hips_valid"] = int(left_hip_valid and right_hip_valid)

                    for side, index in [
                        ("left_wrist", LEFT_WRIST),
                        ("right_wrist", RIGHT_WRIST),
                    ]:
                        if valid_keypoint(
                            kpts_xy, kpts_conf, index, kpt_conf_threshold
                        ):
                            position = kpts_xy[index].astype(np.float64)
                            row[f"{side}_x"] = round(float(position[0]), 3)
                            row[f"{side}_y"] = round(float(position[1]), 3)

                            previous = previous_wrist_position[side]
                            if previous is not None:
                                jitter = float(
                                    np.linalg.norm(position - previous) / scale
                                )
                                wrist_jitter_values[side].append(jitter)
                                row[f"{side}_jitter_norm"] = round(jitter, 6)

                            previous_wrist_position[side] = position
                        else:
                            # 누락 프레임 뒤 재등장 시 큰 점프를 jitter로 오인하지 않음
                            previous_wrist_position[side] = None

                if target_idx is not None and target_idx < len(keypoints_xy):
                    for side, index in [("left_hip", LEFT_HIP), ("right_hip", RIGHT_HIP)]:
                        if valid_keypoint(kpts_xy, kpts_conf, index, kpt_conf_threshold):
                            position = kpts_xy[index]
                            row[f"{side}_x"] = round(float(position[0]), 3)
                            row[f"{side}_y"] = round(float(position[1]), 3)

                # 화면 정보
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (source_width, 102),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    f"Model: {model_tag} | Frame: {raw_frame_index}/{source_total_frames}",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"People: {people_count} | Target: {target_id} | Inference: {inference_ms:.1f} ms",
                    (18, 59),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"KPT threshold: {kpt_conf_threshold:.2f} | Wrists L/R: {row['left_wrist_valid']}/{row['right_wrist_valid']} | Hips L/R: {row['left_hip_valid']}/{row['right_hip_valid']}",
                    (18, 87),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.57,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                csv_writer.writerow(row)

                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow("YOLO Pose Validation", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[사용자 중단]")
                        break

                if processed_frames % 100 == 0:
                    elapsed = time.perf_counter() - processing_start
                    current_fps = processed_frames / elapsed if elapsed > 0 else 0
                    print(
                        f"\r[{model_tag}] {processed_frames} frames "
                        f"| 처리 {current_fps:.1f} FPS "
                        f"| 대상 유지 {pct(target_present_frames, processed_frames):.1f}%",
                        end="",
                        flush=True,
                    )

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    processing_duration = time.perf_counter() - processing_start
    print()

    summary = ModelSummary(
        model=model_path,
        video=str(video_path),
        tracker=tracker,
        input_width=source_width,
        input_height=source_height,
        source_fps=source_fps,
        total_frames=source_total_frames,
        processed_frames=processed_frames,
        duration_sec=round(processing_duration, 3),
        avg_processing_fps=round(
            processed_frames / processing_duration
            if processing_duration > 0
            else 0.0,
            3,
        ),
        avg_inference_ms=round(safe_mean(inference_ms_values), 3),
        p95_inference_ms=round(percentile(inference_ms_values, 95), 3),
        avg_people_detected=round(safe_mean(people_counts), 3),
        frames_with_person_pct=round(
            pct(frames_with_person, processed_frames), 3
        ),
        target_present_pct=round(
            pct(target_present_frames, processed_frames), 3
        ),
        target_id_switches=target_id_switches,
        target_relocks=target_relocks,
        left_wrist_valid_pct=round(
            pct(valid_counts["left_wrist"], target_present_frames), 3
        ),
        right_wrist_valid_pct=round(
            pct(valid_counts["right_wrist"], target_present_frames), 3
        ),
        both_wrists_valid_pct=round(
            pct(both_wrists_valid_frames, target_present_frames), 3
        ),
        left_elbow_valid_pct=round(
            pct(valid_counts["left_elbow"], target_present_frames), 3
        ),
        right_elbow_valid_pct=round(
            pct(valid_counts["right_elbow"], target_present_frames), 3
        ),
        left_hip_valid_pct=round(
            pct(valid_counts["left_hip"], target_present_frames), 3
        ),
        right_hip_valid_pct=round(
            pct(valid_counts["right_hip"], target_present_frames), 3
        ),
        left_wrist_avg_conf=round(
            safe_mean(conf_values["left_wrist"]), 4
        ),
        right_wrist_avg_conf=round(
            safe_mean(conf_values["right_wrist"]), 4
        ),
        left_elbow_avg_conf=round(
            safe_mean(conf_values["left_elbow"]), 4
        ),
        right_elbow_avg_conf=round(
            safe_mean(conf_values["right_elbow"]), 4
        ),
        left_hip_avg_conf=round(
            safe_mean(conf_values["left_hip"]), 4
        ),
        right_hip_avg_conf=round(
            safe_mean(conf_values["right_hip"]), 4
        ),
        left_wrist_jitter_norm_avg=round(
            safe_mean(wrist_jitter_values["left_wrist"]), 6
        ),
        right_wrist_jitter_norm_avg=round(
            safe_mean(wrist_jitter_values["right_wrist"]), 6
        ),
        output_video=str(output_video_path) if save_video else "",
        frame_csv=str(frame_csv_path),
    )

    model_summary_path = model_dir / f"{model_tag}_summary.json"
    model_summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[완료] {model_tag}")
    print(f"  처리 FPS          : {summary.avg_processing_fps}")
    print(f"  평균 추론시간     : {summary.avg_inference_ms} ms")
    print(f"  대상 유지율       : {summary.target_present_pct}%")
    print(f"  왼손목 유효율     : {summary.left_wrist_valid_pct}%")
    print(f"  오른손목 유효율   : {summary.right_wrist_valid_pct}%")
    print(f"  왼골반 유효율     : {summary.left_hip_valid_pct}%")
    print(f"  오른골반 유효율   : {summary.right_hip_valid_pct}%")
    print(f"  ID 변경           : {summary.target_id_switches}회")
    print(f"  프레임 CSV        : {frame_csv_path}")
    if save_video:
        print(f"  결과 영상         : {output_video_path}")

    # 모델 객체 GPU 메모리 정리
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def save_comparison(
    summaries: Sequence[ModelSummary],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> None:
    summary_csv = output_dir / "model_comparison.csv"
    summary_json = output_dir / "model_comparison.json"
    environment_json = output_dir / "environment.json"

    if summaries:
        fields = list(asdict(summaries[0]).keys())
        with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(asdict(summary))

    summary_json.write_text(
        json.dumps(
            [asdict(summary) for summary in summaries],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    environment = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "selected_device": device,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }

    environment_json.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("[전체 비교 완료]")
    print(f"요약 CSV : {summary_csv}")
    print(f"요약 JSON: {summary_json}")
    print(f"환경 정보: {environment_json}")
    print("=" * 76)

    # 터미널 비교표
    header = (
        f"{'MODEL':24}"
        f"{'FPS':>9}"
        f"{'INF ms':>11}"
        f"{'TARGET%':>11}"
        f"{'L-WRIST%':>12}"
        f"{'R-WRIST%':>12}"
        f"{'ID SW':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{sanitize_model_name(s.model):24}"
            f"{s.avg_processing_fps:>9.2f}"
            f"{s.avg_inference_ms:>11.2f}"
            f"{s.target_present_pct:>11.2f}"
            f"{s.left_wrist_valid_pct:>12.2f}"
            f"{s.right_wrist_valid_pct:>12.2f}"
            f"{s.left_hip_valid_pct:>10.2f}"
            f"{s.right_hip_valid_pct:>10.2f}"
            f"{s.target_id_switches:>8d}"
        )


def main() -> int:
    args = parse_args()

    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[오류] 입력 영상이 없습니다: {video_path}", file=sys.stderr)
        return 2

    if args.frame_stride < 1:
        print("[오류] --frame-stride는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    device = resolve_device(args.device)

    print(f"[입력 영상] {video_path}")
    print(f"[결과 폴더] {output_dir}")
    print(f"[실행 장치] {device}")
    print(f"[비교 모델] {', '.join(args.models)}")

    summaries: List[ModelSummary] = []

    for model_path in args.models:
        try:
            summary = run_model(
                model_path=model_path,
                video_path=video_path,
                output_dir=output_dir,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                kpt_conf_threshold=args.kpt_conf,
                tracker=args.tracker,
                target_timeout=args.target_timeout,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                save_video=not args.no_video,
                show=args.show,
            )
            summaries.append(summary)
        except Exception as exc:
            print(
                f"\n[모델 실패] {model_path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not summaries:
        print("[오류] 성공적으로 완료된 모델이 없습니다.", file=sys.stderr)
        return 1

    save_comparison(summaries, output_dir, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

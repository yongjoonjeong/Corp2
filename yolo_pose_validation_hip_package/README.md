# YOLO Pose 복싱 영상 검증기

## 포함된 영상 정보

- 파일: `realsense_20260805_163743.mp4`
- 해상도: 1280 × 720
- FPS: 30
- 프레임 수: 4,747
- 길이: 약 158.23초

## 1. 설치

가상환경 사용을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

PyTorch GPU 버전은 컴퓨터의 CUDA 환경에 맞춰 별도로 설치해야 할 수 있습니다.

## 2. 빠른 동작 확인

처음에는 전체 영상을 돌리지 말고 300프레임만 테스트하세요.

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt yolo11s-pose.pt \
  --max-frames 300 \
  --no-video
```

모델 파일이 없으면 Ultralytics가 처음 실행할 때 자동으로 내려받습니다.

## 3. 전체 정식 비교

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt yolo11s-pose.pt \
  --tracker botsort.yaml \
  --imgsz 640 \
  --conf 0.25 \
  --kpt-conf 0.35
```

GPU가 아닌 CPU로 강제 실행:

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt yolo11s-pose.pt \
  --device cpu
```

특정 GPU 사용:

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt yolo11s-pose.pt \
  --device 0
```

화면을 보면서 실행:

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt \
  --show
```

## 4. 최신/다른 모델로 비교

설치된 Ultralytics 버전이 해당 모델을 지원한다면 모델명만 바꾸면 됩니다.

```bash
python3 validate_yolo_pose.py realsense_20260805_163743.mp4 \
  --models yolo11n-pose.pt yolo11s-pose.pt yolo26n-pose.pt
```

## 5. 결과 파일

기본 저장 폴더:

```text
yolo_pose_results/
├── model_comparison.csv
├── model_comparison.json
├── environment.json
├── yolo11n-pose/
│   ├── yolo11n-pose_annotated.mp4
│   ├── yolo11n-pose_frames.csv
│   └── yolo11n-pose_summary.json
└── yolo11s-pose/
    ├── yolo11s-pose_annotated.mp4
    ├── yolo11s-pose_frames.csv
    └── yolo11s-pose_summary.json
```

## 6. 핵심 지표 해석

- `avg_processing_fps`: 영상 읽기, 추론, 추적, 기록을 포함한 실제 처리 FPS
- `avg_inference_ms`: 프레임당 YOLO 처리시간
- `target_present_pct`: 고정한 중앙 운동자 Track ID를 유지한 프레임 비율
- `left/right_wrist_valid_pct`: 대상이 검출된 프레임 중 해당 손목 confidence가 기준 이상인 비율
- `both_wrists_valid_pct`: 양쪽 손목이 동시에 유효한 비율
- `target_id_switches`: 대상 재탐색 과정에서 ID가 변경된 횟수
- `wrist_jitter_norm_avg`: 어깨 너비로 정규화한 손목 프레임 간 이동량

`jitter`에는 실제 펀치 이동도 포함되므로 단독으로 정확도를 판단하지 말고, 같은 영상에서 모델 간 상대 비교용으로 사용하세요.

## 7. 모델 선택 기준

로봇 샌드백에서는 다음 순서로 판단하는 것이 좋습니다.

1. 실제 처리 FPS가 목표 속도를 만족하는가
2. 좌우 손목 유효율이 높은가
3. 대상 유지율이 높은가
4. ID 변경 횟수가 적은가
5. 결과 영상을 직접 확인했을 때 손목이 잘못된 위치로 튀지 않는가

예:

- `s-pose`가 충분한 FPS를 유지하면서 손목 유효율이 높으면 `s-pose`
- `s-pose`가 너무 느리고 `n-pose` 성능이 유사하면 `n-pose`
- BoT-SORT에서 ID 변경이 심하면 이후 Deep OC-SORT 비교 진행

## 8. 중요 한계

이 코드는 정답 라벨이 없는 영상에서 수행하는 **실사용 성능 비교 코드**입니다.

다음 값은 측정할 수 있습니다.

- 속도
- 관절 confidence 기반 유효율
- 대상 Track ID 유지
- 모델별 결과 비교

하지만 진짜 키포인트 정확도인 OKS, pose mAP, 픽셀 오차를 구하려면 일부 프레임에 정답 관절 좌표를 직접 라벨링해야 합니다.


## 골반 추적 수정사항

이번 수정본은 COCO Pose의 `left_hip(11)`, `right_hip(12)`을 화면과 CSV에 포함합니다. 무릎과 발목은 표시·평가하지 않습니다. 골반 좌표는 이후 어깨선-골반선 비교, 몸통 기울기, 골반 회전 코칭 지표 개발에 사용할 수 있습니다. 단, 정면 RGB 2D 영상만으로 실제 3D 골반 회전각을 정확히 산출할 수는 없으며 RealSense Depth 결합 시 정확도가 향상됩니다.

<div align="center">

# K.O — Intelligent Robotic Boxing Training System

### AI Vision · ROS 2 · Doosan M0609 기반 지능형 복싱 트레이닝 시스템

> 팀 E-3 | 정용준 · 정진목 · 김승주 · 김윤식

<br>

<a href="docs/images/robot_boxing_hero.png">
  <img src="docs/images/robot_boxing_hero.png" width="86%" alt="K.O Robotic Boxing Training System">
</a>

<br>

![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E?style=flat-square&logo=ros)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat-square&logo=ubuntu&logoColor=white)
![Robot](https://img.shields.io/badge/Robot-Doosan%20M0609-0078D4?style=flat-square)
![Vision](https://img.shields.io/badge/Vision-YOLO11n%20%2B%20MediaPipe-B22222?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Web](https://img.shields.io/badge/Web-Flask-000000?style=flat-square&logo=flask)
![DB](https://img.shields.io/badge/DB-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)

</div>

---

## 팀 구성

| 이름 | 역할 |
|---|---|
| 정용준 | PM, 요구사항 정의, 산출물 조율, 리스크 관리 및 최종 의사결정 |
| 정진목 | Web UI, 데이터 시각화, 사용자 훈련 인터페이스 및 백엔드 연동 |
| 김승주 | 시스템 아키텍처, ROS 2 통합, 세션·데이터 처리 인터페이스 |
| 김윤식 | 로봇 동작, 미트 위치·자세 제어, 힘 감지 및 하드웨어 연동 |

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 설계](#시스템-설계)
3. [주요 기능](#주요-기능)
4. [전체 훈련 흐름](#전체-훈련-흐름)
5. [ROS 2 통신](#ros-2-통신)
6. [개발 환경](#개발-환경)
7. [설치와 빌드](#설치와-빌드)
8. [실행 순서](#실행-순서)
9. [프로젝트 구조](#프로젝트-구조)
10. [설정과 보정 데이터](#설정과-보정-데이터)
11. [검증](#검증)
12. [문제 해결](#문제-해결)
13. [안전](#안전)

---

## 프로젝트 개요

K.O는 3대의 카메라로 사용자의 자세와 주먹 움직임을 인식하고,
사용자별 도달 거리와 실제 타격 중심을 보정한 뒤 Doosan M0609 협동로봇의
미트 위치를 개인에게 맞게 제어하는 ROS 2 기반 지능형 복싱 트레이닝 시스템입니다.

훈련 중에는 Vision, Voice, UI, Robot, Force가 하나의 세션으로 연결되며,
실제 타격 결과는 SQLite에 저장되어 BEST / CHECK와 코칭 리포트 생성에 사용됩니다.

- Front RealSense + Side C270 × 2 기반 3-Camera Vision
- YOLO11n + BoT-SORT 기반 사용자 ID 고정
- MediaPipe Pose 기반 Guard 및 상체 관절 분석
- Camera → Robot BASE 직접 정렬과 3D 좌표 변환
- Force Contact 기반 Reach 보정 + 5회 타격 기반 Mitt Center 보정
- M0609 Weaving, 맞춤형 미트 위치, Hit / Rebound / Return
- Wake Word + OpenAI Whisper STT 기반 비접촉 훈련 제어
- 사용자별 훈련 기록, HitResult, BEST / CHECK 및 코칭 리포트

현재 최종 통합본의 **Production 물리 검증 우선 범위는 JAB / STRAIGHT**입니다.
Hook / Uppercut 관련 코드와 파라미터는 보존되어 있지만, 잽·스트레이트 물리 검증
완료 후 확장하는 것을 기준으로 합니다.

---

## 시스템 설계

<p align="center">
  <a href="docs/images/ko_system_architecture.png">
    <img src="docs/images/ko_system_architecture.png" width="100%" alt="K.O 전체 시스템 아키텍처">
  </a>
</p>

| 구성 요소 | 역할 |
|---|---|
| User UI | 사용자 등록, 신체 정보, 리치 측정, 훈련 선택, 결과 표시 |
| Voice | `Wake Up KO`, OpenAI Whisper STT, Browser Speech Synthesis |
| Vision | 3-Camera 입력, Person ID, Pose, Guard, 3D Position / Velocity |
| ROS 2 Core | SessionBridge, Robot Bridge, Topic / Service, 상태 동기화 |
| Robot / Force | M0609 Weaving, Mitt Positioner, Hit Detection, Rebound / Return |
| Training DB | 사용자 정보, 보정값, 세션 및 HitResult 저장 |
| Coaching | BEST / CHECK, 이전 기록 비교, 이미지 기반 코칭 피드백 |

전체 통합의 중심은 `SessionBridge`입니다. 훈련 세션, 로봇 준비 상태,
사용자 보정, HitTest, HitResult 저장과 다음 타점 진행을 하나의 상태 흐름으로
관리하여 모듈 간 상태 불일치를 줄입니다.

---

## 주요 기능

### 3-Camera Vision과 Guard Ready

Front RealSense와 좌·우 C270 영상을 이용해 사용자를 추적합니다.
YOLO11n + BoT-SORT는 복서의 Person ID와 ROI를 유지하고,
MediaPipe Pose는 ROI 내부에서 손목·팔꿈치·어깨·코 관절을 추론합니다.

Guard 판정은 양 어깨 폭을 Body Scale로 사용하여 양 손목과 코 사이 거리를
정규화하고, 손목 속도까지 함께 확인합니다. 조건을 연속 4프레임 만족하면
Vision 상태가 `READY`로 전환됩니다.

```text
guard_max_wrist_nose_ratio = 1.85
guard_max_speed_body_s     = 0.45
guard_ready_frames         = 4
```

`READY`는 Guard 자세가 안정되었다는 Vision 상태이며 실제 훈련 시작 신호와는
다릅니다. 실제 Countdown은 두 단계의 사용자 보정까지 끝난 뒤
`TRAINING_READY`가 확인되어야 시작됩니다.

### 사용자별 2단계 미트 보정

최종 시스템은 빠른 펀치를 카메라로 끝까지 추종하는 방식에만 의존하지 않고,
훈련 전 실제 사용자 도달 거리와 미트 중심을 먼저 보정합니다.

1차 **Reach Contact Calibration**에서는 Jab-side 팔을 끝까지 뻗은 상태에서
로봇 미트가 Tool +Z 방향으로 저속 접근하고, 실제 Force Contact 위치를
사용자의 Reach 기준으로 저장합니다.

2차 **Five-Hit Force Centering**에서는 사용자가 미트를 5회 타격하고,
각 HitResult의 `hit_x_mm`, `hit_y_mm`을 이용해 미트 중심을 반복 보정합니다.
각 보정 이동 뒤에는 Settled Pose를 다시 획득하고 Wrench Zero를 재설정하며,
5회가 끝나면 사용자별 Mitt Correction을 DB에 저장합니다.

### M0609 Weaving과 미트 제어

대기 중 M0609은 BASE XZ 평면에서 U형 Weaving을 반복합니다.
훈련 시작 요청이 들어오면 Weaving을 Soft Stop하고 Punching Ready 상태로
전환한 뒤 사용자별 보정값을 반영한 미트 자세를 생성합니다.

### Force Hit Detection과 Rebound

RT Force Stack은 실제 외력을 기준으로 타격을 감지하고,
Peak Force, Impulse, Contact Time, Hit Position, Center Error 등의
HitResult를 생성합니다. 타격 이후에는 Compliance 기반 Rebound와 Return을
수행한 뒤 다음 타격 상태로 전환합니다.

### Voice와 Coaching

`Wake Up KO` 호출어 이후 OpenAI Whisper STT를 통해 훈련 명령을 인식합니다.
브라우저 Speech Synthesis를 이용해 사용자 안내를 제공하며, 훈련 종료 후에는
로컬 측정값과 대표 타격 이미지를 이용해 BEST / CHECK와 코칭 피드백을 생성합니다.

---

## 전체 훈련 흐름

<p align="center">
  <a href="docs/images/ko_execution_sequence.png">
    <img src="docs/images/ko_execution_sequence.png" width="100%" alt="K.O 전체 실행 시퀀스">
  </a>
</p>

```text
시스템 실행 및 Hardware Preflight
    ↓
사용자 등록 / 불러오기
    ↓
카메라 정렬 및 Guard Ready
    ↓
M0609 Weaving 정지 / Punching Ready
    ↓
1차 Reach Contact Calibration
    ↓
2차 Five-Hit Force Centering
    ↓
TRAINING_READY
    ↓
Countdown / 실제 훈련
    ↓
Vision Target + Robot Mitt Control
    ↓
Force HitResult / Rebound / Return
    ↓
다음 타격 또는 훈련 종료
    ↓
SQLite 저장 / BEST · CHECK / Coaching Report
    ↓
Weaving Ready 복귀
```

상세 코드 및 모듈 내부 흐름은 아래 최종 통합 플로우차트에서 확인할 수 있습니다.

<p align="center">
  <a href="docs/images/KO_flowchart.jpg">
    <img src="docs/images/KO_flowchart.jpg" width="100%" alt="K.O 전체 통합 플로우차트">
  </a>
</p>

---

## ROS 2 통신

### 주요 구성

| 구성 | 설명 |
|---|---|
| `SessionBridge` | 훈련 상태, 두 단계 보정, HitTest / HitResult 및 다음 타점 흐름 관리 |
| `robot_weaving_node.py` | HOME / Weaving Ready, BASE XZ Weaving, Soft Stop |
| `ui_robot_bridge.py` | UI Command Queue와 ROS Command / Status 연결 |
| Vision Runtime | Guard, Fist State, Robot BASE Position / Velocity 생성 |
| Force Stack | Hit Detection, Compliance, Rebound / Return, HitResult 생성 |
| Flask UI | 사용자 세션, 훈련 모드, 결과 리포트 및 ADMIN 화면 |

### 주요 Vision 출력

```text
/sandbag/fist_state
/sandbag/fist_position_base_mm/left
/sandbag/fist_position_base_mm/right
/sandbag/fist_velocity_base_mm_s/left
/sandbag/fist_velocity_base_mm_s/right
```

로봇 측에서는 좌표값만 사용하는 것이 아니라 `valid`, `measurement_age_ms`,
`position_std_mm` 등의 상태도 함께 확인합니다. 보정이 끝난 뒤에도 Vision은
완전히 제거되지 않고 제한된 범위의 punch-target predictor 입력으로 사용됩니다.

### 상태 흐름

```text
Guard READY
    ↓
Reach Calibration
    ↓
5-Hit Calibration
    ↓
TRAINING_READY
    ↓
WAITING_FOR_HIT
    ↓
HitResult
    ↓
Rebound / Return
    ↓
Next Target / Finish
```

---

## 개발 환경

| 항목 | 내용 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Language | Python 3.10 |
| Robot | Doosan M0609 |
| Front Camera | Intel RealSense D435 / D435i |
| Side Camera | Logitech C270 × 2 |
| Vision | YOLO11n, BoT-SORT, MediaPipe Pose, OpenCV |
| 3D / Filter | Triangulation, RealSense Depth, EKF |
| Voice | openWakeWord, OpenAI Whisper STT, Browser Speech Synthesis |
| UI | Flask, HTML, CSS, JavaScript |
| Database | SQLite |
| AI Coaching | OpenAI Responses API (선택) |

### 외부 선행 조건

- ROS 2 Humble
- Python 3.10, pip, venv
- `colcon`
- RealSense SDK / udev
- C270 V4L2 접근 권한
- Doosan `roboton`, `dsr_msgs2`, `DR_init`, `DSR_ROBOT2`
- GPU 사용 시 NVIDIA Driver

---

## 설치와 빌드

### 1. 실행 권한 설정

프로젝트 루트에서 주요 스크립트에 실행 권한을 부여합니다.

```bash
chmod +x ./*.sh ui/*.sh robot_control/*.sh
```

### 2. 환경 구성 및 Force Workspace 빌드

```bash
./setup.sh --build-force
```

`setup.sh`는 NVIDIA GPU 존재 여부를 확인하여 PyTorch CPU / CUDA 환경을
자동 선택하고 Vision / UI 환경을 준비합니다.

CPU 환경을 강제로 사용할 경우:

```bash
./setup.sh --cpu --build-force
```

CUDA 환경을 강제로 사용할 경우:

```bash
./setup.sh --cuda --build-force
```

### 3. 기본 설정 확인

```bash
source /opt/ros/humble/setup.bash
./.venv/bin/python tools/check_config.py
./.venv/bin/python tools/assign_cameras.py
```

---

## 실행 순서

### USER Mode

실제 실행 전 Doosan `roboton`과 필수 서비스가 정상적으로 준비되어 있어야 합니다.

```bash
./run_final.sh
```

`run_final.sh`는 중복 실행 Lock, Force workspace, Hardware Preflight를 확인한 뒤
USER UI, Vision, Robot Weaving, Force / SessionBridge를 통합 실행합니다.

### ADMIN Mode

```bash
./run_final.sh --admin-mode
```

ADMIN 화면에서는 LEFT / FRONT / RIGHT 카메라, Guard / Impact 상태,
Robot BASE Position / Velocity와 시스템 진단 정보를 확인할 수 있습니다.

### 개발용 부분 실행

```bash
# Vision
./run_vision.sh

# UI + Vision 중심 통합
./run_integrated.sh

# Force Stack
./force_control/run_force_stack.sh
```

### 종료

```bash
./stop_all.sh
```

---

## 프로젝트 구조

```text
KO/
├── calibration/                  # 최종 Robot-world / Intrinsic 결과
├── calibration_tools/            # Camera ↔ Robot BASE Calibration 도구
├── config/                       # Vision / Camera / Mitt 설정
├── docs/                        # 배포·통합·테스트 문서 및 README 이미지
│   ├── images/                   # 시스템·실행·플로우 이미지
│   └── archive/                  # 개발 과정 참고 문서
├── force_control/
│   └── boxing_robot_ws/
│       └── src/
│           ├── boxing_integration/
│           ├── boxing_interfaces/
│           ├── mitt_hit_bringup/
│           └── mitt_hit_system/
├── interfaces/                   # 프로젝트 인터페이스 정의
├── models/                       # YOLO / MediaPipe 모델
├── msg/                          # FistState schema
├── robot_control/                # M0609 Weaving / UI-Robot Bridge
├── sandbag_vision/               # 3-Camera Vision Runtime
├── tests/                        # Vision 회귀 테스트
├── tools/                        # Preflight / 설정 / 배포 도구
├── ui/                           # Flask UI / Voice / Reporting / DB
├── setup.sh
├── preflight.sh
├── run_final.sh
├── run_integrated.sh
├── test_final.sh
├── stop_all.sh
├── FINAL_INTEGRATION.md
├── FINAL_TEST_REPORT.md
└── DEPLOYMENT.md
```

실행 결과, 사용자 DB, API Key, Python 가상환경과 빌드 산출물은 배포본에서
제외하며 다른 환경에서는 새로 구성합니다.

---

## 설정과 보정 데이터

| 항목 | 위치 / 기준 |
|---|---|
| Vision Runtime | `config/runtime.yaml` |
| Camera Intrinsic | `calibration/intrinsics/` |
| Robot-world Transform | `calibration/results/robot_world.yaml` |
| Reach Calibration | 사용자별 Reach 보정 데이터 |
| Mitt Calibration | 사용자별 5-Hit Mitt Correction |
| Training / Hit | SQLite 및 세션별 HitResult |
| OpenAI API Key | `ui/.env` |

카메라 설치 위치나 각도가 달라지면 Camera ↔ Robot BASE Calibration을 다시
검증해야 합니다. 사용자별 Reach / Mitt Calibration은 일반 카메라 캘리브레이션과
구분되는 **훈련 사용자 개인 보정값**입니다.

---

## 검증

### 소프트웨어 검증

```bash
./test_final.sh
```

최종 통합본 기준 자동 검증 결과:

| 검증 | 결과 |
|---|---:|
| Final Integration Contract | **PASS** |
| UI/API + Force/Rebound/Mitt | **154 PASS + 18 subtests PASS** |
| Vision ROS-independent Regression | **21 PASS** |
| Python Compile | **PASS** |
| YAML Parse | **15 files / 0 errors** |
| ROS `package.xml` Parse | **4 files / 0 errors** |
| JavaScript Syntax | **PASS** |

### Hardware Preflight

```bash
./test_final.sh --hardware
```

Hardware Preflight는 실제 프로젝트 Robot Motion을 실행하지 않고 C270,
RealSense RGB / Depth, 마이크, Wake Word, ROS 2, Doosan 필수 서비스와
Force workspace overlay를 확인합니다.

자동 테스트의 `PASS`는 소프트웨어 연결 계약과 로직이 준비되었다는 의미이며,
실제 로봇의 물리 안전성을 인증한다는 의미는 아닙니다.

---

## 문제 해결

| 증상 | 확인 방법 |
|---|---|
| 카메라 좌·우가 바뀜 | C270 Device Role Mapping 확인 |
| 영상 지연이 누적됨 | Latest Frame 처리와 Camera Queue 확인 |
| Guard READY가 안 됨 | 손목-코 거리, 손목 속도, 4-frame 조건 확인 |
| 보정 후 TRAINING_READY가 안 됨 | Reach / 5-Hit Calibration 완료 상태 확인 |
| 오래된 Fist 좌표가 남음 | EKF Stale / timestamp 상태 확인 |
| Hit 후 다음 단계가 진행되지 않음 | HitResult, Return 완료, WAITING_FOR_HIT 확인 |
| 로봇 도착 전 UI가 진행됨 | Robot Ready / SessionBridge 상태 확인 |
| ROS 2 통신 실패 | ROS 환경 source, Domain ID, Topic / Service 확인 |

카메라나 로봇 기준 좌표를 변경한 뒤에는 기존 보정값을 그대로 사용하지 않고
Calibration과 사용자 보정 상태를 다시 확인합니다.

---

## 안전

- 실제 타격 전 HOME / Weaving Ready / Punching Ready를 저속으로 검증합니다.
- Weaving 동작과 사용자 작업영역, 주변 구조물의 간섭을 확인합니다.
- Camera Calibration 또는 사용자 Target Correction 변경 후 단일 펀치부터 확인합니다.
- Reach Contact는 Tool +Z 접근 방향과 Force Threshold를 낮은 속도에서 먼저 검증합니다.
- 5-Hit 보정에서는 각 보정 이동 방향과 Wrench Re-zero 타이밍을 확인합니다.
- Compliance / Rebound 변경 후 약한 타격부터 단계적으로 검증합니다.
- 자동 테스트 PASS와 실제 물리 안전 검증을 동일하게 취급하지 않습니다.
- 현재 Production 검증 우선 범위는 JAB / STRAIGHT입니다.
- 비상 상황에서는 Doosan Robot의 안전 정지 및 비상정지 기능을 우선 사용합니다.

---

## Project Summary

> **K.O는 3-Camera Vision으로 사용자를 인식하고, Force 기반 2단계 개인 미트 보정을 거쳐 Doosan M0609이 사용자에 맞게 반응하며, 실제 HitResult와 코칭 데이터를 하나의 ROS 2 세션으로 연결하는 지능형 로봇 복싱 트레이닝 시스템입니다.**

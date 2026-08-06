# Sandbag 3D 통합 실행본

이 폴더는 다음 구성요소를 한 번에 실행합니다.

- KO 최신 UI, 음성 인식 및 로봇 위빙 제어
- 정면 RealSense와 좌·우 C270의 YOLO11n-Pose 관절 검출
- 캘리브레이션 재투영과 시간 연속성을 이용한 세 화면의 동일 복서 고정
- 세 카메라 관절을 결합한 하나의 3D 궤적
- 준비 자세에 고정된 몸통 좌표계 기반 펀치 분류
- 훅의 바깥 챔버→안쪽 스윕, 어퍼컷의 낮은 챔버→상승 궤적을 이용한 실제 접촉점 선택
- 펀치 점수·관절 증거 JPEG·주먹 좌표 ROS 토픽과 KO UI 연동
- 로봇 BASE 기준 사람 포인트클라우드와 상체 3D 렌더링

배포 ZIP에서는 하위 폴더가 `ui`, `vision`이고 Git 저장소에서는 각각 `KO_UI`,
`sandbag_robot_world_calibration`입니다. 최상위 실행기는 두 구조를 자동 인식합니다.

## 중요한 캘리브레이션 조건

통합본은 다음 측정 결과를 그대로 사용합니다.

```text
vision/calibration/results/robot_world.yaml
vision/calibration/intrinsics/front.yaml
vision/calibration/intrinsics/left.yaml
vision/calibration/intrinsics/right.yaml
vision/config/camera_roles.yaml
```

ChArUco 실물 규격의 기준은 `vision/config/board.yaml`입니다.

```yaml
dictionary: DICT_4X4_50
squares_x: 11
squares_y: 8
square_length_mm: 34.0
marker_length_mm: 20.0
legacy_pattern: true
```

캘리브레이션 후 카메라 위치·각도를 움직이면 외부 캘리브레이션을 다시 해야 합니다. 좌·우 C270은 캘리브레이션 당시 USB 포트에 연결해야 하며 `swap_webcams:=true`를 사용하면 안 됩니다. 최상위 실행기는 실수로 이 옵션을 전달하면 실행을 중단합니다.

## 권장 환경

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble (`/opt/ros/humble`)
- 정면 RGB-D RealSense 1대
- 좌·우 Logitech C270 2대
- GPU는 선택 사항이며 CUDA를 사용할 수 없으면 CPU로 실행할 수 있습니다.

UI와 3D 비전은 서로 다른 가상환경을 사용합니다. 다른 컴퓨터의 `.venv`를 복사해서 사용하지 마세요.

## 최초 한 번 설치

압축 파일을 받은 경우 다음 순서로 진행합니다.

```bash
cd "$HOME/Downloads"
unzip sandbag_3d_integrated.zip
cd sandbag_3d_integrated

chmod +x setup_integrated.sh run_integrated.sh
./setup_integrated.sh
```

설치 과정에서 Ubuntu 패키지 설치를 위한 `sudo` 비밀번호와 Python 패키지 설치를 위한 인터넷 연결이 필요할 수 있습니다. YOLO11n-Pose 모델은 통합본에 포함되어 있습니다.

최초 UI 실행에서는 OpenAI API 키 입력 창이 한 번 표시됩니다. 입력한 키는 `ui/.env`에 저장됩니다.

## 로봇을 포함한 실제 실행

사람 근처에서 로봇을 실행하기 전에 비상정지와 TP 속도 오버라이드를 확인하세요. UI의 현재 위빙 설정에는 빠른 이동값이 포함되어 있으므로 첫 실물 시험은 낮은 속도에서 진행해야 합니다.

터미널 1에서 Doosan bringup을 먼저 시작합니다.

```bash
roboton
```

bringup과 ROS 서비스가 완전히 올라온 뒤, 터미널 2에서 통합본을 실행합니다.

```bash
cd "$HOME/Downloads/sandbag_3d_integrated"
./run_integrated.sh
```

Doosan workspace 자동 탐색이 되지 않을 때만 실제 설치 경로를 지정합니다.

```bash
DSR_WORKSPACE="$HOME/ws_cobot_pjt/ws_dsr" ./run_integrated.sh
```

실행 후 브라우저에서 다음 주소를 엽니다.

```text
http://localhost:5000
```

통합 실행은 UI와 두 ROS 브리지가 항상 같은 서버를 사용하도록
`127.0.0.1:5000`으로 고정합니다. 셸에 남아 있는 `HOST`, `PORT`,
`KO_UI_BASE_URL`, `KO_3D_VISION_RUNNER` 값은 통합 실행에서 사용하지 않습니다.

`Ctrl+C`를 누르면 UI, 로봇/UI 브리지, 3D 비전, KO UI 비전 브리지가 함께 종료됩니다.

## 로봇 없이 UI와 3D 비전 실행

로봇 bringup과 로봇 제어 노드를 제외하고 UI·카메라·비전 연동만 개발하려면 다음처럼 실행합니다.

```bash
KO_WITHOUT_ROBOT=1 ./run_integrated.sh
```

이 모드는 `ui/run_ui_only.sh`와 3D 비전·KO UI 브리지를 시작합니다. 로봇은 필요하지 않지만 실제 3D 영상을 처리하려면 세 카메라와 ROS 2 Humble은 필요합니다.

## 비전 파라미터 전달

최상위 실행기가 `--ros-args`를 자동으로 추가하므로 사용자는 `-p`부터 입력합니다.

```bash
./run_integrated.sh \
  -p pose_device:=cpu \
  -p processing_fps:=8.0 \
  -p point_cloud_stride:=5
```

CUDA GPU 0을 명시하려면:

```bash
./run_integrated.sh -p pose_device:=0
```

`calibration_path`, `camera_role_map`은 통합본이 자동 지정하므로 직접 전달할 수 없습니다. `--ros-args`도 직접 쓰지 않습니다.

## 주요 ROS 토픽

```text
/sandbag/fist_coordinates
/sandbag/form/score
/sandbag/form/joint_evidence/compressed
/sandbag/form/preview/compressed
/sandbag/form/status
```

- `/sandbag/fist_coordinates`: 실시간 좌·우 손목의 RealSense optical frame 및 가능한 경우 로봇 BASE 좌표
- `/sandbag/form/score`: 펀치 종류, 손, 분류 근거, 접촉점, 점수와 위반 관절
- `/sandbag/form/joint_evidence/compressed`: 피드백이 필요한 펀치의 JPEG
- `/sandbag/form/preview/compressed`: KO UI용 경량 3카메라 YOLO 프리뷰 JPEG
- `/sandbag/form/status`: 동일 복서 잠금, 3D 유효성, 가드, 동기화·재투영 품질 JSON

로컬 BASE 3D 맵은 추론 루프가 끊기지 않도록 별도 주기로 갱신합니다. 기본값은
프리뷰 6Hz, 상태 4Hz, 3D 맵 5Hz, RGB-D 포인트클라우드 2.5Hz이며 최근 1.5초의
좌·우 손목 BASE 궤적을 제한된 점 개수로 표시합니다. 지도 범위는 로봇·카메라와
펀치 작업 공간을 기준으로 고정되어 사람의 움직임에 따라 화면이 확대·축소되지
않습니다. 완전한 3D 관절이 잠깐 빠질 때만 골격을 0.4초 유지하고, 동일 복서 잠금이
풀리면 이전 골격·포인트클라우드·궤적을 즉시 지웁니다.

```bash
./run_integrated.sh \
  -p preview_publish_fps:=6.0 \
  -p map_render_fps:=5.0 \
  -p point_cloud_update_fps:=2.5 \
  -p trajectory_trail_seconds:=1.5 \
  -p map_skeleton_ttl_s:=0.4
```

화면에 여러 사람이 있어도 세 카메라의 모든 사람 조합을 무제한 비교하지 않습니다.
각 화면에서 시간 연속성·크기·중앙 위치를 이용해 기본 3명만 먼저 남긴 뒤 캘리브레이션
재투영으로 같은 복서를 결정합니다. 일반적으로 다음 기본값을 유지하세요.

```bash
./run_integrated.sh \
  -p pose_maximum_detections:=5 \
  -p pose_association_candidate_top_k:=3
```

비전 노드는 로봇 이동 명령을 직접 보내지 않습니다. 계산된 접촉점과 점수 토픽을 로봇 제어기가 구독하도록 연결하는 방식입니다.

## 문제 확인

카메라 역할과 연결 상태:

```bash
cd vision  # Git 저장소에서는: cd sandbag_robot_world_calibration
.venv/bin/python 00_check_cameras.py
```

토픽 확인:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /sandbag/form/score
```

분류가 실행되지 않으면 영상 상태창에서 다음 항목을 먼저 확인합니다.

- 세 화면 모두 `YOLO OK`
- 대상 상태가 `LOCKED`
- 동기화 시간이 제한 이내
- 3D 재투영 오차가 허용 범위 이내
- `WAIT_GUARD`에서 양손 guard 게이지가 누적되는지

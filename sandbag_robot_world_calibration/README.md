# 로봇 샌드백 3카메라 → M0609 BASE 캘리브레이션

이 프로젝트는 기존 pairwise 캘리브레이션과 분리해 새로 작성한 코드입니다.

최종적으로 다음 값을 계산합니다.

```text
T_base_front_camera
T_base_left_camera
T_base_right_camera
T_flange_board
```

외부 캘리브레이션의 기본식은 다음과 같습니다.

```text
T_base_flange × T_flange_board
=
T_base_camera × T_camera_board
```

`T_flange_board`도 데이터로 자동 추정하므로 플랜지와 ChArUco 보드 사이의 거리·각도를 줄자로 입력하지 않습니다.

---

## 파일 구성

```text
00_check_cameras.py                 카메라 설정 확인
01_intrinsic_calibration.py         카메라별 내부 캘리브레이션
02_collect_external_samples.py      카메라별 외부 캘리 데이터 수집
03_solve_external_calibration.py    세 카메라와 로봇 BASE 통합 계산
04_validate_robot_world.py          미사용 자세로 외부 캘리 검증
05_robot_world_transform.py         점·벡터·미트 목표 자세 변환
06_enable_manual_teaching.py        직접교시용 MANUAL 모드 전환 보조

config/board.yaml                   ChArUco 실제 규격
config/cameras.yaml                 카메라 종류와 해상도
config/camera_roles.yaml            최초 배정 후 생성되는 C270 좌·우 USB 경로
config/mitt.yaml                    punching TCP 및 미트 법선축 설정
```

---

# 0. 가장 먼저 확인할 것

## ChArUco 규격

현재 기본값은 이전 대화 기준입니다.

```yaml
dictionary: DICT_4X4_50
squares_x: 11
squares_y: 8
square_length_mm: 24.0
marker_length_mm: 15.0
```

- `11×8 squares`는 내부 ChArUco 코너 `10×7`에 해당합니다.
- `square_length_mm=24.0`은 실제 한 칸 크기입니다.
- `marker_length_mm`와 dictionary는 **실제 출력한 보드와 정확히 일치해야 합니다.**
- OpenCV 4.6 이전 방식으로 만든 짝수 행 ChArUco 출력물이라면 `legacy_pattern: true`가 필요할 수 있습니다. 새로 출력한 보드는 기본 `false`를 사용합니다.
- 종이는 평평한 아크릴·MDF·알루미늄 판 등에 부착합니다.

## 카메라 장치 분류

이 버전은 `/dev/video2`, `/dev/video8`처럼 번호를 직접 고정하지 않습니다.

- `front`: `pyrealsense2`로 RealSense 장치를 열고 RGB와 Depth 스트림을 명시적으로 선택
- `left`, `right`: 장치 이름이 `C270`인 **RGB 캡처 노드만** 탐색
- RealSense의 흑백 IR·메타데이터 노드는 C270 후보에서 자동 제외
- 두 C270의 좌우 역할은 `/dev/v4l/by-path/...-video-index0` 경로로 저장되어 재부팅 후에도 유지

전체 코드는 기본적으로 RGB `640×480 @ 30 FPS`를 요청하며, RealSense Depth도 `640×480 @ 30 FPS`를 사용합니다.

---

# 1. Python 환경

프로젝트 루트에서 실행합니다.

```bash
cd sandbag_robot_world_calibration
./setup_python.sh
source .venv/bin/activate
```

이미 OpenCV ArUco, SciPy, PyYAML, pyrealsense2가 설치돼 있으면 기존 환경을 사용해도 됩니다.

ROS/Doosan 서비스를 사용하는 외부 캘리 단계에서는 시스템 ROS 패키지가 보여야 하므로 가상환경을 만들 때 `--system-site-packages`를 사용합니다.

---

# 2. 카메라 역할 최초 배정 및 영상 확인

## 2-1. C270 두 대를 최초 한 번 좌·우로 배정

두 C270은 같은 모델명이므로 프로그램만으로는 어느 장치가 물리적으로 사용자 좌측인지 알 수 없습니다. 다음 명령을 최초 한 번 실행합니다.

```bash
python3 00_check_cameras.py --assign-c270
```

프로그램은 다음 조건을 만족하는 장치만 두 개 찾습니다.

```text
제품명: C270
노드: RGB video-index0
제외: RealSense RGB/Depth/IR/metadata
```

화면에 `C270 candidate 1`, `C270 candidate 2`가 나옵니다.

```text
1 : candidate 1이 사용자 LEFT 카메라일 때
2 : candidate 2가 사용자 LEFT 카메라일 때
Q : 취소
```

선택 결과는 다음 파일에 저장됩니다.

```text
config/camera_roles.yaml
```

USB 허브의 다른 포트로 카메라를 옮겼다면 이 명령을 다시 실행합니다. 단순 재부팅이나 `/dev/videoN` 번호 변경에는 다시 실행할 필요가 없습니다.

탐색된 C270 경로만 텍스트로 확인하려면:

```bash
python3 00_check_cameras.py --list-c270
```

## 2-2. 전체 카메라 확인

```bash
python3 00_check_cameras.py
# 또는
python3 00_check_cameras.py --camera all
```

전체 화면 배치는 사용자 기준입니다.

```text
USER LEFT C270 RGB | FRONT RealSense RGB | USER RIGHT C270 RGB
                      FRONT RealSense DEPTH
```

정면 RealSense만 확인하면 RGB와 RGB 좌표계에 정렬된 Depth가 나란히 표시됩니다.

```bash
python3 00_check_cameras.py --camera front
```

좌우 C270를 각각 확인할 수도 있습니다.

```bash
python3 00_check_cameras.py --camera left
python3 00_check_cameras.py --camera right
```

중요한 점:

- `front`는 `/dev/videoN`을 사용하지 않고 `pyrealsense2`로 RGB+Depth를 선택합니다.
- `left`, `right`는 `config/camera_roles.yaml`에 저장된 C270의 안정적인 USB 포트 경로만 사용합니다.
- left 화면에 RealSense 흑백 영상이 나오는 잘못된 연결은 구조적으로 차단됩니다.
- Depth 영상은 확인 편의를 위해 컬러맵으로 표시되며 실제 데이터는 mm 단위입니다.
- `Q` 또는 `Esc`로 종료합니다.

---

# 3. 내부 캘리브레이션

로봇은 필요 없습니다. 카메라는 흔들리지 않게 두고 보드를 손으로 움직여도 됩니다.

카메라마다 약 30~40장의 서로 다른 영상을 수집합니다.

```bash
python3 01_intrinsic_calibration.py --camera front
python3 01_intrinsic_calibration.py --camera left
python3 01_intrinsic_calibration.py --camera right
```

키:

```text
S : 현재 영상 저장
C : 저장을 마치고 계산
Q : 종료 후 현재 저장 영상으로 계산
```

촬영 조건:

- 중앙, 상하좌우, 네 모서리를 모두 채웁니다.
- 가까운 거리와 먼 거리를 섞습니다.
- 상하·좌우 기울기와 보드 자체 회전을 섞습니다.
- 흐림, 반사, 종이 휨, 거의 같은 자세 반복을 피합니다.

결과:

```text
calibration/intrinsics/front.yaml
calibration/intrinsics/left.yaml
calibration/intrinsics/right.yaml
```

이미지만 다시 계산하려면:

```bash
python3 01_intrinsic_calibration.py --camera front --solve-only
```

권장 확인값:

```text
평균 재투영 오차 0.5 px 이하: 매우 좋음
0.5~0.8 px: 보통 사용 가능
0.8 px 초과: 이미지 분포·흐림·보드 규격 재확인
```

---

# 4. 외부 캘리브레이션 준비

1. 세 카메라를 실제 운용 위치에 단단히 고정합니다.
2. 이후 카메라 위치와 각도를 바꾸지 않습니다.
3. ChArUco 보드를 **플랜지에 단단한 지그로 고정**합니다.
4. 외부 캘리 전체가 끝날 때까지 보드를 분리하거나 다시 장착하지 않습니다.

세 카메라를 공동 계산할 때 `T_flange_board`는 하나의 공통 상수입니다. 카메라별 수집 중간에 보드를 떼었다 다시 달면 이 조건이 깨집니다.

미트 TCP `punching`은 외부 캘리 수집에 사용하지 않습니다. 코드는 항상 `DR_BASE` 기준 **플랜지 자세**를 읽습니다. 위치와 회전 모두 `get_current_tool_flange_posx`에서 가져오므로 현재 선택된 TCP 오프셋에 영향을 받지 않습니다.

---

# 5. 로봇 bringup과 직접교시

터미널에서 기존에 사용하던 M0609 real bringup을 먼저 실행합니다. 다른 터미널에서는:

```bash
cd sandbag_robot_world_calibration
source /opt/ros/humble/setup.bash
source ./source_doosan_ros2.sh
source .venv/bin/activate
```

서비스 확인:

```bash
ros2 service list | grep get_current_tool_flange_posx
```

직접교시용 MANUAL 모드 전환 보조:

```bash
python3 06_enable_manual_teaching.py --namespace /dsr01
```

이 스크립트는 BACKDRIVE를 임의로 활성화하지 않습니다. 컨트롤러가 지원하는 Cockpit/핸드가이드 조작과 현장 안전 절차를 사용합니다.

샘플 저장 순서:

```text
직접교시로 이동
→ 핸드가이드 해제
→ 로봇 흔들림 완전 정지
→ 카메라 창에서 S
→ 로봇 안정성 검사
→ 플랜지 자세와 이미지 저장
```

---

# 6. 외부 캘리 데이터 수집

세 카메라는 보드를 동시에 볼 필요가 없습니다. 각 카메라가 잘 보는 영역으로 로봇을 이동합니다.

## 정면 RealSense

RealSense가 로봇 후면에 있다면 조인트를 회전시켜 보드가 RealSense를 향하게 합니다.

```bash
python3 02_collect_external_samples.py --camera front
```

## 사용자 좌측 C270

```bash
python3 02_collect_external_samples.py --camera left
```

## 사용자 우측 C270

```bash
python3 02_collect_external_samples.py --camera right
```

키:

```text
S : 현재 로봇 자세와 카메라 영상을 한 샘플로 저장
Q : 종료
```

카메라마다 권장 20~30개, 최소 12개 이상의 유효 샘플을 수집합니다.

위치만 바꾸지 말고 회전도 다양하게 변경합니다.

```text
정면
상하 기울기
좌우 기울기
보드 Roll
가까움/멀어짐
화면의 상하좌우 영역
```

너무 비슷한 로봇 자세는 기본적으로 거부됩니다.

저장 위치:

```text
calibration/external_samples/front/<sample_id>/
calibration/external_samples/left/<sample_id>/
calibration/external_samples/right/<sample_id>/
```

---

# 7. 세 카메라와 로봇 BASE 통합 계산

```bash
python3 03_solve_external_calibration.py
```

이 단계에서는 내부 파라미터를 다시 계산하지 않습니다. 앞에서 저장한 `K`, `D`를 고정하고 다음을 공동 최적화합니다.

```text
공통 T_flange_board
T_base_front_camera
T_base_left_camera
T_base_right_camera
```

결과 파일:

```text
calibration/results/robot_world.yaml
```

기본 오차 필터가 너무 엄격한 경우 원인을 먼저 확인한 뒤 임계값을 조절합니다.

```bash
python3 03_solve_external_calibration.py \
  --maximum-reprojection-error-px 1.8 \
  --outlier-translation-mm 15 \
  --outlier-rotation-deg 6
```

무조건 임계값을 크게 하기보다는 흐린 이미지, 보드 유격, 비슷한 회전 자세, 잘못된 보드 규격을 먼저 확인하는 것이 좋습니다.

---

# 8. 반드시 미사용 자세로 검증

캘리브레이션에 쓰지 않은 새 로봇 자세에서 실행합니다.

```bash
python3 04_validate_robot_world.py --camera front
python3 04_validate_robot_world.py --camera left
python3 04_validate_robot_world.py --camera right
```

직접교시 후 정지하고 `S`를 누르면 다음 두 결과를 비교합니다.

```text
로봇 경로: T_base_flange × T_flange_board
카메라 경로: T_base_camera × T_camera_board
```

출력:

```text
translation error: mm
rotation error: deg
```

실제 미트 동작 전에는 여러 검증 자세에서 반복 측정해야 합니다. 정확도 목표는 설치 강성·보드 크기·거리·영상 품질에 따라 달라지지만, 큰 셀 오차가 반복되면 로봇을 움직이지 말고 재캘리브레이션해야 합니다.

---

# 9. 카메라 좌표를 로봇 BASE로 변환

## 점 좌표

RealSense에서 얻은 점이 미터 단위라면:

```bash
python3 05_robot_world_transform.py point \
  --camera front \
  --xyz 0.10 -0.05 0.80 \
  --unit m
```

C270 삼각측량 결과가 mm 단위라면:

```bash
python3 05_robot_world_transform.py point \
  --camera left \
  --xyz 120 -30 850 \
  --unit mm
```

## 방향 벡터

방향 벡터에는 이동값이 적용되지 않고 회전만 적용됩니다.

```bash
python3 05_robot_world_transform.py vector \
  --camera front \
  --xyz 0.1 0.0 0.9
```

## 펀치 지점 + 진행 방향 → 미트 목표 XYZABC

```bash
python3 05_robot_world_transform.py mitt \
  --camera front \
  --point 0.10 -0.05 0.80 \
  --direction 0.02 0.01 0.99 \
  --unit m
```

계산 구조:

```text
카메라 펀치 예상점 → BASE XYZ
카메라 펀치 방향 → BASE 방향 벡터
미트 타격면 법선 = 펀치 진행 방향의 반대
BASE +Z를 기준으로 미트 Roll 안정화
회전행렬 → Doosan ZYZ ABC
```

`config/mitt.yaml` 기본값:

```yaml
tcp_name: punching
tcp_offset_mm_deg: [0, 0, 55, 0, 0, 0]
surface_normal_axis: "+Z"
local_up_axis: "+Y"
```

중요: 미트 타격면의 바깥쪽 법선이 실제로 TCP `+Z`인지 `-Z`인지는 반드시 저속으로 확인해야 합니다. 반대라면:

```yaml
surface_normal_axis: "-Z"
```

로 바꿉니다.

출력 ABC는 동일한 회전행렬을 표현하는 여러 ZYZ 값 중 하나입니다. ABC 성분별 숫자를 직접 빼지 말고 미트 법선 각도와 전체 회전행렬을 비교해야 합니다.

---

## 3카메라 실시간 사람 3D 맵과 주먹 좌표

이 저장소 안의 캘리브레이션 결과와 C270 역할맵을 직접 사용해 실행합니다.

```bash
./run_ros_3d_mvp.sh
```

실행 시 3카메라 영상 창과 `M0609 Robot Base 3D Map` 창이 함께 표시됩니다.
3D 맵에는 로봇 BASE 축, 카메라 위치/시선, RealSense RGB-D 사람 컬러
포인트클라우드, 입체형 상체 아바타와 손목 BASE 좌표가 나옵니다. 배경은
MediaPipe 사람 분할 마스크로 제거합니다. 이 런타임은 로봇 이동 명령을 보내지 않습니다.

포인트클라우드 밀도나 사람 분할 경계를 조정하려면:

```bash
./run_ros_3d_mvp.sh --ros-args \
  -p point_cloud_stride:=3 \
  -p segmentation_threshold:=0.50
```

`point_cloud_stride`가 작을수록 표면이 촘촘하지만 처리량이 증가합니다. 기본값 `4`는
640×480 입력의 약 1/16 픽셀을 사용합니다.

좌표 토픽 확인:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /sandbag/fist_coordinates
```

펀치 결과 확인:

```bash
ros2 topic echo /sandbag/form/score
```

화면 없이 실행하려면:

```bash
./run_ros_3d_mvp.sh --ros-args -p display:=false
```

# 10. RealSense Depth 사용 시 좌표계

외부 캘리브레이션은 RealSense **컬러 카메라 optical frame**을 기준으로 계산합니다.

따라서 실시간 주먹 3D 점도 다음 방식이어야 합니다.

```text
Depth를 color에 align
→ color 픽셀 기준 deprojection
→ front color optical frame의 XYZ
→ T_base_front_camera 적용
```

Depth 원본 optical frame 좌표를 그대로 사용할 경우에는 RealSense depth→color 변환을 먼저 적용해야 합니다.

C270는 Depth가 없어도 내부·외부 캘리브레이션이 가능합니다. 실시간 주먹 깊이는 RealSense Depth 또는 다중 시점 삼각측량으로 계산합니다.

---

# 11. 카메라가 움직였을 때

- 카메라 위치·각도만 바뀜: 외부 캘리브레이션 재수행
- 해상도·렌즈·초점·줌이 바뀜: 내부 캘리브레이션부터 재검토
- 보드 지그가 움직임: 외부 샘플 전체를 다시 수집

---

# 12. 테스트

하드웨어 없이 수학적 좌표식과 미트 법선 계산을 테스트합니다.

```bash
python3 -m pytest -q
```

테스트는 알려진 합성 변환으로부터 `T_flange_board`와 세 카메라의 `T_base_camera`를 다시 복원하는지 확인합니다.

---

## 실물 검증 상태

코드의 문법, 합성 좌표 데이터, 변환 방향은 테스트했습니다. 실제 M0609·세 카메라에서의 서비스 연결, 보드 검출 품질, 최종 위치·회전 정확도는 실물 환경에서 검증해야 합니다. 검증 전에는 계산된 목표 자세로 고속 로봇 이동을 실행하지 마세요.

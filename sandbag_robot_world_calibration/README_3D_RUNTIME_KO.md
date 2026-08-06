# 3카메라 YOLO11-Pose 펀치 분류 런타임

이 패키지는 다음 순서로 한 번의 펀치를 처리합니다.

1. FRONT RealSense와 LEFT/RIGHT C270 프레임을 시간 정렬합니다.
2. 세 화면의 YOLO11n-Pose 후보 중 재투영 오차와 이전 위치가 일치하는 같은 복서를 고릅니다.
3. 어깨·팔꿈치·손목 등 관절을 캘리브레이션 좌표계에서 삼각측량하고 RealSense Depth와 융합합니다.
4. 준비 자세의 몸통 좌표계를 고정한 뒤 하나의 3D 손목 궤적을 만듭니다.
5. 스트레이트는 전방 접촉, 훅은 바깥쪽 챔버 뒤 안쪽 스윕, 어퍼컷은 낮은 챔버 뒤 상승 접촉 프레임을 각각 선택합니다.
6. 접촉점과 자세 점수를 ROS 토픽으로 발행합니다.

## 중요한 전제

- 외부 캘리브레이션 이후 세 카메라의 위치와 각도가 바뀌지 않아야 합니다.
- LEFT/RIGHT C270은 캘리브레이션할 때 사용한 USB 포트에 연결합니다.
- 실행할 때 `swap_webcams:=true`를 사용하지 않습니다. 좌우를 바꾸면 영상은 보여도 3D 좌표가 틀립니다.
- FRONT RealSense는 RGB와 Depth `640x480 @ 30 FPS`를 모두 제공해야 합니다. D430 depth module 단독 장치처럼 RGB 스트림이 없는 장치는 현재 런타임을 실행할 수 없습니다.
- RealSense가 여러 대 연결돼 있다면 런타임용 한 대만 연결하는 것이 안전합니다.

## 처음 한 번 설치

ROS 2 Humble이 설치된 Ubuntu 22.04 PC에서 실행합니다.

```bash
cd "$HOME/Downloads"
unzip sandbag_3d_phase_runtime_v1.zip
cd sandbag_3d_phase_runtime

chmod +x setup_3d.sh run_ros_3d_mvp.sh run_ros_3d_with_calibration.sh
./setup_3d.sh
```

압축 파일에는 YOLO11n-Pose 가중치가 들어 있습니다. `.venv`는 PC마다 다시 만드는 것이 안전합니다.

## 캘리 결과를 보존해서 실행

아래 `CALIB_ROOT`만 동료 PC에서 실제 캘리브레이션 프로젝트가 있는 경로로 바꿉니다.

```bash
cd "$HOME/Downloads/sandbag_3d_phase_runtime"

CALIB_ROOT="$HOME/Downloads/sandbag_robot_world_calibration"
test -f "$CALIB_ROOT/calibration/results/robot_world.yaml"
test -f "$CALIB_ROOT/config/camera_roles.yaml"

./run_ros_3d_with_calibration.sh "$CALIB_ROOT"
```

GPU를 명시하려면 마지막 명령에 ROS 파라미터만 이어 붙입니다.

```bash
./run_ros_3d_with_calibration.sh "$CALIB_ROOT" \
  -p pose_device:=0
```

GPU 호환 문제가 있거나 CPU로 확인하려면:

```bash
./run_ros_3d_with_calibration.sh "$CALIB_ROOT" \
  -p pose_device:=cpu
```

## 발행 토픽

- `/sandbag/form/score`: 종류, 점수, 선택된 접촉점, 품질 지표를 담은 JSON 문자열
- `/sandbag/fist_coordinates`: 실시간 주먹 3D 좌표 JSON 문자열
- `/sandbag/form/joint_evidence/compressed`: 피드백이 필요할 때 표시된 JPEG

`robot_world.yaml`에 `T_base_front`가 정상 저장되어 있으면 접촉점 JSON에 `robot_base_mm` 좌표도 함께 들어갑니다.

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /sandbag/form/score --once
ros2 topic echo /sandbag/fist_coordinates
```

## 화면에서 먼저 확인할 것

- LEFT/FRONT/RIGHT 영상에 같은 복서의 스켈레톤이 표시되는지
- `SYNC`가 대체로 35 ms 이내인지
- `TARGET`이 `LOCKED`인지
- `REPROJ`가 갑자기 커지지 않는지
- 펀치 후 표시되는 타격점이 실제 접촉 위치인지

실측 영상 없이 2D 임계값을 3D에 그대로 숫자 복사한 것은 아닙니다. 모든 거리값은 준비 자세의 3D 어깨너비로 정규화하지만, 첫 현장 실행에서는 훅/어퍼컷의 3D 궤적 로그를 보고 소폭 재조정할 수 있습니다.

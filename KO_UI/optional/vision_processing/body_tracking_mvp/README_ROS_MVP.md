# 웹캠 펀치 자세 피드백 ROS 2 MVP

웹캠 한 대와 MediaPipe Pose를 사용해 펀치 동작 한 번을 검출하고 다음 두 토픽을 발행한다.

- `/sandbag/form/score` (`std_msgs/msg/String`): 펀치 종류, 손, 최종 점수, 관절별 오차가 담긴 JSON
- `/sandbag/form/joint_evidence/compressed` (`sensor_msgs/msg/CompressedImage`): Threshold를 넘겼을 때 잘못된 관절을 표시한 JPEG

점수 토픽은 검출된 펀치마다 한 번 발행한다. JPEG 토픽은 최종 점수가 낮거나 관절 오차가 Threshold를 넘은 경우에만 펀치당 한 번 발행한다.

최종 점수가 `30.0` 미만이면 동일한 관절 표시 JPEG를 `feedback_images/`에도 저장한다. 정확히 30점인 경우에는 저장하지 않는다.

관절 표시 JPEG의 펀치 정보와 피드백 문구는 한글로 렌더링한다.

## 설치

필수 환경은 Ubuntu 22.04, Python 3.10, ROS 2 Humble과 웹캠이다. ZIP을 원하는 위치에 압축 해제한 다음 최초 한 번만 설치 스크립트를 실행한다.

```bash
unzip body_tracking_mvp.zip
cd body_tracking_mvp
./setup.sh
```

`setup.sh`는 현재 프로젝트 폴더 안에 `.venv`를 만들고 검증된 버전의 Python 패키지를 설치한다. ROS 2 Humble의 `rclpy`, `std_msgs`, `sensor_msgs`는 시스템에 설치된 것을 사용한다.

## 실행

```bash
cd body_tracking_mvp
./run_ros_mvp.sh
```

다른 카메라 인덱스를 사용하려면:

```bash
./run_ros_mvp.sh --ros-args -p camera_index:=2
```

기본 실행 화면은 상태 패널 `960x138`과 상태창에 가리지 않는 전체 카메라 화면 `960x720`을 위아래로 합쳐 표시한다. 처리 해상도는 기존 `640x480`을 유지한다. 더 크게 또는 작게 표시하려면 카메라 표시 크기를 변경한다.

```bash
./run_ros_mvp.sh --ros-args \
  -p display_width:=1280 \
  -p display_height:=960
```

`display_height`는 상태 패널을 제외한 카메라 화면 높이다. 전체 창 높이는 `display_height + status_panel_height`이다.

화면 없이 실행하려면:

```bash
./run_ros_mvp.sh --ros-args -p display:=false
```

저장 폴더를 변경하려면:

```bash
./run_ros_mvp.sh --ros-args \
  -p feedback_image_dir:="$PWD/my_feedback_images"
```

종료는 영상 창에서 `q` 또는 `Esc`, 터미널에서는 `Ctrl+C`를 사용한다.

## 사용 방법

1. 상체와 양손, 골반이 모두 웹캠 화면에 나오도록 선다.
2. 가드 자세로 잠시 멈춰 화면 상태가 `READY`가 되도록 한다.
3. 스트레이트, 훅 또는 어퍼컷 한 번을 수행한다.
4. 손이 다시 느려지면 한 번의 펀치가 확정되고 점수 토픽이 발행된다.
5. Threshold를 넘긴 경우 표시된 JPEG도 발행된다.

펀치 확정 후 쿨다운 동안 타격 손목 위치에 펄스 타격점이 표시된다. 타격점 아래에는 30점 이상이면 작은 초록색 `GOOD!`, 미만이면 작은 빨간색 `BAD..`가 나타난다. 상태 패널의 펀치 종류는 스트레이트 빨강, 훅 초록, 어퍼컷 파랑으로 표시된다.

`READY`는 손이 펀치 속도로 가속되는 동안 유지된다. 양손 디버그 줄에는 다음 값이 표시된다.

상단 `GUARD 0/6` 게이지는 안정적인 가드가 누적된 프레임 수를 표시한다. READY가 되면 게이지가 초록색으로 가득 차고 `GUARD READY`로 바뀐다.

- `SPD`: 최근 프레임 중앙값을 적용한 손목 속도
- `MOVE`: READY 시점의 가드 손목에서 이동한 거리
- `ELB+`: READY 시점보다 팔꿈치가 펴진 각도
- `START`: 시작 조건이 연속으로 확인된 프레임 수

현재 분류는 2D 웹캠용 임시 규칙이다. 전방 깊이를 직접 측정할 수 없으므로 스트레이트와 짧은 훅이 혼동될 수 있다.

분류에서는 ACTIVE 전체를 그대로 사용하지 않고 평면 도달 거리, 전방 깊이, 팔꿈치 신전의 첫 peak로 타격 시점을 추정한다. 그 이후 손이 가드로 복귀하는 프레임은 제외하므로 Straight의 회수 동작이 바깥쪽→안쪽 Hook으로 계산되지 않는다.

Straight는 시작점에서 타격점까지의 손목 경로가 직선인지 우선 확인한다. 따라서 타깃이 카메라 정중앙이 아니라 좌우에 있어도 직선 경로라면 Straight가 될 수 있다. Hook은 좌우 이동이 우세하면서 경로가 휘거나 타격 전 안쪽으로 꺾이고, 팔꿈치가 굽어 있는지를 사용한다. 곡률이 충분히 강하면 2D에서 팔이 크게 펴져 보여도 Hook을 우선한다. Uppercut은 위쪽 이동 우세와 손목-팔꿈치 높이를 계속 사용한다.

점수 JSON의 `classification_reason`과 `motion_features`에서 `path_linearity`, `path_curvature_ratio`, `direct_travel_ratio`, `impact_sample_index`, `recovery_frames_ignored`, `straight_linear_path_pass`, `hook_priority_pass` 등을 확인할 수 있다. 실행 로그에도 판정 사유와 직선성/곡률이 한 줄로 출력된다.

쿨다운 동안 타격점 아래에 통과 시 작은 초록색 `GOOD!`, 실패 시 작은 빨간색 `BAD..`가 표시된다. 통과 기준은 YAML의 `feedback.pass_score_threshold`이다.

## 토픽 확인

다른 터미널에서:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /sandbag/form/score
```

```bash
source /opt/ros/humble/setup.bash
ros2 topic info /sandbag/form/joint_evidence/compressed --verbose
```

`rqt_image_view`가 설치되어 있다면 JPEG 토픽을 선택해 표시할 수 있다.

```bash
source /opt/ros/humble/setup.bash
rqt_image_view
```

## 임시 정자세와 Threshold 조정

[config/temporary_form_reference.yaml](config/temporary_form_reference.yaml)에서 다음 값을 조정한다.

- `detector.guard_speed`: WAIT_GUARD에서 안정적인 가드로 인정할 최대 속도
- `detector.start_speed`: 펀치 시작 손목 속도
- `detector.end_speed`: 펀치 종료 손목 속도
- `detector.speed_window`: 속도 중앙값을 계산할 최근 프레임 수
- `detector.start_displacement_ratio`: 가드 기준 최소 손목 이동 거리
- `detector.start_extension_deg`: 가드 기준 최소 팔꿈치 신전 각도
- `detector.start_confirm_frames`: 시작 조건을 연속 확인할 프레임 수
- `detector.guard_max_wrist_face_ratio`: 가드로 인정할 얼굴-손목 최대 거리
- `classification`: 스트레이트·훅·어퍼컷 임시 분류 기준
- `feedback.score_threshold`: 이 점수 미만일 때 JPEG 발행
- `feedback.joint_error_threshold`: `실제 오차 / 허용 오차`의 관절별 기준
- `feedback.save_score_threshold`: 이 점수 미만일 때 표시된 JPEG를 로컬에 저장
- `feedback.pass_score_threshold`: 이 점수 이상이면 `GOOD!`, 미만이면 `BAD..` 표시
- `profiles.*.features`: 펀치 종류별 목표값, 허용오차, 가중치, 표시할 관절

비율 특징은 화면 픽셀이 아니라 사용자의 어깨너비로 정규화한다. YAML의 현재 수치는 동작 검증용 임시값이며, 정자세 영상을 수집한 뒤 평균과 허용 범위로 교체해야 한다.

증상별 조정 순서는 다음과 같다.

- `WAIT_GUARD`에서 READY가 되지 않음: `guard_speed` 또는 `guard_max_wrist_face_ratio`를 조금 높인다.
- READY지만 펀치가 시작되지 않음: 화면의 `SPD`, `MOVE`, `ELB+`를 보고 `start_speed` 또는 `start_displacement_ratio`를 낮춘다.
- 작은 손 움직임이 펀치로 잡힘: `start_speed`, `start_displacement_ratio` 또는 `start_confirm_frames`를 높인다.
- `ACTIVE`가 너무 오래 유지됨: `end_speed`를 높이거나 `end_frames`를 낮춘다.

## 점수 JSON 예시

```json
{
  "punch_id": 1,
  "punch_type": "straight",
  "punch_side": "right",
  "classification_confidence": 0.81,
  "total_score": 68.4,
  "feedback_required": true,
  "violations": [
    {
      "joint": "guard_wrist",
      "code": "guard_dropped",
      "error_ratio": 1.23
    }
  ]
}
```

실제 메시지에는 동작 분류 특징과 채점에 사용된 모든 특징의 값도 함께 포함된다.

## 테스트

웹캠 없이 핵심 검출·분류·채점 로직을 검사한다.

```bash
cd body_tracking_mvp
./.venv/bin/python -m unittest discover -s tests -v
```

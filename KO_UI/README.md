# KO SPORTY UI + 연속 음성 대화 + 자동 U자 위빙 — V3

이 통합본은 `KO_UI_SPORTY_VOICE_SESSION_V2`의 UI·사용자 등록·SQLite·Wake Word·Whisper STT·연속 음성 세션 기능을 유지하면서, 최신 자동 시작 위빙 노드를 반영한 버전입니다.

## 실제 동작 흐름

1. 터미널 1에서 사용자가 직접 `roboton`을 실행합니다.
2. 터미널 2에서 `./run.sh`를 실행하면 UI, ROS 브리지, 위빙 노드가 함께 시작됩니다.
3. 로봇이 HOME `[-180, 0, 90, -180, -90, 0]`으로 이동합니다.
4. 별도의 Wake Word 입력 없이 위빙 준비 자세 `[-180, 0, 90, -270, 90, 0]`으로 이동합니다.
5. 준비 자세 도착이 완료되면 해당 TCP를 기준으로 U자 위빙을 자동 반복합니다.
6. 사용자가 `웨이크 업 케이오`라고 말하면 `/wakeword_detected=True`가 전달됩니다.
7. 로봇은 현재 위빙을 Soft Stop하고 위빙 준비 자세로 복귀합니다.
8. 준비 자세에서 사용자의 후속 STT 명령을 기다립니다.
9. `오른손 스트레이트 1분 훈련` 같은 훈련 명령이 확정되면 `/robot_boxing/action_command`를 거쳐 `/robot_boxing/action_ready`로 전달됩니다.

```text
프로그램 실행
→ HOME 이동
→ 위빙 준비 자세 이동
→ U자 위빙 자동 시작
→ Wake Word 인식
→ 위빙 Soft Stop
→ 위빙 준비 자세 복귀
→ 연속 STT 명령 대기
```

## 최초 설치

```bash
chmod +x setup.sh run.sh run_ui_only.sh run_vision_optional.sh robot_control/*.sh
./setup.sh
```

## 실행

터미널 1:

```bash
roboton
```

두산 bringup이 완전히 올라온 뒤 터미널 2:

```bash
./run.sh
```

브라우저:

```text
http://localhost:5000
```

최초 실행에서는 OpenAI API 키만 한 번 입력합니다. `.env`에 Wake Word나 로봇 설정을 추가로 작성할 필요는 없습니다.

## 로봇 설정

- 자동 위빙 시작: `AUTO_START_WEAVING_ON_STARTUP = True`
- HOME: `[-180, 0, 90, -180, -90, 0]`
- 위빙 준비: `[-180, 0, 90, -270, 90, 0]`
- 별도 접근 자세: 사용하지 않음
- U자 X 범위: 준비 위치 기준 `0 ~ -170 mm`
- U자 깊이: `68 mm`
- 위빙 속도: `WEAVE_VEL = [450.0, 15.0]`
- 위빙 가속도: `WEAVE_ACC = [900.0, 30.0]`
- 한 번의 `movesx()`에 4왕복, 총 80개 경로점 사용

> `450 mm/s`는 사람 근처에서 매우 빠른 설정입니다. 첫 실물 시험은 TP 속도 오버라이드를 낮추고 비상정지를 즉시 사용할 수 있는 상태에서 진행해야 합니다.

## Wake Word 역할

이 버전에서 Wake Word는 위빙 시작 신호가 아닙니다.

```text
기존 방식
Wake Word → 위빙 시작

현재 방식
프로그램 시작 → 위빙 자동 시작
Wake Word → 위빙 정지 및 준비 자세 복귀
```

호출어가 감지되면 로봇은 `STOPPING_FOR_VOICE` 상태를 발행하고 준비 자세로 복귀합니다.

## 1회 호출 · 1회 음성 명령

호출 한 번에 명령을 하나만 처리합니다. UI 프로세스를 처음 실행했을 때는 전체
호출어 `웨이크 업 케이오`를 말한 뒤 명령 하나를 말합니다. 명령 처리 직후 세션은
종료됩니다. 이후 명령부터는 짧게 `케이오`라고 부른 뒤 명령 하나를 말합니다.
명령이 끝날 때마다 다시 `케이오` 호출 대기로 돌아갑니다. UI를 재시작하면 다시
최초 호출어 `웨이크 업 케이오`부터 시작합니다.

```text
웨이크 업 케이오
→ 새 사용자 등록
→ 이름은 정우
→ 키는 175센티
→ 오른손잡이야
→ 저장하고 측정
```

리치 측정과 실제 훈련 중에는 음성 세션이 더 길게 유지됩니다. `대화 종료` 또는 `음성 대기 모드로 돌아가`라고 말하면 호출어 대기 상태로 복귀합니다.

## 수동 확인

위빙을 정지하고 준비 자세로 복귀시키기:

```bash
ros2 topic pub --once /wakeword_detected \
  std_msgs/msg/Bool "{data: true}"
```

위빙 다시 시작:

```bash
ros2 topic pub --once /robot_boxing/weave_command \
  std_msgs/msg/String "{data: '위빙 시작'}"
```

훈련 동작 전환:

```bash
ros2 topic pub --once /robot_boxing/action_command \
  std_msgs/msg/String "{data: '오른손 스트레이트 1분 훈련'}"
```

상태 확인:

```bash
ros2 topic echo /robot_boxing/weave_state
```

예상 시작 상태 흐름:

```text
MOVING_HOME
→ IDLE_HOME
→ MOVING_WEAVE_READY
→ WEAVING
```

Wake Word 인식 후:

```text
STOPPING_FOR_VOICE
→ RETURNING_WEAVE_READY
→ READY
```

훈련 명령 전환 시:

```text
STOPPING_FOR_ACTION
→ RETURNING_WEAVE_READY
→ ACTION_READY
```

## 비전

비전은 카메라 오류가 UI나 로봇을 종료시키지 않도록 기본 실행에서 분리되어 있습니다.

사용자 리치 측정에 사용하는 MediaPipe Tasks Vision `0.10.14`와 Pose Landmarker
Lite 모델은 `static/vendor/mediapipe/`에 포함되어 있습니다. 리치 측정은 인터넷이나
외부 CDN 없이 브라우저에서 로컬 JS·WASM·모델 파일로 실행됩니다.

양팔 리치는 손가락을 편 자세에서 손목과 검지/새끼손가락 끝 랜드마크 중 더 먼
지점을 사용합니다. 좌·우 펀치 리치는 어깨→팔꿈치→손목 길이에 손목부터 주먹
앞면까지의 고정 보정값 9cm를 별도로 더합니다.

```bash
./run_vision_optional.sh --ros-args -p camera_index:=0
```

## ROS 환경 점검

로봇 노드는 UI의 `.venv`가 아니라 `/usr/bin/python3`로 실행됩니다. 실행 전에 환경만 확인하려면:

```bash
./robot_control/check_robot_env.sh
```

Doosan workspace 자동 탐색이 실패하면 해당 실행에만 workspace를 지정할 수 있습니다.

```bash
DSR_WORKSPACE=~/doosan_ws ./run.sh
```

## UI

메인 화면은 블랙·레드 복싱 콘셉트이며, 장치 연결 상태는 `시스템 설정`에서 작은 상태 항목으로만 표시됩니다. 사용자 등록, 리치 측정, 훈련 기록, Wake Word, Whisper STT, 연속 음성 조작 기능은 V2와 동일하게 유지됩니다.

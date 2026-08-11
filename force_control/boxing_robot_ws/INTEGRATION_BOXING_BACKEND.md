# KO Vision/UI + 미트 로봇 통합

ZIP 배포본은 `sandbag_vision_realtime_v3_deploy_20260807_r2/`에 배치했다.
Vision 및 캘리브레이션 설정값과 기존 로봇 YAML 파라미터는 변경하지 않았다.

## 동작 순서

```text
시스템 시작 → HOME에서 대기
→ Wake Word 감지 → 위빙 시작
→ 이름·생년월일 조회/등록 → 키·리치 확인 → 훈련 선택
→ 위빙 정지 → X=0 사용자 기준 미트 자세 준비
→ /mitt/start_test → 기준 설정 → WAITING_FOR_HIT
→ 훈련 → /mitt/stop_test → 기록 저장
→ 위빙 재시작 상태에서 결과 및 이전 기록 피드백
```

UI 안내 문구는 다음과 같다.

```text
미트를 치거나 누르지 마세요.
로봇이 기준을 설정하고 있습니다.
```

## 빌드

```bash
cd ~/boxing_robot_ws
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash
colcon build --symlink-install
```

## 실행

Doosan `roboton`을 먼저 실행한 뒤:

```bash
cd ~/boxing_robot_ws
./run_full_system.sh
```

음성 입력은 API 키를 등록한 뒤 활성화한다.

```bash
cd ~/boxing_robot_ws/sandbag_vision_realtime_v3_deploy_20260807_r2
python3 ui/configure_api_key.py
cd ~/boxing_robot_ws
KO_ENABLE_WAKEWORD=1 ./run_full_system.sh
```

API 키와 음성 환경이 준비된 현재 구성에서는 `run_full_system.sh`가 음성 기능을
자동 활성화한다. 필요할 때만 `KO_ENABLE_WAKEWORD=0`으로 끌 수 있다.

`mitt_positioner.params.yaml`의 현재 `allow_real_motion` 값을 그대로 사용한다.
실제 로봇 작업영역이 확보되지 않은 상태에서는 실행하지 않는다.

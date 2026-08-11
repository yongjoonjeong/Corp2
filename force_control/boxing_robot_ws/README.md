# M0609 AI 복싱 미트 로봇

두산 M0609에 장착한 평면 미트의 RT 외력으로 타격을 감지하고, 타격 위치·방향·
정확도와 순응 복귀 상태를 기록하는 ROS 2 Humble 프로젝트입니다. 비전 팀은 사용자
자세와 예상 타격점 좌표를 제공하고, 이 저장소는 로봇의 RT 타격 처리와 세션 단위
순응제어를 담당합니다.

## 현재 장착값

- TCP `punching`: `[0, 0, 55, 0, 0, 0] mm/deg`
- Tool `punching_weight`: `0.480 kg`
- TP 자동 측정 무게중심: `[18.740, -70.600, -123.310] mm`
- Tool Shape: `punching_shape`
- 미트: `190 x 150 x 50 mm`
- Tool Shape Point 1: `[-75, -95, -50] mm`
- Tool Shape Point 2: `[75, 95, 0] mm`

TP가 자동 측정한 Tool Weight/CoG와 등록된 TCP는 애플리케이션 코드에서 변경하지
않습니다. 미트 정면 기준 `+X=RIGHT`, `-X=LEFT`, `+Y=UP`, `-Y=DOWN`, 누르는
법선 방향은 `-Fz`입니다.

## 운영 구조

실제 운영 경로는 `hit_analyzer` 하나입니다. 순응제어는 펀치마다 켰다 끄지 않고
활성 세션 전체에서 계속 유지합니다.

```text
노드 실행
  -> 3초 무접촉 영점 보정(순응 OFF)
  -> READY
  -> /mitt/start_test
  -> STATE_STANDBY 및 서비스 확인
  -> Task Compliance 1회 활성화
  -> STABILIZING_COMPLIANCE
       -> 무접촉 정지 상태 유지
       -> 활성화 후 3초 wrench 기준 재측정
       -> 안정된 TCP를 세션 복귀 기준으로 재설정
  -> WAITING_FOR_HIT
       -> ANALYZING
       -> RETURNING_TO_REFERENCE
       -> WAITING_FOR_HIT (다음 타격, release 없음)
  -> 목표 횟수 완료 또는 /mitt/stop_test
  -> compliance release 1회
  -> TEST_COMPLETE 또는 READY
```

정상 경로에서 release하는 시점은 세션 종료뿐입니다.

- 목표 타격 횟수 완료
- 명시적인 `/mitt/stop_test`
- 활성 세션 중 노드 종료

RT stream timeout, 로봇 상태 이상, TCP 병진·회전 변위 초과, 원시 총힘·총토크 초과,
활성화 안정화 timeout, 복귀 timeout 같은 안전 fault는 세션을 강제 종료하고 즉시
release합니다. 이는 정상 수명주기의 예외이며 제거하지 않습니다. release 최종 실패
뒤에는 재시작을 잠급니다.

## 현재 구현

- 약 `250 Hz` Doosan RT 데이터 수집
- 동일 RT 응답의 wrench·TCP pose·TCP velocity·robot state를 묶은
  `/mitt/rt_sample` 발행
- BASE wrench를 Euler ZYZ TCP 자세로 Tool 좌표에 보정
- 안정 영점, 접촉 시작·종료, debounce, 긴 누름 제외
- peak force, 접촉 시간, impulse 계산
- 미트 타격 위치·방향과 정확도 점수 계산
- 타격별 JSON/CSV 저장
- 명시적인 Start/Stop과 목표 횟수 자동 종료
- 세션 단위 continuous compliance
- 순응 활성화 후 정지 확인과 TCP/wrench 기준 재측정
- 세션 JSON에 기준 wrench 평균·표준편차·3-sigma 범위와 TCP 기준 저장
- 세션 전체 최대 TCP 병진·회전 변위와 절대 총힘·총토크 및 종료 원인 저장
- 타격/복귀 판정은 고정된 세션 기준과의 편차 사용
- Ctrl+C/SIGTERM에서도 executor가 살아 있는 동안 shutdown release 완료
- 타격 뒤 기준 위치 복귀 판정과 복귀 전 다음 타격 차단
- RT timeout·상태·TCP 병진/회전 변위·원시 총힘·총토크 watchdog
- 제한 재시도 release와 실패 후 Start 잠금
- 읽기 전용 compliance preflight

개발 중 사용한 일회성 `compliance_commissioning`과 `compliance_return_probe` 실행기는
운영 수명주기와 충돌하므로 제거했습니다. 당시 실측 JSON/CSV는 분석 증거로 `data/`에
보존합니다.

## 중요한 현재 제한

2026-08-05 실측에서 Tool은 `punching_weight`, 로봇은 `STATE_STANDBY(1)`였지만
무접촉 Tool-force가 약 `6.5 N`, `1.29 Nm`로 남았습니다. 순응 전환 시험은 다음처럼
재현성이 없었습니다.

- PASS: 1초 전환 + 2초 유지, 최대 변위 `0.018 mm`
- FAIL: 전환 `0.316초`에 변위 `0.515 mm`, 최대 속도 `6.32 mm/s`
- 다른 FAIL: 영점 모멘트 표준편차 초과로 활성화 전 차단

애플리케이션 타격 판정은 순응 활성화가 안정된 뒤 잔류 wrench를 다시 측정하고 그
편차를 사용하므로, 일정한 잔류값 자체를 0으로 만들 필요는 없습니다. 이 기준은
타격과 복귀 중에는 갱신하지 않습니다. 단, Doosan 내부 순응제어는 애플리케이션의
편차값을 받지 않으므로 전환 중 움직임 위험은 별도입니다. 그래서 전환 전 TCP 기준의
변위 상한과 원시 총힘·총토크 상한은 계속 적용합니다.

기본 운영 설정의 compliance는 여전히 fail-closed입니다. 전환 안정 조건, 절대 상한,
복귀 tolerance를 무접촉 시험으로 검증하기 전에는 `compliance_enabled: true`로 바꾸지
않습니다. 관련 값은 YAML에서 `0.0`으로 두어 실수로 활성화되지 않게 했습니다.
Tool/TCP/CoG 값을 코드에서 덮어써서 문제를 숨기지 않습니다.

## 빌드와 테스트

```bash
cd ~/boxing_robot_ws
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash

ROS_DOMAIN_ID=177 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test \
  --packages-select mitt_hit_system
colcon test-result --verbose
```

실제 로봇은 `ROS_DOMAIN_ID=77`, 테스트는 `177`을 사용합니다.
현재 기대 결과는 `115 tests, 0 errors, 0 failures, 0 skipped`입니다.

## 현재 안전한 실행

터미널 1에서 Doosan 드라이버를 실행한 뒤 터미널 2에서 RT 스트림을 실행합니다.

```bash
cd ~/boxing_robot_ws
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash
source install/setup.bash

ros2 launch mitt_hit_bringup rt_force_diagnostic.launch.py
```

터미널 3에서 읽기 전용 preflight를 실행합니다.

```bash
cd ~/boxing_robot_ws
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash
source install/setup.bash

ros2 run mitt_hit_system compliance_preflight
```

`compliance_no_contact_session.params.yaml`은 운영 기본값이 아니라 다음 무접촉 수명주기
시험 전용 override입니다. 이 파일을 사용한 동안에는 미트를 치거나 누르지 않습니다.
현재 override는 Tool 기준 비등방 강성
`[15000, 15000, 5000, 500, 500, 500]`을 사용합니다. 이는 무접촉 안정화 확인값이며
실타격 승인값이 아닙니다. TCP 병진 변위는 `1.0 mm`, 실제 ZYZ 회전행렬 간 최단
회전각은 `0.30 deg`로 제한합니다.

## 연속 펀치·반발 모드

`punch_rebound.launch.py`는 RT 수집 노드가 이미 실행 중일 때 분석 노드만 시작합니다.
Compliance는 세션 전체에서 계속 활성화하지만 Tool Z 강성은 상태별로 전환합니다.
대기/복귀에서는 `5000 N/m`, 접촉이 `10 N`을 넘으면 `set_stiffnessx`로 `500 N/m`,
접촉 종료 뒤 다시 `5000 N/m`로 선형 전환합니다. X/Y는 `15000 N/m`, 회전은
`800 Nm/rad`로 유지합니다.

활성화 중에는 `5 mm / 0.30 deg / 10 N / 3 Nm`의 좁은 watchdog을 적용합니다.
안정화와 TCP 기준 재측정이 완료된 뒤에만 세션 한계
`60 mm / 2 deg / 95 N / 8 Nm`로 전환합니다. 이 값은 애플리케이션 감시값이며
Doosan의 인증 안전 설정을 대체하거나 변경하지 않습니다.
티치펜던트 실측 한계는 일반 모드 `96 N`, 감소 모드 `48 N`이므로 이 프로파일은
Start 직전에 `GetRobotSpeedMode`를 읽고 `NORMAL(0)`일 때만 시작합니다. RT 샘플
watchdog은 실측된 ROS/DDS 지연을 반영해 `250 ms`이며, timeout이면 계속 fail-closed로
compliance를 release합니다.

실타격 프로파일은 compliance를 유지한 채 첫 `10 N` 접촉 샘플에서 즉시 비동기 직선
`MoveLine`으로 Tool -Z 방향 `50 mm` 후퇴를 명령하고, 목표 도달 뒤 기준 TCP로
부드럽게 복귀합니다. 후퇴는 실제 연속 시험에서 검증한 `25 mm/s, 50 mm/s²`,
복귀는 `20 mm/s, 40 mm/s²`입니다. 복귀 중 새 접촉이 감지되면
`DR_SSTO(2)` 소프트 정지로 현재 직선 이동만 중단하고 새 타격을 수용합니다.
`MoveJ`, TCP/Tool/CoG 변경, 충돌 민감도 및 인증 안전한계 변경은 사용하지 않습니다.
95 N 앱 상한은 일반모드 TP TCP 힘 한계 96 N보다 낮으며, 컨트롤러 정지를 무력화하지
않습니다. 후퇴 도달은 순응으로 생기는 횡방향 편차와 분리해 명령 축 진행량으로
판정하고 전체 병진·회전 watchdog은 계속 유지합니다.

`80 mm/s, 400 mm/s²` 후퇴 시험은 최대 힘 `51.57 N` 상태에서도 후퇴 속도
`83.11 mm/s`에서 `STATE_SAFE_OFF2(10)`을 발생시켰으므로 사용하지 않습니다. 접촉
종료까지 기다리던 소프트웨어 지연만 제거하고 이동 동특성은 검증값을 유지합니다.

```bash
ros2 launch mitt_hit_bringup punch_rebound.launch.py
```

## 세션 서비스

영점 보정 뒤 `READY`를 확인하고 연속 세션을 시작합니다. `target_hit_count: 0`은
목표 횟수 없이 명시적인 Stop까지 계속 펀치를 받는다는 뜻입니다.

```bash
ros2 service call /mitt/start_test boxing_interfaces/srv/StartHitTest \
  "{target_hit_count: 0, auto_recover: false}"
```

명시적으로 세션을 종료하면 활성화돼 있던 compliance도 함께 release됩니다.

```bash
ros2 service call /mitt/stop_test boxing_interfaces/srv/StopHitTest "{}"
```

## 주요 파일

```text
src/mitt_hit_system/mitt_hit_system/hit_analyzer_node.py  ROS 운영 노드
src/mitt_hit_system/mitt_hit_system/hit_test_manager.py  세션/순응 수명주기
src/mitt_hit_system/mitt_hit_system/compliance_controller.py  Doosan 서비스 어댑터
src/mitt_hit_system/mitt_hit_system/return_to_reference.py  복귀 판정
src/mitt_hit_system/mitt_hit_system/rt_force_diagnostic_node.py  RT 수집
src/mitt_hit_bringup/config/hit_analyzer.params.yaml  운영 파라미터
src/mitt_hit_bringup/config/punch_rebound.params.yaml  연속 펀치·복귀 프로파일
src/mitt_hit_bringup/launch/punch_rebound.launch.py  연속 펀치 분석 실행
src/mitt_hit_system/config/robot.yaml  확인된 로봇 장착 정보
data/compliance_return_probe/  삭제하지 않은 과거 실측 기록
data/hit_records/  타격·복귀 기록
```

완료된 일회성 TCP 안정성 진단과 저하중 커미셔닝 실행 코드·설정은 운영 경로에서
제거했습니다. 기존 `data/` 실측 기록은 삭제하지 않았습니다.

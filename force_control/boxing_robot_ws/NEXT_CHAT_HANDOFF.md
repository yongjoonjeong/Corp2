# M0609 미트 로봇 인수인계

갱신일: 2026-08-06 (Asia/Seoul)

## 결정된 운영 정책

순응제어는 타격마다 토글하지 않는다. `StartHitTest`가 성공할 때 한 번 활성화하고,
동일 세션의 타격·복귀·다음 타격 동안 계속 유지한다. 정상 release는 다음 세션 종료
경로에서만 수행한다.

- 목표 타격 횟수 완료
- `/mitt/stop_test`
- 활성 세션 중 노드 shutdown

RT timeout, robot state 이상, TCP 병진·회전 변위 초과, 복귀 timeout은 안전
종료이므로 즉시 release한다. release 실패 시 Start를 잠근다.

`StartHitTest` 직후에는 바로 타격을 받지 않는다. 순응 활성화 뒤
`STABILIZING_COMPLIANCE`에서 정지 상태를 확인하고 3초 wrench 기준을 새로 측정한 뒤,
그때의 TCP를 세션 복귀 기준으로 재설정한다. 이후에만 `WAITING_FOR_HIT`로 전환한다.
새 기준은 타격·복귀 중 갱신하지 않는다.

타격과 복귀 힘은 새 잔류 wrench 기준과의 편차로 판정한다. 동시에 편차와 무관한
원시 총힘·총토크 상한, TCP 병진·회전 변위, RT timeout, robot state를 계속
감시한다. 회전 변위는 Doosan ZYZ 성분을 직접 빼지 않고 두 회전행렬 사이의 최단
물리 회전각으로 계산한다. 안정화가 전체 timeout 안에 끝나지 않아도 release하고
ERROR로 종료한다.

이 정책은 `hit_test_manager.py`에 구현돼 있으며, 타격 사이와 복귀 완료 뒤에는
release 호출이 0회라는 회귀시험을 추가했다.

기존 main 종료 순서는 executor shutdown 뒤 release를 시도할 수 있었다. 현재는 ROS
signal handler를 애플리케이션이 관리하고 executor thread를 유지한 상태에서
`shutdown_session()`으로 release 응답을 받은 뒤 executor와 노드를 종료한다.

## 정리한 코드

운영과 별도로 compliance를 켰다 자동 해제하던 다음 개발용 실행기를 제거했다.

- `compliance_commissioning_node.py`
- `compliance_return_probe_node.py`
- 두 실행기의 테스트
- commissioning/return-probe 전용 YAML 4개
- 사용되지 않던 중복 `config/compliance.yaml`
- setup.py console entry point 2개

실제 운영에 필요한 `compliance_controller.py`, `compliance_preflight_node.py`,
`hit_analyzer_node.py`, `hit_test_manager.py`, `return_to_reference.py`, RT 진단과 관련
테스트는 유지했다. 실측 JSON/CSV는 분석 증거이므로 삭제하지 않았다.

## 장착 기준

- TCP `punching`: `[0, 0, 55, 0, 0, 0] mm/deg`
- Tool `punching_weight`: `0.480 kg`
- TP 자동 측정 CoG: `[18.740, -70.600, -123.310] mm`
- Tool Shape `punching_shape`: `190 x 150 x 50 mm`
- 누르는 법선 방향: Tool `-Z`

TP 자동 측정값은 코드에서 변경하지 않는다.

## 현재 실제 로봇 상태와 blocker

읽기 전용 조회에서 현재 Tool은 `punching_weight`, robot state는
`STATE_STANDBY(1)`였다. 그러나 무접촉 Tool-force는 약 `6.5 N`, `1.29 Nm`였다.

최근 전환 기록:

- `20260805_214010_617281_return_probe.json`: PASS, 3초 전체 수집,
  변위 `0.018 mm`, 총힘 `0.458 N`, 총토크 `0.165 Nm`
- `20260805_214151_436866_return_probe.json`: FAIL, 0.316초에 변위
  `0.515 mm`, 속도 `6.32 mm/s`, 총힘 `3.486 N`, 총토크 `1.047 Nm`, release 성공
- `20260805_214541_227831_return_probe.json`: 영점 moment stddev 초과로 compliance
  서비스 호출 전 차단
- `2026-08-06 08:54`: 기본 강성에서 TCP 변위 `0.504 > 0.500 mm`, release 성공
- `2026-08-06 08:56`: 기본 강성에서 TCP 변위 `1.002 > 1.000 mm`, release 성공.
  안정화 기준이 계속 깨져 post-activation baseline을 만들지 못함

일정한 잔류 외력은 활성화 후 새 세션 기준으로 흡수하고 평균·표준편차·3-sigma 범위를
세션 JSON에 기록하도록 구현했다. 그러나 Doosan 내부 순응 전환에는 이 편차가 전달되지
않으므로 전환 재현성 문제까지 없어진 것은 아니다. `hit_analyzer.params.yaml`의
compliance와 모든 미검증 상한은 계속 fail-closed다. 무접촉 운영 경로 시험 전에는
실타격하지 않는다. Tool/TCP/CoG setter나 safety-limit setter는 호출하지 않는다.

## 운영 실행 경로

```text
rt_force_diagnostic -> /mitt/rt_sample -> hit_analyzer
```

```text
CALIBRATING -> READY -> ENABLING_COMPLIANCE -> STABILIZING_COMPLIANCE
  -> post-activation TCP/wrench baseline capture -> WAITING_FOR_HIT
  -> ANALYZING -> RETURNING_TO_REFERENCE -> WAITING_FOR_HIT
  -> target complete/Stop/shutdown -> release -> TEST_COMPLETE/READY
```

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

현재 기대 결과는 `115 tests, 0 errors, 0 failures, 0 skipped`다.

## 다음 개발 단계

1. `punch_rebound.launch.py`로 연속 세션을 시작한다.
2. `target_hit_count: 0`으로 여러 펀치를 받고 매번 자동 복귀를 확인한다.
3. `/mitt/stop_test`에서만 release 1회가 발생하는지 확인한다.
4. 저장된 최대 변위·회전·힘·토크로 펀치 프로파일을 조정한다.
5. 이후 비전 팀 좌표를 받아 미트 목표 위치 이동 경로를 연결한다.

### 명령형 반발 추가

`rebound_motion.py`를 추가했다. `punch_rebound` 프로파일에서는 compliance를 세션 내내
유지하면서 첫 `10 N` 접촉 샘플에서 비동기 절대 BASE `MoveLine`으로 Tool -Z 방향
`50 mm` 후퇴를 즉시 시작한 뒤 기준 TCP로 복귀한다. 후퇴는 검증된
`25 mm/s, 50 mm/s²`, 복귀는 `20 mm/s, 40 mm/s²`이며 애플리케이션 TCP 변위
watchdog은 동작 여유를 포함해 `60 mm`다. 복귀 중 재접촉은
`MoveStop(DR_SSTO=2)`로 현재 이동을 소프트 정지하고 연속 세션의 새 타격으로 다시
연다. TP 인증 안전설정은 변경하지 않는다.

다음 실제 로봇 단계는 읽기 전용 preflight와 RT 진단 뒤,
`compliance_no_contact_session.params.yaml`을 명시한 운영 노드에서
Start→자동 안정화/기준 저장→Stop만 시험하는 것이다. 이 override는 무접촉 진단
전용이며 실타격에는 사용하지 않는다. 최신 override 강성은 Tool 기준
`[15000, 15000, 5000, 500, 500, 500]`이다.

완료된 `compliance_low_force_return` 및 TCP 안정성 진단 코드·설정·테스트는 운영
경로에서 제거했다. 활성화 과정의 최대 TCP 병진·회전 변위와 절대 총힘·총토크는
세션 JSON에 계속 저장한다.

2026-08-06 09:17 저하중 1회 실측은 성공했다. 활성화 최대 변위 `0.621 mm`,
peak normal `11.27 N`, 접촉 `103.9 ms`, 복귀 `756 ms`, 복귀 최대 변위
`0.0756 mm`였고 마지막에 release 1회 후 `TEST_COMPLETE`가 됐다.

2026-08-06 09:45 저하중 3회 연속 실측도 성공했다. 활성화 최대 변위 `0.551 mm`,
세 타격 peak normal은 `6.47 / 7.78 / 9.23 N`, 복귀 시간은
`720 / 708 / 620 ms`, 복귀 최대 변위는 `0.0379 / 0.0437 / 0.0449 mm`였다.
타격 사이 release는 없었고 세 번째 복귀 뒤 release 1회와 `TEST_COMPLETE`를
확인했다. 기록은 `data/hit_records/20260806_094515_905578_session.json`이다.

이번 코드부터 정상 종료·fault·수동 Stop·shutdown 직전에 세션 전체 최대 병진 변위,
회전 변위, 절대 총힘, 절대 총토크와 종료 원인을 `compliance_summary`에 저장한다.

2026-08-06 10:03 새 회전 watchdog을 포함한 저하중 3회 재시험도 성공했다.
세션 최대 병진 `0.602 mm`, 회전 `0.0246 deg`, 총힘 `11.15 N`, 총토크
`1.17 Nm`였으며 3회 모두 복귀 후 마지막에만 release됐다.

이후 최종 목표에 맞춰 `punch_rebound.params.yaml`과
`punch_rebound.launch.py`를 추가했다. 연속 모드에서는 `target_hit_count=0`을 사용하며
수동 Stop까지 계속 동작한다. 첫 실제 연속 세션에서 `27~34 N`, `156~304 ms` 접촉에도
후퇴가 `0.23~0.54 mm`뿐이어서 Tool Z 강성을 `500 N/m`로 낮췄다. 세션 watchdog은
`60 mm / 2 deg / 95 N / 8 Nm`다. 단, 활성화 전환 중에는 별도 한계
`5 mm / 0.30 deg / 10 N / 3 Nm`가 유지되고 안정화 기준 재캡처 뒤에만 펀치 한계가
적용된다. TP Tool/TCP/CoG와 controller safety setting은 코드에서 변경하지 않는다.

2026-08-06 티치펜던트 확인 결과 TCP 힘 제한은 일반 모드 `96 N`, 감소 모드
`48 N`, 일반 모드 충돌 민감도는 `50%`이며 충돌 감지와 TCP 힘 제한 위반의 정지
모드는 `SS2`다. `punch_rebound`는 Start 직전 읽기 전용
`GetRobotSpeedMode`를 호출해 `NORMAL(0)`이 아니면 활성화를 거절한다. 또한 정상 RT
publisher가 유지된 상태에서 발생한 단일 ROS/DDS 지연 오판을 반영해 이 프로파일의
RT sample timeout을 `100 ms`에서 `250 ms`로 조정했다. 안전 설정 자체는 변경하지
않았다.

2026-08-06 13:13 지속 접촉 시험에서는 1초 접촉, peak normal `47.8 N`, 최대 변위
`6.32 mm`로 접촉 시간이 길면 실제 후퇴가 증가함을 확인했다. 그러나 힘 제거 뒤
`3.76 mm` 잔류 변위에서 멈춰 8초 복귀 timeout이 발생했다. 따라서
`punch_rebound`는 compliance를 끄지 않은 채 대기/복귀 Kz=`5000 N/m`, 접촉 중
Kz=`500 N/m`로 `set_stiffnessx`를 전환하는 적응형 강성 모드로 변경했다. 접촉 시작
전환은 `0.10 s`, 복귀 전환은 `0.50 s`이며 Stop/fault/shutdown에서만 compliance를
release한다.

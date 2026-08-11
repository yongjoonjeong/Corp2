# KO ALL-IN-ONE 실행 가이드

## 1. 최초 설치

GPU가 없는 PC도 그대로 실행할 수 있습니다.

```bash
chmod +x setup.sh run_integrated.sh
./setup.sh
```

번들된 힘 감지 ROS 2 workspace까지 빌드하려면 ROS 2 Humble/colcon 환경에서:

```bash
./setup.sh --build-force
```

## 2. 일반 사용자 / 시연

```bash
./run_integrated.sh --user-mode
```

- 상세 BASE 좌표/속도/진단 정보 숨김
- 시스템 설정 메뉴 숨김
- 비전은 `정상 인식 / 인식 불안정 / 인식 불가`로 요약
- 리치 측정은 30연속 유효 프레임마다 자동으로 양팔 → 오른팔 → 왼팔 전환
- Wake Word는 현재 테스트 정책대로 호출어 1회당 명령 1개

## 3. 관리자 / 개발 테스트

```bash
./run_integrated.sh --admin-mode
```

- LEFT / FRONT / RIGHT 3카메라 관리자 프리뷰
- 관절선, 미트 추적, Guard, Impact state, BASE P/V 등 상세 진단
- 시스템 설정 표시

## 4. 힘 감지 모니터만 연결

```bash
./run_integrated.sh --admin-mode --force-monitor
```

힘 RT 수집/진단만 시작합니다. UI 훈련 시작과 `/mitt/start_test`를 연결하지 않습니다.

## 5. 실제 힘 제어까지 연결

```bash
./run_integrated.sh --admin-mode --force-control
```

이 모드는 번들 `boxing_robot_ws`를 먼저 빌드해야 합니다. 훈련 시작/종료와 `/mitt/start_test`, `/mitt/stop_test`를 연결합니다.

`punch_rebound`는 실제 M0609 모션에 영향을 줄 수 있으므로 실물 검증 전에는 기본 OFF 상태를 유지합니다. 로봇 자세/작업공간/속도/비상정지 상태를 현장에서 확인한 뒤 사용합니다.

## 6. 콤비네이션

음성 예:

```text
케이오 → 콤비네이션 1 시작해줘
케이오 → 원투 훅 1분 훈련
```

현재 정의:

1. 원투: 잽 → 스트레이트
2. 잽잽 스트레이트: 잽 → 잽 → 스트레이트
3. 원투 훅: 잽 → 스트레이트 → 훅
4. 원투 원투: 잽 → 스트레이트 → 잽 → 스트레이트
5. 원투 어퍼: 잽 → 스트레이트 → 어퍼

UI/STT/구조화된 ROS 명령 전달까지 통합돼 있습니다. 실제 각 펀치별 미트 타점 좌표/자세 시퀀스는 안전한 실물 좌표가 아직 제공되지 않아 임의 구현하지 않았습니다.

## 7. 결과 보고서

훈련 종료 시 가능한 데이터만 사용해 다음을 구성합니다.

- 이전 동일 훈련/이전 개선 목표와의 발전 비교
- BEST PUNCH
- CHECK POINT
- 현재 강점
- 우선 개선 포인트
- 힘 데이터가 실제 수신된 경우 힘/방향 분석
- 다음 추천 훈련

`결과 읽어줘` 음성 명령으로 현재 보고서를 TTS로 읽을 수 있습니다.

# KO Final Integration Baseline · 2026-08-11

## 기준
- 최종 몸체/UI/비전/Wakeword/STT/DB/보고서: `KO(5)` 유지
- 실제 로봇/Force에서 standalone의 검증 가치가 높은 최신 기능만 함수·파일 단위로 병합
- 기존 KO의 콤비네이션, Pause/Resume, UI API, 실행 구조는 삭제하지 않음
- 현재 실물 훈련 범위는 **잽 / 스트레이트**이며 훅·어퍼컷은 이후 단계

## 확정 로봇 자세 / 위빙
- HOME: `[0, 0, 90, 0, 90, 0]`
- 위빙 준비: `[-180, 0, 90, 90, 90, 0]`
- 펀칭 준비: `[-90, 60, 30, -90, -90, 0]`
- 위빙 평면: robot BASE `XZ`
- 범위: `X=-85~+85 mm`, `Y=0`, `Z=0~-68 mm`
- 위빙 준비 자세와 펀칭 준비 자세는 서로 다른 자세로 유지

## 최종 플로우
1. 터미널에서 최종 런처 실행
2. 로봇이 위빙 준비 자세로 이동 후 XZ U자 위빙 반복
3. 사용자가 UI에서 훈련 진행 → 카메라 정렬 수행(이 동안 위빙 유지)
4. 카메라 정렬 완료 → `training_start` handoff → 위빙 Soft Stop
5. MittPositioner가 펀칭 준비/사용자 기준 미트 위치로 이동
6. 1차 Calibration: 미트를 미트면 법선인 Tool `+Z`로 천천히 전진시키며 실제 접촉 기반 리치 추가 보정
7. 2차 Calibration: 5회 Force 타격으로 미트 중심 위치 후보정
8. 2차 보정 이동 뒤 Wrench 영점 재조정 동안 UI/TTS: `영점 조정 중입니다. 잠시 기다려주세요.`
9. 영점 조정 완료 후 UI/TTS: `다시 펀치하세요.`
10. 5회 보정 완료 후 `TRAINING_READY` → 잽/스트레이트 실제 훈련 시작
11. 훈련 중 비전 예상 타점을 안전 범위 내에서 미트 이동에 사용하고 Force 결과 저장
12. 훈련 종료 → Force 세션 종료 → 위빙 준비 자세 복귀 → 위빙 재시작
13. UI는 보고서를 생성해 DB에 저장하고, 다음 보고서에서 이전 훈련 기록을 비교/참고

## 이번 통합에서 병합한 standalone 기능
- 최신 RT Wrench fusion (`external_tcp_force + external_joint_torque + Jacobian`)
- 접촉식 Reach Calibration helper 및 SessionBridge 연결
- 주먹 예상 타점/미트 타점 안정화에 필요한 로봇-side helper
- 의도된 미트 이동을 Compliance watchdog과 충돌시키지 않기 위한 motion guard / post-move re-zero

## UI / 보고서 수정
- 2차 Calibration Wrench zeroing 상태를 UI에 명시
- Zeroing 시작 TTS: `영점 조정 중입니다. 잠시 기다려주세요.`
- Zeroing 완료 TTS: `다시 펀치하세요.`
- 코칭 보고서 생성 프롬프트에 `England Boxing Level 1 Coaching Handbook` 기준을 추가
- 단, 실제 측정된 KO 데이터가 최우선이며 측정하지 않은 자세/동작은 추정하지 않도록 유지

## 검증 경계
정적/ROS 비의존 테스트는 최종 ZIP 생성 전 수행한다. 실제 M0609의 미트면 Tool +Z 접근 방향, 접촉 임계값, 5회 Force 후보정 부호, motion-guard 타이밍, 충돌 여유와 Compliance 체감은 대상 실물에서 저속으로 최종 확인해야 한다.

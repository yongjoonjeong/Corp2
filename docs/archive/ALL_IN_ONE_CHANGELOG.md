# KO ALL-IN-ONE V1 변경사항

기준: `KO_TEAM_R2_PHASE2_FEEDBACK_BEST_WORST`

## 통합 완료

- USER / ADMIN 런타임 모드 유지
- ADMIN 3카메라(LEFT/FRONT/RIGHT) 진단 프리뷰 유지
- USER 종합 비전 상태 유지
- 시스템 설정 ADMIN 전용 유지
- 모든 주요 화면의 `호출어 1회 → 명령 1개` 음성 도움말 유지
- 리치 측정 30연속 유효 프레임 자동화
  - 양팔 → 오른팔 → 왼팔 자동 전환
  - 각 단계 완료 TTS
  - 반대팔 가드 조건으로 단계 오검출 감소
  - 카메라 실패 시 재연결 버튼 복원
- 콤비네이션 1~5 추가
  - STT 별칭/번호 파싱
  - UI 훈련 선택/현재 단계 표시
  - 주손 기반 lead/rear 손 결정
  - 구조화된 training payload를 ROS bridge로 전달
- 힘 제어 workspace 번들
  - `force_control/boxing_robot_ws/src`
  - `/mitt/hit_result` UI/DB 인터페이스
  - `--force-monitor`, `--force-control`
  - 실제 제어는 기본 OFF
- 이전 피드백 추적 및 deterministic progress 비교
- 펀치 이벤트 DB + force 결과 DB
- BEST PUNCH / CHECK POINT 선정 및 3카메라 증거 이미지 연결
- OpenAI 최종 코칭 입력 확장
  - 이전 발전 비교
  - BEST/CHECK 이미지
  - 현재 측정 수치
  - 실제 존재하는 힘 데이터
- 최종 보고서 UI 확장
  - 강점
  - 개선 포인트
  - 힘 분석(데이터 있을 때만)
  - 다음 추천 훈련
  - `결과 읽어줘` TTS
- GPU 없는 PC에서 기존 CPU 사용 가능한 PyTorch 재사용

## 의도적으로 미완성으로 남긴 부분

- 콤비네이션 각 펀치별 실제 M0609 미트 좌표/자세 시퀀스
  - 실제 좌표가 제공되지 않아 임의 생성하지 않음
- 힘 제어/반발 동작 실물 안전 검증
  - 기존 fail-closed 설정 유지
- `power_score`, `safety_stop`을 코칭 근거로 사용하지 않음
  - 현재 제공된 힘 코드에서 완성/검증된 값으로 볼 수 없기 때문

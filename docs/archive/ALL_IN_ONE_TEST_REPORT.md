# KO ALL-IN-ONE V1 테스트 보고서

테스트 일자: 2026-08-07

## 통과

- JavaScript 문법: `node --check ui/static/js/app.js`
- Python 문법: UI server / reporting / vision coach / robot bridge `py_compile`
- Shell 문법: setup / integrated launcher / bridge / force helper `bash -n`
- ROS 비의존 vision pytest: **16 passed**
- UI/API unittest: **39 passed**

검증 항목에는 다음이 포함됩니다.

- USER/ADMIN 모드 계약
- ADMIN 3카메라 프리뷰 계약
- 자동 리치 30프레임 + 단계 TTS 문구
- 콤비네이션 1~5 UI/STT/명령 구조
- 피드백 progress 비교
- BEST/CHECK 선정
- force accuracy 결합 시 미완성 power_score를 사용하지 않는 것
- punch event / force result 저장 및 세션 연결
- 결과 보고서 확장 UI 계약
- force monitor/control 런처 옵션

## 이 환경에서 실행하지 못한 것

전체 `./run_tests.sh`는 현재 컨테이너에 ROS 2 Python 패키지 `rclpy`가 없어 `sandbag_vision.node`를 직접 import하는 2개 테스트가 collection 단계에서 실패합니다. 따라서 해당 2개는 ROS 2 Humble 환경에서 다시 실행해야 합니다.

또한 다음은 실물 장비에서 검증되지 않았습니다.

- M0609 실제 자동 위빙
- 3대 실카메라 동시 동작/FPS
- `/mitt/hit_result` 실제 타격 동기화
- `punch_rebound` 실제 로봇 힘 반발 제어
- 콤비네이션 실제 미트 이동

따라서 본 ZIP은 소프트웨어 통합/정적·비ROS 테스트까지 완료된 상태이며 실물 로봇 검증 완료본으로 표현하지 않습니다.

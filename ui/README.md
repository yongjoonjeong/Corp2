# KO 웹 UI — 현재 realtime v3 비전 연결본

> **최종 통합 실행은 프로젝트 루트의 `./run_final.sh`를 사용합니다.** 최종 실행에서는 Wakeword와 실제 `boxing_robot_ws` Force/SessionBridge가 기본 활성화되며, `./run_final.sh --admin-mode`로 ADMIN 진단 화면을 사용할 수 있습니다. 아래의 `run_integrated.sh` 설명은 UI/비전 부분 실행 및 개발용 경로입니다.

이 폴더는 ZIP에서 UI와 사용자/훈련 기록 기능만 가져온 것입니다. ZIP에 포함된
카메라·YOLO·MediaPipe 펀치 런타임은 복사하거나 실행하지 않습니다.

`vision_bridge.py`는 현재 프로젝트가 발행하는 다음 ROS 2 토픽만 읽어 로컬 UI로
전달합니다.

- `/sandbag/vision/status`
- `/sandbag/fist_state`
- `/sandbag/impact_event`
- `/sandbag/impact_feedback_image/compressed`
- `/sandbag/vision/preview/compressed`
- `/sandbag/vision/front/compressed`

프로젝트 루트에서 `./run_integrated.sh`를 실행하고 브라우저로
`http://127.0.0.1:5000`을 여세요. 기본 모드는 음성 기능을 끈 상태라 OpenAI API
키와 별도 UI 설치가 필요 없습니다. 이 경우 결과 화면의 `KO COACH`에는 기존의
로컬 규칙 피드백이 표시됩니다.

## 리치 측정 카메라 공유

통합 실행 중에는 리치 측정 화면이 카메라 장치를 다시 열지 않습니다. 현재 비전
노드가 점유한 전면 RealSense 컬러 프레임을
`/sandbag/vision/front/compressed`로 10 Hz 발행하고, UI 브리지가
`/api/vision/front.jpg`로 전달합니다. 브라우저 MediaPipe는 이 공유 프레임을
분석하므로 `Could not start video source` 카메라 점유 충돌이 발생하지 않습니다.
ROS 비전이 연결되지 않은 UI 단독 실행에서만 브라우저 카메라를 대체 입력으로
사용합니다.

## 저장 사진을 이용한 OpenAI KO COACH

OpenAI 사진 코칭을 사용할 때만 API 키를 한 번 등록합니다.

```bash
python3 ui/configure_api_key.py
./run_integrated.sh
```

키는 브라우저로 전달되지 않고 `ui/.env`에 권한 `600`으로 저장됩니다. 훈련이 끝나면
훈련 시작 이후 저장된 `impact_triptych.jpg`를 펀치 이벤트와 연결한 뒤 **BEST PUNCH 1장 + CHECK POINT 1장**을 우선 선택해
OpenAI Responses API로 전송하고, 이전 동일 훈련의 측정값/피드백과 함께 결과 화면의 `KO COACH` 보고서 및 DB 세션 피드백을 갱신합니다. API 요청에는 `store: false`를 사용합니다. 키가 없거나 네트워크
분석이 실패하거나 이번 세션에 새 사진이 없으면 훈련 종료와 결과 저장을 막지 않고
기존 로컬 피드백으로 자동 복귀합니다.

정지사진에서 직접 보이는 팔 신전, 반대손 가드, 어깨·상체 정렬만 분석 대상으로
삼습니다. 사진만으로 확인할 수 없는 타격 세기, 실제 접촉, 전체 궤적, 가드 복귀
속도는 평가하지 않습니다.

필요하면 `ui/.env`에 다음 값을 추가할 수 있습니다.

```dotenv
OPENAI_VISION_COACH_MODEL=gpt-5
# Phase 2 기본 증거는 BEST + CHECK POINT 두 장
OPENAI_VISION_COACH_DETAIL=auto
```

Phase 2에서는 펀치별 이벤트 점수로 BEST/CHECK POINT를 로컬에서 먼저 결정합니다. OpenAI는 이 순위를 다시 판단하지 않고, 제공된 수치와 정지영상에 근거해 설명만 생성합니다.

## 실시간 미트 타깃과 판정 상태

훈련 화면의 원형 펄스 타깃은 고정 좌표가 아니라 현재 비전의
`mitt_tracker.roi_normalized` 중심에 배치됩니다. 초록색 `IMPACT ZONE`과 동일한
빨간 미트 추적 결과를 사용하며, 미트를 잃으면 원형 타깃도 숨깁니다.

확정 타격은 `READY → ACTIVE → IMPACT → COOLDOWN`, 미확정 동작은
`READY → ACTIVE → READY` 순으로 표시됩니다. `IMPACT`만 빨간색이며 확정 이벤트가
발생한 짧은 구간에만 보입니다.

## ZIP 위빙 로봇 복원

ZIP의 비전 런타임은 사용하지 않고 `robot_control/`의 두산 M0609 위빙 노드와
UI–ROS 브리지만 복원했습니다. `./run_integrated.sh`는
`/dsr01/motion/move_stop` 서비스가 보일 때만 HOME → 위빙 준비 → U자 위빙을
자동 시작합니다. roboton 서비스가 없으면 로봇 모션만 건너뛰고 UI와 현재 v3
비전은 계속 실행합니다. 통합 실행 종료 시에는 로봇 프로세스를 종료하기 전에
Soft Stop 명령을 먼저 전달합니다.

자동 위빙을 끄고 UI/비전만 실행하려면 다음과 같이 시작합니다.

```bash
KO_ENABLE_ROBOT_WEAVING=0 ./run_integrated.sh
```

로봇 노드는 실행하되 시작 직후 자동 위빙만 막으려면 다음을 사용합니다.

```bash
KO_ROBOT_AUTO_START_WEAVING=0 ./run_integrated.sh
```

Wake Word/음성 기능까지 사용할 때만 `./ui/setup_ui.sh`를 한 번 실행한 뒤
`KO_ENABLE_WAKEWORD=1 ./run_integrated.sh`로 시작합니다.

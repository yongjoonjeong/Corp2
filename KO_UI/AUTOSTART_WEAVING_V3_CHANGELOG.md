# V3 자동 위빙 통합 변경사항

- 기준 패키지: `KO_UI_SPORTY_VOICE_SESSION_V2`
- `robot_control/robot_weaving_node.py`를 최신 자동 시작 위빙 노드로 교체
- HOME을 `[-180, 0, 90, -180, -90, 0]`으로 변경
- 위빙 준비 자세를 `[-180, 0, 90, -270, 90, 0]`으로 변경
- 별도 위빙 접근 자세 제거
- 노드 시작 시 HOME → 위빙 준비 → U자 위빙 자동 시작
- Wake Word 수신 시 위빙 Soft Stop → 위빙 준비 자세 복귀
- UI-ROS 브리지의 Wake Word 상태 문구와 명령 의미를 새 흐름에 맞게 변경
- `prepare`/`start` 명령은 수동 위빙 재시작 명령으로 유지
- 기존 UI, 사용자 등록, SQLite, Whisper STT, 연속 음성 세션, 선택형 비전 기능 보존

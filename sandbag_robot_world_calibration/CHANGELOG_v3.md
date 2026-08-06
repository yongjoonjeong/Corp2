# v3 변경 사항

- `/dev/videoN` 고정 방식 제거
- RealSense는 `pyrealsense2`로만 열어 RGB·Depth를 명시적으로 분리
- RealSense IR/metadata 노드가 left/right C270로 선택되는 문제 차단
- `C270` 제품명과 Logitech VID/PID로 C270 RGB 장치만 자동 탐색
- `/dev/v4l/by-path/*-video-index0` 안정 경로 사용
- `python3 00_check_cameras.py --assign-c270` 최초 좌·우 배정 기능 추가
- 배정 결과를 `config/camera_roles.yaml`에 저장하고 모든 내부·외부 캘리 코드에서 공통 사용
- `--list-c270` 장치 진단 기능 추가

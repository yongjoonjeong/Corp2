# v2 변경사항

- `front`를 RealSense RGB-D 장치로 명확히 분리했습니다.
- 카메라 확인 시 RealSense RGB와 Depth를 동기화해 함께 표시합니다.
- Depth 프레임을 RGB 좌표계에 정렬합니다.
- 전체 확인 화면을 사용자 기준 물리 배치로 구성했습니다.
  - 상단: 좌측 C270 | 정면 RealSense RGB | 우측 C270
  - 하단 중앙: 정면 RealSense Depth
- RealSense는 `/dev/videoN` 순서가 아니라 `pyrealsense2`와 시리얼로 선택합니다.
- 내부·외부 캘리브레이션에서는 기존처럼 RealSense RGB 영상만 사용합니다.

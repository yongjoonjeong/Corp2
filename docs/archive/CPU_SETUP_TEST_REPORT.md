# CPU/GPU 자동 설치 보완 테스트

## 수정 범위

애플리케이션/비전/로봇 동작 코드는 변경하지 않았습니다.
다음 설치 관련 파일만 변경했습니다.

- `setup.sh`
- `requirements.txt`
- `README.md`
- `DEPLOYMENT.md`
- `PHASE1_CHANGELOG.md`
- `SETUP_CPU_GPU.md` 추가

## CPU PC 동작 정책

`./setup.sh` 기본 실행 시 NVIDIA GPU가 없으면 CPU 모드를 선택합니다.
기존 `torch`와 `torchvision`이 정상 import되면 버전 문자열이 `+cuXXX`여도
CPU 실행용으로 그대로 사용하므로 CUDA 12.8 wheel을 다시 다운로드하지 않습니다.

현재 사용자 환경 예시:

- torch `2.6.0+cu124`
- torchvision `0.21.0+cu124`
- `torch.cuda.is_available() == False`

이 경우 기존 패키지를 재사용하도록 설계했습니다.

## 검증

- `setup.sh --help`: 정상
- `bash -n setup.sh`: 통과
- `bash -n run_integrated.sh`: 통과
- `bash -n run_vision.sh`: 통과
- `bash -n run_ui_bridge.sh`: 통과
- `bash -n run_tests.sh`: 통과
- UI/API 테스트: **28 passed + 18 subtests passed**
- ROS 비의존 비전 코어 테스트: **13 passed**

컨테이너 환경에는 ROS 2 `rclpy`가 없어 ROS 노드를 import하는 테스트 2개는 실행하지 못했습니다.
실제 Ubuntu/ROS 2/카메라 환경의 실시간 FPS와 장치 인식은 사용자 PC에서 확인해야 합니다.

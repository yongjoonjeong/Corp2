# Sandbag Vision Realtime v3 배포 및 실행

이 배포본에는 현재 v3 비전, GPU YOLO 모델, MediaPipe 모델, 삼각측량
캘리브레이션 도구, KO 웹 UI, 전면 RealSense 공유 리치 측정, 두산 M0609
위빙 제어 코드가 포함되어 있습니다.

현재 소스에서 경량 배포 압축본을 만들 때는 다음 명령을 사용합니다.

```bash
./tools/create_deploy_bundle.sh
```

생성물은 `deploy/KO_deploy_YYYYMMDD.tar.gz`이며, 압축 해제 후
`./setup.sh --build-force`로 가상환경과 ROS 빌드를 재생성합니다.

다음 항목은 의도적으로 포함하지 않습니다.

- 타격 저장 사진과 실행 결과(`output/`)
- 사용자 훈련 기록 DB(`ui/instance/`)
- OpenAI API 키(`ui/.env`)
- Python 가상환경(`.venv/`, `ui/.venv/`)
- 로그, 캐시, `.vscode`, Git 메타데이터

## 1. 필수 시스템

- Ubuntu 22.04, Python 3.10
- ROS 2 Humble: `/opt/ros/humble/setup.bash`
- RealSense 및 좌·우 V4L2 카메라 접근 권한
- NVIDIA 드라이버가 설치된 RTX 5080 환경
- 로봇 위빙 사용 시 Doosan `roboton`, `dsr_msgs2`, `DR_init`,
  `DSR_ROBOT2`가 설치된 ROS 2 workspace

기본 패키지가 없다면 먼저 설치합니다.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip v4l-utils \
  libgl1 libglib2.0-0 portaudio19-dev
```

ROS 2 Humble과 Doosan 패키지는 사용하는 로봇 환경의 설치 절차에 따라 별도로
준비해야 합니다.

## 2. 압축 해제와 Python 환경 구성

```bash
cd ~/Downloads
tar -xzf KO_deploy_YYYYMMDD.tar.gz
cd KO
chmod +x ./*.sh ui/*.sh robot_control/*.sh
./setup.sh --build-force
```

`setup.sh`는 프로젝트 `.venv`를 만들고 CPU/GPU 환경을 자동 판별한 뒤 필요한 PyTorch를 준비하며,
Torchvision, Ultralytics, MediaPipe, LAP, RealSense Python 패키지를
설치합니다. 네트워크 속도에 따라 시간이 걸릴 수 있습니다.

## 3. 카메라와 설정 확인

```bash
source /opt/ros/humble/setup.bash
./.venv/bin/python tools/check_config.py
./.venv/bin/python tools/assign_cameras.py
```

배포본에는 현재 장비의 intrinsic과
`calibration/results/robot_world.yaml`이 포함되어 있습니다. 카메라 위치나
방향을 바꾸거나 다른 장비에 설치하면 `calibration_tools/README.md` 순서로
다시 캘리브레이션해야 합니다.

## 4. 기본 통합 실행

로봇 없이 UI와 현재 v3 비전만 먼저 확인할 때:

```bash
KO_ENABLE_ROBOT_WEAVING=0 ./run_integrated.sh
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:5000
```

비전, UI, 로봇 위빙을 함께 실행할 때는 먼저 별도 터미널에서 `roboton`을
완전히 실행하고 작업 영역이 비어 있는지 확인한 뒤:

```bash
./run_integrated.sh
```

`/dsr01/motion/move_stop` 서비스가 확인되면 HOME → 위빙 준비 자세 → U자
위빙이 자동 시작됩니다. 서비스가 없으면 로봇 모션만 건너뛰고 UI와 비전은
계속 실행됩니다.

자동 위빙을 막고 로봇 노드만 준비하려면:

```bash
KO_ROBOT_AUTO_START_WEAVING=0 ./run_integrated.sh
```

종료는 실행 터미널에서 `Ctrl+C`입니다. 통합 실행기는 로봇 프로세스를
종료하기 전에 Soft Stop 명령을 먼저 전송합니다.

## 5. Wake Word와 OpenAI 기능

음성 기능용 환경을 한 번 설치하고, 새 OpenAI API 키를 등록합니다.

```bash
./ui/setup_ui.sh
python3 ui/configure_api_key.py
KO_ENABLE_WAKEWORD=1 ./run_integrated.sh
```

API 키는 새로 생성되는 `ui/.env`에 권한 600으로 저장되며 브라우저로
전달되지 않습니다. 배포 ZIP에는 API 키가 포함되지 않습니다.

## 6. 개별 실행과 검증

```bash
# 비전만 실행
./run_vision.sh

# 전체 자동 테스트
./run_tests.sh

# 로봇 Python/ROS 환경만 점검
./robot_control/check_robot_env.sh
```

정상 통합 실행 시 터미널에 다음 항목이 표시됩니다.

```text
UI      : http://127.0.0.1:5000
VISION  : 현재 sandbag_vision/node.py (ZIP 비전 미사용)
ROBOT   : HOME → 준비 자세 → U자 위빙 자동 시작
```


## CPU / GPU 설치 모드

GPU가 없는 PC에서는 CUDA PyTorch 대용량 다운로드를 생략합니다. 자세한 내용은 `SETUP_CPU_GPU.md`를 참고하세요.

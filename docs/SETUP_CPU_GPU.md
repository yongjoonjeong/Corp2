# KO CPU / GPU 자동 설치

## 기본 실행

```bash
./setup.sh
```

`setup.sh`가 NVIDIA GPU 존재 여부를 확인합니다.

- NVIDIA GPU 없음 → CPU 모드
- NVIDIA GPU 있음 → CUDA 모드

## CPU PC

GPU가 없는 노트북에서는 기존 `torch` / `torchvision`이 정상 import되면 그대로 재사용합니다.
따라서 CUDA 계열 wheel이 설치되어 있어도 `torch.cuda.is_available() == False`인 CPU PC라면
수백 MB의 CUDA 12.8 PyTorch를 다시 다운로드하지 않습니다.

강제로 CPU 모드를 선택하려면:

```bash
./setup.sh --cpu
```

기존 torch가 없거나 import가 깨져 있을 때만 PyTorch CPU wheel을 설치합니다.

## NVIDIA GPU PC

```bash
./setup.sh --cuda
```

기존 PyTorch가 CUDA를 실제로 사용할 수 있으면 재사용합니다.
그렇지 않으면 CUDA 12.8용 `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`을 설치합니다.

## 강제 재설치

```bash
KO_FORCE_TORCH_REINSTALL=1 ./setup.sh --cpu
```

또는

```bash
KO_FORCE_TORCH_REINSTALL=1 ./setup.sh --cuda
```

## 실행 장치

비전 런타임의 YOLO 장치는 기존 코드의 `device: auto` 정책을 유지합니다.
`torch.cuda.is_available()`이 `True`이면 GPU(`0`), 아니면 `cpu`를 사용합니다.

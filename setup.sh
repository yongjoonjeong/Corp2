#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$project_dir/.venv"

usage() {
  cat <<'TXT'
KO setup

Usage:
  ./setup.sh              # auto: NVIDIA GPU가 있으면 CUDA, 없으면 CPU
  ./setup.sh --cpu        # CPU 모드 강제
  ./setup.sh --cuda       # CUDA 모드 강제
  ./setup.sh --build-force # Python + Wakeword UI + force ROS workspace 준비
  ./setup.sh --no-voice    # Wakeword UI 환경 설치를 건너뜀

Environment overrides:
  KO_TORCH_MODE=auto|cpu|cuda
  KO_FORCE_TORCH_REINSTALL=1   # 기존 torch가 있어도 선택 모드로 재설치
TXT
}

mode="${KO_TORCH_MODE:-auto}"
build_force=0
setup_voice=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) mode="cpu" ;;
    --cuda) mode="cuda" ;;
    --auto) mode="auto" ;;
    --build-force) build_force=1 ;;
    --no-voice) setup_voice=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 알 수 없는 옵션: $1"; usage; exit 2 ;;
  esac
  shift
done

case "$mode" in
  auto|cpu|cuda) ;;
  *) echo "[ERROR] KO_TORCH_MODE는 auto/cpu/cuda 중 하나여야 합니다: $mode"; exit 2 ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3가 없습니다. sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "[KO] .venv 생성 (--system-site-packages)"
  python3 -m venv --system-site-packages "$venv_dir"
fi

py="$venv_dir/bin/python"
if ! "$py" -m pip --version >/dev/null 2>&1; then
  echo "[ERROR] 가상환경에 pip가 없습니다."
  echo "sudo apt install -y python3-venv python3-pip 후 .venv를 지우고 다시 실행하세요."
  exit 1
fi

has_nvidia_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    return 0
  fi
  if [[ -d /proc/driver/nvidia/gpus ]] && compgen -G '/proc/driver/nvidia/gpus/*' >/dev/null 2>&1; then
    return 0
  fi
  if [[ -e /dev/nvidia0 ]]; then
    return 0
  fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Eqi 'NVIDIA.*(VGA|3D|Display)|VGA.*NVIDIA|3D.*NVIDIA'; then
    return 0
  fi
  return 1
}

torch_info() {
  "$py" - <<'PY'
try:
    import torch, torchvision
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
except Exception as exc:
    print(f"torch_check_error={type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
}

torch_import_ok() {
  "$py" - <<'PY' >/dev/null 2>&1
import torch, torchvision
_ = torch.__version__
_ = torchvision.__version__
PY
}

torch_cuda_ok() {
  "$py" - <<'PY' >/dev/null 2>&1
import torch, torchvision
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if [[ "$mode" == "auto" ]]; then
  if has_nvidia_gpu; then
    selected_mode="cuda"
  else
    selected_mode="cpu"
  fi
else
  selected_mode="$mode"
fi

echo "[KO] Torch mode requested: $mode"
echo "[KO] Torch mode selected : $selected_mode"

force_reinstall="${KO_FORCE_TORCH_REINSTALL:-0}"

if [[ "$selected_mode" == "cpu" ]]; then
  if [[ "$force_reinstall" != "1" ]] && torch_import_ok; then
    echo "[KO] 기존 PyTorch를 CPU 실행용으로 재사용합니다."
    torch_info || true
    echo "[KO] CUDA wheel 표기가 있어도 GPU가 없는 PC에서는 CPU 연산으로 사용할 수 있습니다."
  else
    echo "[KO] CPU 전용 PyTorch 설치"
    "$py" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.11.0' 'torchvision==0.26.0'
  fi
else
  if ! has_nvidia_gpu; then
    echo "[ERROR] --cuda가 선택됐지만 NVIDIA GPU를 찾지 못했습니다."
    echo "GPU가 없는 PC에서는 ./setup.sh --cpu 를 사용하세요."
    exit 1
  fi
  if [[ "$force_reinstall" != "1" ]] && torch_cuda_ok; then
    echo "[KO] 기존 CUDA PyTorch를 재사용합니다."
    torch_info || true
  else
    echo "[KO] CUDA 12.8 PyTorch 설치"
    echo "[KO] 대용량 다운로드가 발생할 수 있습니다."
    "$py" -m pip install --upgrade --timeout 1000 --retries 20 \
      --index-url https://download.pytorch.org/whl/cu128 \
      'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
  fi
fi

echo "[KO] pip / build tools 업데이트"
"$py" -m pip install --upgrade pip setuptools wheel

echo "[KO] 프로젝트 Python 의존성 설치"
"$py" -m pip install --timeout 1000 --retries 20 -r "$project_dir/requirements.txt"

echo "[KO] 핵심 패키지 확인"
"$py" - <<'PY'
import torch
import torchvision
import lap
import ultralytics
print(f"[OK] torch={torch.__version__}")
print(f"[OK] torchvision={torchvision.__version__}")
print(f"[OK] CUDA available={torch.cuda.is_available()}")
print("[OK] lap / Ultralytics")
PY

echo "[OK] Python 환경 준비 완료 ($selected_mode mode)"
if [[ "$setup_voice" == "1" ]]; then
  echo "[KO] Wakeword/STT UI 환경 준비"
  if ! "$project_dir/ui/setup_ui.sh"; then
    echo "[ERROR] Wakeword 환경 설치 실패" >&2
    echo "Ubuntu에서 PyAudio 빌드 오류가 나면: sudo apt install -y portaudio19-dev python3-dev" >&2
    exit 1
  fi
fi
if [[ "$build_force" == "1" ]]; then
  if [[ -f /opt/ros/humble/setup.bash ]] && command -v colcon >/dev/null 2>&1; then
    "$project_dir/force_control/build_force_ws.sh"
  else
    echo "[WARN] ROS 2 Humble/colcon이 없어 force workspace 빌드를 건너뜁니다." >&2
  fi
fi
echo "다음: ./run_integrated.sh --user-mode"

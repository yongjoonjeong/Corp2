#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d .venv ]]; then
    python3 -m venv --system-site-packages .venv
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "[ERROR] .venv/bin/python was not created." >&2
    exit 1
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

# Ultralytics installs the non-contrib OpenCV wheel. Reinstall contrib last so
# both YOLO runtime and ChArUco calibration use one cv2 implementation.
.venv/bin/python -m pip install --force-reinstall --no-deps \
    opencv-contrib-python==4.11.0.86

mkdir -p models
pushd models >/dev/null
../.venv/bin/python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

weight = Path("yolo11n-pose.pt")
YOLO(str(weight))
if not weight.is_file():
    raise SystemExit("yolo11n-pose.pt download failed")
print(f"[OK] YOLO11n-Pose: {weight.resolve()}")
PY
popd >/dev/null

.venv/bin/python - <<'PY'
import cv2
import mediapipe
import pyrealsense2
import torch
import ultralytics

assert hasattr(cv2, "aruco")
print(f"[OK] OpenCV {cv2.__version__} / MediaPipe {mediapipe.__version__}")
print(f"[OK] Ultralytics {ultralytics.__version__} / Torch {torch.__version__}")
print(f"[OK] YOLO device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
print("[OK] pyrealsense2")
PY

echo "[DONE] 3D runtime dependencies are ready."
echo "[NEXT] Standalone vision: ./run_ros_3d_mvp.sh"
echo "[NEXT] Full UI integration: cd .. && ./run_integrated.sh"

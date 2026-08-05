#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy PyYAML pytest

if python - <<'PY'
import cv2
assert hasattr(cv2, "aruco")
print(f"[OK] OpenCV {cv2.__version__} with ArUco")
PY
then
    :
else
    python -m pip install "opencv-contrib-python>=4.8"
fi

if python - <<'PY'
import pyrealsense2
print("[OK] pyrealsense2")
PY
then
    :
else
    python -m pip install "pyrealsense2>=2.54" || \
        echo "[WARN] pyrealsense2 installation failed. Install librealsense/Python bindings before using front."
fi

echo "Activate with: source .venv/bin/activate"

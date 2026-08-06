#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$ROOT/ui"
VISION_DIR="$ROOT/vision"
[[ -d "$UI_DIR" ]] || UI_DIR="$ROOT/KO_UI"
[[ -d "$VISION_DIR" ]] || VISION_DIR="$ROOT/sandbag_robot_world_calibration"

required_files=(
  "$ROOT/run_integrated.sh"
  "$UI_DIR/setup.sh"
  "$UI_DIR/run.sh"
  "$UI_DIR/run_ui_only.sh"
  "$UI_DIR/run_vision_optional.sh"
  "$UI_DIR/robot_control/run_bridge.sh"
  "$UI_DIR/robot_control/run_robot_node.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_with_ko_ui.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_ros_mvp.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_ko_ui_bridge.sh"
  "$VISION_DIR/setup_3d.sh"
  "$VISION_DIR/run_ros_3d_mvp.sh"
  "$VISION_DIR/calibration/results/robot_world.yaml"
  "$VISION_DIR/calibration/intrinsics/front.yaml"
  "$VISION_DIR/calibration/intrinsics/left.yaml"
  "$VISION_DIR/calibration/intrinsics/right.yaml"
  "$VISION_DIR/config/camera_roles.yaml"
  "$VISION_DIR/config/board.yaml"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] 통합본 필수 파일이 없습니다: $path" >&2
    exit 1
  fi
done

executable_scripts=(
  "$ROOT/setup_integrated.sh"
  "$ROOT/run_integrated.sh"
  "$UI_DIR/setup.sh"
  "$UI_DIR/run.sh"
  "$UI_DIR/run_ui_only.sh"
  "$UI_DIR/run_vision_optional.sh"
  "$UI_DIR/robot_control/run_bridge.sh"
  "$UI_DIR/robot_control/run_robot_node.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_with_ko_ui.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_ros_mvp.sh"
  "$UI_DIR/optional/vision_processing/body_tracking_mvp/run_ko_ui_bridge.sh"
  "$VISION_DIR/setup_3d.sh"
  "$VISION_DIR/run_ros_3d_mvp.sh"
)
chmod +x "${executable_scripts[@]}"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] ROS 2 Humble을 찾을 수 없습니다: /opt/ros/humble/setup.bash" >&2
  echo "        ROS 2 Humble 설치 후 다시 실행하세요." >&2
  exit 1
fi

echo "[1/2] KO UI 환경을 설치합니다."
bash "$UI_DIR/setup.sh"

echo
echo "[2/2] 3카메라 YOLO11n-Pose 환경을 설치합니다."
bash "$VISION_DIR/setup_3d.sh"

echo
echo "[VERIFY] ROS 2 시스템 Python과 두 비전 프로세스의 import를 확인합니다."
if [[ ! -x /usr/bin/python3 ]]; then
  echo "[ERROR] ROS 2용 시스템 Python을 찾을 수 없습니다: /usr/bin/python3" >&2
  exit 1
fi

# ROS setup scripts may reference unset variables, so temporarily relax nounset.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

/usr/bin/python3 - <<'PY'
import rclpy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

print(f"[OK] KO UI ROS bridge: rclpy={rclpy.__file__}")
print(f"[OK] ROS messages: {String.__name__}, {CompressedImage.__name__}")
PY

"$VISION_DIR/.venv/bin/python" - <<'PY'
import cv2
import numpy
import PIL
import pyrealsense2
import rclpy
import sensor_msgs.msg
import std_msgs.msg
import torch
import ultralytics
import yaml

assert hasattr(cv2, "aruco")
print(f"[OK] 3D vision venv: rclpy={rclpy.__file__}")
print(f"[OK] YOLO={ultralytics.__version__}, torch={torch.__version__}")
PY

echo
echo "============================================================"
echo "통합 설치가 완료되었습니다."
echo
echo "로봇 포함 실행:"
echo "  터미널 1: roboton"
echo "  터미널 2: ./run_integrated.sh"
echo
echo "로봇 없이 UI + 3D 비전 실행:"
echo "  KO_WITHOUT_ROBOT=1 ./run_integrated.sh"
echo "============================================================"

#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
    exit 1
fi

PYTHON_BIN="python3"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

if [[ ! -f "$SCRIPT_DIR/models/yolo11n-pose.pt" ]]; then
    echo "YOLO11n-Pose model is missing. Run: $SCRIPT_DIR/setup_3d.sh" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
set -u

if ! "$PYTHON_BIN" -c \
    "import cv2, numpy, yaml, PIL, rclpy, torch, ultralytics" >/dev/null 2>&1; then
    echo "YOLO 2D dependencies are missing. Run: $SCRIPT_DIR/setup_3d.sh" >&2
    exit 1
fi

exec "$PYTHON_BIN" \
    "$SCRIPT_DIR/webcam_punch_feedback_node.py" \
    --ros-args \
    -p pose_backend:=yolo11n \
    -p reference_path:="$SCRIPT_DIR/config/temporary_form_reference_yolo11_2d.yaml" \
    "$@"

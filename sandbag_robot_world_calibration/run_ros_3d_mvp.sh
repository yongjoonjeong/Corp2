#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
    exit 1
fi

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    echo "Virtual environment is missing. Run: $SCRIPT_DIR/setup_3d.sh" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
set -u

if ! "$SCRIPT_DIR/.venv/bin/python" -c \
    "import cv2, mediapipe, numpy, yaml, PIL, pyrealsense2" >/dev/null 2>&1; then
    echo "3D Python dependencies are missing. Run: $SCRIPT_DIR/setup_3d.sh" >&2
    exit 1
fi

exec "$SCRIPT_DIR/.venv/bin/python" \
    "$SCRIPT_DIR/three_camera_punch_feedback_node.py" "$@"

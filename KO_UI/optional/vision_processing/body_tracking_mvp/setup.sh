#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
    echo "Install ROS 2 Humble (including rclpy, std_msgs, sensor_msgs) first." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Setup complete. Start the MVP with:"
echo "  $SCRIPT_DIR/run_ros_mvp.sh"

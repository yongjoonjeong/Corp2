#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 브리지는 Doosan 메시지 패키지가 필요 없고 rclpy만 사용한다.
# shellcheck disable=SC1091
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_ros_base
exec "$KO_ROS_PYTHON" "$ROOT/robot_control/ui_robot_bridge.py"

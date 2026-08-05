#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_robot_environment
ko_print_robot_python_paths

echo
echo "Doosan 서비스 확인:"
ros2 service list 2>/dev/null | grep -E '^/dsr01/' | head -n 20 || true

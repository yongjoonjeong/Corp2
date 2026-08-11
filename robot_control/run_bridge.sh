#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 브리지는 Doosan 메시지 패키지가 필요 없고 rclpy만 사용한다.
# shellcheck disable=SC1091
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_ros_base
set +u
force_setup="${BOXING_ROBOT_WS:-$ROOT/force_control/boxing_robot_ws}/install/setup.bash"
if [[ -f "$force_setup" ]]; then
  source "$force_setup"
  echo "[KO] robot bridge force services enabled: $force_setup"
fi
set -u
exec "$KO_ROS_PYTHON" "$ROOT/robot_control/ui_robot_bridge.py"

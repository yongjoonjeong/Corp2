#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_robot_environment
ko_print_robot_python_paths

SERVICE="/dsr01/motion/move_stop"
echo "roboton 연결 확인 중: $SERVICE"
for _ in $(seq 1 30); do
  if ros2 service list 2>/dev/null | grep -qx "$SERVICE"; then
    echo "두산 로봇 서비스 확인 완료"
    exec "$KO_ROS_PYTHON" "$ROOT/robot_control/robot_weaving_node.py"
  fi
  sleep 0.5
done

echo >&2
echo "[오류] 두산 로봇 서비스가 보이지 않습니다: $SERVICE" >&2
echo "터미널 1에서 roboton을 먼저 실행하고 완전히 올라온 뒤 다시 ./run.sh를 실행하세요." >&2
exit 2

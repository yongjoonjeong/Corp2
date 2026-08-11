#!/usr/bin/env bash
# ROS 2 setup scripts probe unset variables, so nounset is intentionally off while sourcing.
set -eo pipefail
set +u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash

# Source the integrated force workspace when present so all ROS interfaces share
# the same overlay. Force HitResult HTTP forwarding is owned only by SessionBridge.
bundled_force_setup="$project_dir/force_control/boxing_robot_ws/install/setup.bash"
external_force_setup="${BOXING_ROBOT_WS:-$HOME/boxing_robot_ws}/install/setup.bash"
force_setup=""
if [[ -f "$bundled_force_setup" ]]; then
  force_setup="$bundled_force_setup"
elif [[ -f "$external_force_setup" ]]; then
  force_setup="$external_force_setup"
fi
if [[ -n "$force_setup" ]]; then
  source "$force_setup"
  echo "[KO] force interface sourced: $force_setup"
else
  echo "[KO] force interface not built; vision UI bridge continues without force interfaces"
fi
set -u

if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  echo "[ERROR] .venv가 없습니다. 먼저 ./setup.sh를 실행하세요." >&2
  exit 1
fi

export KO_UI_BASE_URL="${KO_UI_BASE_URL:-http://127.0.0.1:5000}"
exec "$project_dir/.venv/bin/python" "$project_dir/ui/vision_bridge.py" "$@"

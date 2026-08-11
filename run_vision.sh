#!/usr/bin/env bash
# ROS 2 Humble's generated setup scripts are not compatible with `set -u`:
# they intentionally probe variables that may not exist yet. Enable nounset
# only after the ROS environment has been loaded.
set -eo pipefail
set +u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
else
  echo "[ERROR] ROS 2 Humble setup이 없습니다: /opt/ros/humble/setup.bash"
  exit 1
fi
set -u

if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  echo "[ERROR] .venv가 없습니다. 먼저 ./setup.sh를 실행하세요."
  exit 1
fi

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sandbag-vision-matplotlib-${UID}"
export YOLO_CONFIG_DIR="${TMPDIR:-/tmp}/sandbag-vision-ultralytics-${UID}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
exec "$project_dir/.venv/bin/python" -m sandbag_vision.node "$@"

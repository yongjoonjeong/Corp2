#!/usr/bin/env bash
set -eo pipefail
set +u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${BOXING_ROBOT_WS:-$ROOT/force_control/boxing_robot_ws}"
if [[ ! -d "$WS/src" ]]; then
  echo "[ERROR] force workspace source를 찾을 수 없습니다: $WS/src" >&2
  exit 2
fi
WS="$(cd "$WS" && pwd -P)"
# Old colcon build/install trees contain absolute paths.  When this deployed
# project is moved (or comes from a ZIP without our build stamp), discard only
# generated artifacts before rebuilding so CMake/egg-link caches cannot point
# back to the previous machine/path.  Same-root source updates remain incremental.
build_stamp="$WS/install/.ko_build_stamp"
root_stamp="$WS/install/.ko_build_source_root"
clean_generated=0
if [[ -d "$WS/build" || -d "$WS/install" || -d "$WS/log" ]]; then
  if [[ ! -f "$build_stamp" || ! -f "$root_stamp" ]]; then
    clean_generated=1
  elif [[ "$(cat "$root_stamp" 2>/dev/null || true)" != "$WS" ]]; then
    clean_generated=1
  fi
fi
if [[ "$clean_generated" == "1" ]]; then
  echo "[KO] 이전 경로의 colcon 생성물 정리 후 재빌드: $WS"
  rm -rf "$WS/build" "$WS/install" "$WS/log"
fi
# Use the exact same Doosan overlay discovery as the actual robot nodes.
# This prevents roboton from working while colcon cannot find dsr_msgs2.
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_robot_environment
set -u
cd "$WS"
echo "[KO] force workspace build: $WS"
colcon build --symlink-install
printf '%s\n' "$WS" > "$WS/install/.ko_build_source_root"
touch "$WS/install/.ko_build_stamp"
printf '\n[OK] force workspace built\nsource %q\n' "$WS/install/setup.bash"

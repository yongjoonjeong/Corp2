#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_URL="${KO_UI_BASE_URL:-http://127.0.0.1:5000}"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -n "${VISION_PID:-}" ]] && kill -TERM "$VISION_PID" 2>/dev/null || true
  [[ -n "${BRIDGE_PID:-}" ]] && kill -TERM "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "${VISION_PID:-}" ]] && wait "$VISION_PID" 2>/dev/null || true
  [[ -n "${BRIDGE_PID:-}" ]] && wait "$BRIDGE_PID" 2>/dev/null || true
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export KO_UI_BASE_URL="$UI_URL"
echo "[VISION] Starting KO UI ROS bridge"
"$SCRIPT_DIR/run_ko_ui_bridge.sh" &
BRIDGE_PID=$!
echo "[VISION] Starting calibrated three-camera YOLO 3D runtime"
"$SCRIPT_DIR/run_ros_mvp.sh" "$@" &
VISION_PID=$!

set +e
wait -n "$BRIDGE_PID" "$VISION_PID"
EXIT_CODE=$?
set -e

if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  echo "[EXIT] KO UI vision bridge exited (status=$EXIT_CODE)." >&2
elif ! kill -0 "$VISION_PID" 2>/dev/null; then
  echo "[EXIT] Three-camera YOLO 3D runtime exited (status=$EXIT_CODE)." >&2
else
  echo "[EXIT] Vision supervisor was interrupted (status=$EXIT_CODE)." >&2
fi

exit "$EXIT_CODE"

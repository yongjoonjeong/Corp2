#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_URL="${KO_UI_BASE_URL:-http://127.0.0.1:5000}"

cleanup() {
  [[ -n "${VISION_PID:-}" ]] && kill "$VISION_PID" 2>/dev/null || true
  [[ -n "${BRIDGE_PID:-}" ]] && kill "$BRIDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export KO_UI_BASE_URL="$UI_URL"
"$SCRIPT_DIR/run_ko_ui_bridge.sh" &
BRIDGE_PID=$!
"$SCRIPT_DIR/run_ros_mvp.sh" "$@" &
VISION_PID=$!
wait "$VISION_PID"

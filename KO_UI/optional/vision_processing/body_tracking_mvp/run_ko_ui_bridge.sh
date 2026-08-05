#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
set -u
exec /usr/bin/python3 "$SCRIPT_DIR/ko_ui_bridge.py" "$@"

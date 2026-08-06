#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <calibration-project-root> [-p name:=value ...]" >&2
    echo "Example: $0 \"\$HOME/Downloads/sandbag_robot_world_calibration\"" >&2
    exit 2
fi

CALIBRATION_ROOT="$(cd "$1" 2>/dev/null && pwd)" || {
    echo "[ERROR] Calibration project directory does not exist: $1" >&2
    exit 1
}
shift

CALIBRATION_FILE="$CALIBRATION_ROOT/calibration/results/robot_world.yaml"
CAMERA_ROLE_FILE="$CALIBRATION_ROOT/config/camera_roles.yaml"

if [[ ! -f "$CALIBRATION_FILE" ]]; then
    echo "[ERROR] Robot-world calibration was not found:" >&2
    echo "        $CALIBRATION_FILE" >&2
    exit 1
fi

if [[ ! -f "$CAMERA_ROLE_FILE" ]]; then
    echo "[ERROR] Calibrated LEFT/RIGHT camera role map was not found:" >&2
    echo "        $CAMERA_ROLE_FILE" >&2
    exit 1
fi

echo "[CALIBRATION] $CALIBRATION_FILE"
echo "[CAMERA ROLES] $CAMERA_ROLE_FILE"
echo "[IMPORTANT] Keep the calibrated C270 USB ports and do not use swap_webcams."

exec "$SCRIPT_DIR/run_ros_3d_mvp.sh" --ros-args \
    -p calibration_path:="$CALIBRATION_FILE" \
    -p camera_role_map:="$CAMERA_ROLE_FILE" \
    "$@"

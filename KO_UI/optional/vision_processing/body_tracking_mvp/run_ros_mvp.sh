#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INTEGRATED_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEFAULT_VISION_RUNNER="$INTEGRATED_ROOT/vision/run_ros_3d_mvp.sh"
if [[ ! -x "$DEFAULT_VISION_RUNNER" ]]; then
    DEFAULT_VISION_RUNNER="$INTEGRATED_ROOT/sandbag_robot_world_calibration/run_ros_3d_mvp.sh"
fi
VISION_RUNNER="${KO_3D_VISION_RUNNER:-$DEFAULT_VISION_RUNNER}"

if [[ ! -x "$VISION_RUNNER" ]]; then
    echo "[ERROR] Integrated 3D vision runner was not found or is not executable:" >&2
    echo "        $VISION_RUNNER" >&2
    echo "Run the integrated vision setup first: $INTEGRATED_ROOT/vision/setup_3d.sh" >&2
    exit 1
fi

exec "$VISION_RUNNER" "$@"

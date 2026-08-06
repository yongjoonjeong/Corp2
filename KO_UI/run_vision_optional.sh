#!/usr/bin/env bash
set -euo pipefail

UI_ROOT="$(cd "$(dirname "$0")" && pwd)"
INTEGRATED_ROOT="$(cd "$UI_ROOT/.." && pwd)"
DEFAULT_VISION_RUNNER="$INTEGRATED_ROOT/vision/run_ros_3d_mvp.sh"
if [[ ! -x "$DEFAULT_VISION_RUNNER" ]]; then
  DEFAULT_VISION_RUNNER="$INTEGRATED_ROOT/sandbag_robot_world_calibration/run_ros_3d_mvp.sh"
fi
export KO_3D_VISION_RUNNER="${KO_3D_VISION_RUNNER:-$DEFAULT_VISION_RUNNER}"

if [[ ! -x "$KO_3D_VISION_RUNNER" ]]; then
  echo "[ERROR] 3D vision runner was not found or is not executable:" >&2
  echo "        $KO_3D_VISION_RUNNER" >&2
  echo "Run: chmod +x \"$DEFAULT_VISION_RUNNER\"" >&2
  exit 1
fi

# Do not add ROS or calibration arguments here. The integrated top-level
# launcher owns those values; this wrapper forwards its argv byte-for-byte.
exec "$UI_ROOT/optional/vision_processing/body_tracking_mvp/run_with_ko_ui.sh" "$@"

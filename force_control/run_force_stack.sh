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
MODE="${1:-monitor}"
force_setup="$WS/install/setup.bash"
force_stamp="$WS/install/.ko_build_stamp"
force_root_stamp="$WS/install/.ko_build_source_root"
force_needs_build=0
if [[ ! -f "$force_setup" || ! -f "$force_stamp" || ! -f "$force_root_stamp" ]]; then
  force_needs_build=1
elif [[ "$(cat "$force_root_stamp" 2>/dev/null || true)" != "$WS" ]]; then
  force_needs_build=1
elif find "$WS/src" -type f -newer "$force_stamp" -print -quit | grep -q .; then
  force_needs_build=1
fi
if [[ "$force_needs_build" == "1" ]]; then
  echo "[KO] force workspace 소스/경로 변경 감지 → 최신 소스로 빌드합니다: $WS"
  BOXING_ROBOT_WS="$WS" "$ROOT/force_control/build_force_ws.sh"
fi
# Reuse the same Doosan overlay search used by weaving/preflight/build.
source "$ROOT/robot_control/ros_env.sh"
ko_prepare_robot_environment
source "$force_setup"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
case "$MODE" in
  monitor)
    echo "[KO] RT force monitor only · robot motion session is NOT started"
    exec ros2 launch mitt_hit_bringup rt_force_diagnostic.launch.py
    ;;
  preflight)
    echo "[KO] read-only compliance preflight"
    exec ros2 run mitt_hit_system compliance_preflight
    ;;
  control|integrated)
    cat <<'TXT'
[KO] FULL ROBOT CONTROL MODE
- RT force diagnostic + punch rebound analyzer + mitt positioner + SessionBridge를 함께 실행합니다.
- StartHitTest/StopHitTest의 소유자는 SessionBridge 하나뿐입니다.
- UI training_start → 위빙 정지 → 사용자 맞춤 미트 이동 → 힘 기준 설정 순서로 동작합니다.
- 로봇/TP 안전 설정을 우회하지 않으며 boxing_robot_ws의 fail-closed 설정을 그대로 사용합니다.
TXT
    exec ros2 launch mitt_hit_bringup ko_integrated.launch.py
    ;;
  *)
    echo "usage: $0 monitor|preflight|control|integrated" >&2
    exit 2
    ;;
esac

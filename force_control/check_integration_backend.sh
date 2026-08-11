#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${BOXING_ROBOT_WS:-$ROOT/force_control/boxing_robot_ws}"

echo "[KO] backend workspace: $WS"
for pkg in boxing_interfaces boxing_integration mitt_hit_system mitt_hit_bringup; do
  if [[ ! -d "$WS/src/$pkg" ]]; then
    echo "[ERROR] missing package: $pkg" >&2
    exit 2
  fi
done

echo "[OK] latest boxing backend source packages present"
echo "Build: $ROOT/force_control/build_force_ws.sh"
echo "Next integration: ui_robot_bridge -> session_bridge"

#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

app_mode="user"
force_mode="off"
vision_args=()
show_help=0
while (($#)); do
  case "$1" in
    --admin-mode|--admin_mode|-admin_mode)
      app_mode="admin"
      shift
      ;;
    --user-mode|--user_mode|-user_mode)
      app_mode="user"
      shift
      ;;
    --force-monitor)
      force_mode="monitor"
      shift
      ;;
    --force-control)
      force_mode="control"
      shift
      ;;
    -h|--help)
      show_help=1
      shift
      ;;
    --)
      shift
      vision_args+=("$@")
      break
      ;;
    *)
      vision_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$show_help" == "1" ]]; then
  cat <<'EOF'
KO 통합 실행

사용법:
  ./run_integrated.sh --user-mode [vision args...]
  ./run_integrated.sh --admin-mode [vision args...]
  ./run_integrated.sh --admin-mode --force-monitor
  ./run_integrated.sh --admin-mode --force-control

모드:
  --user-mode   사용자/시연용. 상세 진단과 시스템 설정을 숨깁니다. (기본값)
  --admin-mode  개발/테스트용. 상세 비전 텔레메트리와 시스템 설정을 표시합니다.

힘 시스템:
  --force-monitor  /mitt/rt_sample RT 수집/진단만 실행합니다. 타격 세션 자동 Start는 하지 않습니다.
  --force-control  최종 통합 로봇 모드입니다. RT force + punch_rebound + mitt_positioner +
                   SessionBridge를 함께 실행합니다. Start/StopHitTest는 SessionBridge가 전담합니다.
                   workspace가 아직 빌드되지 않았으면 최초 실행 시 자동 빌드를 시도합니다.

호환 별칭:
  -user_mode, --user_mode, -admin_mode, --admin_mode
EOF
  exit 0
fi

export KO_APP_MODE="$app_mode"
if [[ "$force_mode" == "control" ]]; then
  export KO_ROBOT_SUPPORTED_PUNCHES="${KO_ROBOT_SUPPORTED_PUNCHES:-jab,straight,hook,uppercut}"
else
  export KO_ROBOT_SUPPORTED_PUNCHES="${KO_ROBOT_SUPPORTED_PUNCHES:-jab,straight,hook,uppercut}"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export HOST="127.0.0.1"
export PORT="5000"
export KO_UI_BASE_URL="http://$HOST:$PORT"
ui_url="$KO_UI_BASE_URL"
child_pids=()
child_pgids=()
robot_node_pid=""
robot_bridge_pid=""
shutdown_timeout_s="${KO_SHUTDOWN_TIMEOUT_S:-8}"

if ! [[ "$shutdown_timeout_s" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] KO_SHUTDOWN_TIMEOUT_S는 1 이상의 정수여야 합니다: $shutdown_timeout_s" >&2
  exit 2
fi

start_child() {
  # A separate session keeps every launcher and all of its descendants in one
  # process group. FD 9 is the run_final lock and must not leak to children.
  setsid "$@" 9>&- &
  local pid=$!
  child_pids+=("$pid")
  child_pgids+=("$pid")
  STARTED_CHILD_PID="$pid"
}

process_group_alive() {
  local wanted_pgid="$1"
  ps -eo pgid=,stat= | awk -v wanted="$wanted_pgid" '
    $1 == wanted && $2 !~ /^Z/ { alive=1; exit }
    END { exit(alive ? 0 : 1) }
  '
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  # In full control mode always attempt a logical emergency stop before killing
  # child processes. This also stops the force session if the weaving node itself
  # is the process that failed.
  if [[ "$force_mode" == "control" || -n "$robot_node_pid" ]]; then
    curl -fsS -X POST "$ui_url/api/robot/command" -H "Content-Type: application/json" -d '{"command":"emergency_stop"}' >/dev/null 2>&1 || true
    sleep 0.4
  fi
  for pgid in "${child_pgids[@]}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done

  local deadline=$((SECONDS + shutdown_timeout_s))
  local alive=1
  while ((SECONDS < deadline)); do
    alive=0
    for pgid in "${child_pgids[@]}"; do
      if process_group_alive "$pgid"; then
        alive=1
        break
      fi
    done
    ((alive == 0)) && break
    sleep 0.2
  done

  for pgid in "${child_pgids[@]}"; do
    if process_group_alive "$pgid"; then
      echo "[WARN] 종료 제한시간 초과 · process group $pgid 강제 종료" >&2
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

start_child "$project_dir/ui/run_ui.sh"
ui_pid="$STARTED_CHILD_PID"

ui_ready=0
for _ in $(seq 1 40); do
  if curl -fsS "$ui_url/api/health" >/dev/null 2>&1; then
    ui_ready=1
    break
  fi
  sleep 0.25
done
if [[ "$ui_ready" != "1" ]]; then
  echo "[ERROR] UI가 시작되지 않았습니다: $ui_url" >&2
  exit 1
fi

start_child env KO_UI_BASE_URL="$ui_url" "$project_dir/run_ui_bridge.sh"
bridge_pid="$STARTED_CHILD_PID"

start_child "$project_dir/run_vision.sh" "${vision_args[@]}"
vision_pid="$STARTED_CHILD_PID"

force_status="OFF"
if [[ "$force_mode" != "off" ]]; then
  force_ws="${BOXING_ROBOT_WS:-$project_dir/force_control/boxing_robot_ws}"
  force_ws="$(cd "$force_ws" && pwd -P)"
  force_setup="$force_ws/install/setup.bash"
  force_stamp="$force_ws/install/.ko_build_stamp"
  force_root_stamp="$force_ws/install/.ko_build_source_root"
  force_needs_build=0
  if [[ ! -f "$force_setup" || ! -f "$force_stamp" || ! -f "$force_root_stamp" ]]; then
    force_needs_build=1
  elif [[ "$(cat "$force_root_stamp" 2>/dev/null || true)" != "$force_ws" ]]; then
    force_needs_build=1
  elif find "$force_ws/src" -type f -newer "$force_stamp" -print -quit | grep -q .; then
    force_needs_build=1
  fi
  if [[ "$force_needs_build" == "1" ]]; then
    echo "[KO] boxing_robot_ws 소스/경로 변경 감지 → 최신 소스로 빌드합니다: $force_ws"
    if ! BOXING_ROBOT_WS="$force_ws" "$project_dir/force_control/build_force_ws.sh"; then
      echo "[ERROR] boxing_robot_ws 빌드 실패" >&2
      exit 2
    fi
  fi
  if [[ "$force_mode" == "monitor" ]]; then
    start_child "$project_dir/force_control/run_force_stack.sh" monitor
    force_monitor_pid="$STARTED_CHILD_PID"
    force_status="RT monitor"
  else
    start_child env KO_UI_BASE_URL="$ui_url" "$project_dir/force_control/run_force_stack.sh" integrated
    force_control_pid="$STARTED_CHILD_PID"
    force_status="FULL · RT force + rebound + mitt positioner + session bridge"
  fi
fi

robot_enabled="$KO_ENABLE_ROBOT_WEAVING"
if [[ -z "$robot_enabled" ]]; then
  robot_enabled=1
fi
robot_status="사용 안 함 (KO_ENABLE_ROBOT_WEAVING=0)"
if [[ "$robot_enabled" != "0" ]]; then
  robot_service_ready=0
  for _ in $(seq 1 16); do
    if (
      set +u
      source /opt/ros/humble/setup.bash
      ros2 service list 2>/dev/null | grep -qx "/dsr01/motion/move_stop"
    ); then
      robot_service_ready=1
      break
    fi
    sleep 0.25
  done
  if [[ "$robot_service_ready" == "1" ]]; then
    start_child env KO_UI_BASE_URL="$ui_url" "$project_dir/robot_control/run_bridge.sh"
    robot_bridge_pid="$STARTED_CHILD_PID"
    start_child "$project_dir/robot_control/run_robot_node.sh"
    robot_node_pid="$STARTED_CHILD_PID"
    robot_status="HOME → 준비 자세 → U자 위빙 자동 시작"
  else
    robot_status="roboton 서비스 없음 · UI/비전만 실행"
    echo "[WARN] /dsr01/motion/move_stop이 없어 위빙 노드는 시작하지 않습니다." >&2
  fi
fi

echo "============================================================"
echo "KO ALL-IN-ONE + M0609 최종 통합 실행 중"
echo "MODE    : ${app_mode^^}"
echo "UI      : $ui_url"
if [[ "$app_mode" == "admin" ]]; then
  echo "VISION  : 현재 sandbag_vision/node.py · LEFT · FRONT · RIGHT 3카메라 관리자 프리뷰 + 상세 진단"
else
  echo "VISION  : 현재 sandbag_vision/node.py · 전면 카메라 사용자 프리뷰 + 종합 인식 상태"
fi
echo "ROBOT   : $robot_status"
echo "FORCE   : $force_status"
echo "종료    : Ctrl+C"
echo "============================================================"

critical_pids=("$ui_pid" "$bridge_pid" "$vision_pid")
if [[ -n "${force_control_pid:-}" ]]; then critical_pids+=("$force_control_pid"); fi
if [[ -n "$robot_bridge_pid" ]]; then critical_pids+=("$robot_bridge_pid"); fi
if [[ -n "$robot_node_pid" ]]; then critical_pids+=("$robot_node_pid"); fi
set +e
wait -n "${critical_pids[@]}"
exit_code=$?
set -e
exit "$exit_code"

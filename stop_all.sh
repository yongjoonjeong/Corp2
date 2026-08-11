#!/usr/bin/env bash
set -euo pipefail

shutdown_timeout_s="${KO_SHUTDOWN_TIMEOUT_S:-8}"
if ! [[ "$shutdown_timeout_s" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] KO_SHUTDOWN_TIMEOUT_S는 1 이상의 정수여야 합니다: $shutdown_timeout_s" >&2
  exit 2
fi

# Ask the running UI/robot bridge to stop motion before terminating ROS nodes.
curl -fsS -X POST http://127.0.0.1:5000/api/robot/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"emergency_stop"}' >/dev/null 2>&1 || true

# Include the current checkout and stale copies moved to Trash. Keep these
# patterns specific so unrelated Python and ROS processes are never killed.
patterns=(
  '/home/rokey/KO/(run_final|run_integrated|run_vision|run_ui_bridge)\.sh'
  '/home/rokey/KO/ui/(run_ui\.sh|app\.py|vision_bridge\.py)'
  '/home/rokey/KO/robot_control/(run_bridge\.sh|run_robot_node\.sh|ui_robot_bridge\.py|robot_weaving_node\.py)'
  '/home/rokey/KO/force_control/.*(run_force_stack\.sh|/mitt_hit_system/|/boxing_integration/)'
  '/home/rokey/\.local/share/Trash/files/KO\.[0-9]+/.*(run_final|run_integrated|run_vision|run_ui_bridge|run_ui|run_force_stack|app\.py|vision_bridge\.py|ui_robot_bridge\.py|robot_weaving_node\.py|/mitt_hit_system/|/boxing_integration/)'
  '(^| )[^ ]*python[^ ]* -m sandbag_vision\.node( |$)'
  'ros2 launch mitt_hit_bringup (ko_integrated|rt_force_diagnostic)\.launch\.py'
)

collect_pids() {
  local pattern pid
  for pattern in "${patterns[@]}"; do
    while read -r pid; do
      [[ -n "$pid" && "$pid" != "$$" && "$pid" != "$PPID" ]] && echo "$pid"
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
  done | sort -nu
}

mapfile -t target_pids < <(collect_pids)
if ((${#target_pids[@]})); then
  echo "[KO] TERM: ${target_pids[*]}"
  kill -TERM "${target_pids[@]}" 2>/dev/null || true
fi

deadline=$((SECONDS + shutdown_timeout_s))
while ((SECONDS < deadline)); do
  mapfile -t target_pids < <(collect_pids)
  ((${#target_pids[@]} == 0)) && break
  sleep 0.2
done

mapfile -t target_pids < <(collect_pids)
if ((${#target_pids[@]})); then
  echo "[KO] KILL: ${target_pids[*]}"
  kill -KILL "${target_pids[@]}" 2>/dev/null || true
  sleep 0.2
fi

# Drop stale graph entries after old orphan nodes are gone.
if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" ros2 daemon stop >/dev/null 2>&1 || true
fi

mapfile -t target_pids < <(collect_pids)
if ((${#target_pids[@]})); then
  echo "[ERROR] 종료되지 않은 KO 프로세스: ${target_pids[*]}" >&2
  ps -o pid,ppid,pgid,stat,cmd -p "$(IFS=,; echo "${target_pids[*]}")" >&2 || true
  exit 1
fi

echo "[OK] KO 프로세스가 모두 종료됐습니다."

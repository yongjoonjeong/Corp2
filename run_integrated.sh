#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$ROOT/ui"
VISION_DIR="$ROOT/vision"
[[ -d "$UI_DIR" ]] || UI_DIR="$ROOT/KO_UI"
[[ -d "$VISION_DIR" ]] || VISION_DIR="$ROOT/sandbag_robot_world_calibration"
CALIBRATION_FILE="$VISION_DIR/calibration/results/robot_world.yaml"
CAMERA_ROLE_FILE="$VISION_DIR/config/camera_roles.yaml"
VISION_RUNNER="$VISION_DIR/run_ros_3d_mvp.sh"
KO_UI_HOST="127.0.0.1"
KO_UI_PORT="5000"
KO_UI_URL="http://$KO_UI_HOST:$KO_UI_PORT"

# The UI server, robot bridge, and vision bridge are one local-only contract.
# Ignore ambient web/runner overrides so all three processes always meet at the
# same address and this package always launches its bundled 3D runtime.
if [[ -n "${HOST:-}" && "$HOST" != "$KO_UI_HOST" ]]; then
  echo "[INFO] Ignoring ambient HOST=$HOST; integrated UI uses $KO_UI_HOST."
fi
if [[ -n "${PORT:-}" && "$PORT" != "$KO_UI_PORT" ]]; then
  echo "[INFO] Ignoring ambient PORT=$PORT; integrated UI uses $KO_UI_PORT."
fi
if [[ -n "${KO_UI_BASE_URL:-}" && "$KO_UI_BASE_URL" != "$KO_UI_URL" ]]; then
  echo "[INFO] Ignoring ambient KO_UI_BASE_URL=$KO_UI_BASE_URL."
fi
if [[ -n "${KO_3D_VISION_RUNNER:-}" && "$KO_3D_VISION_RUNNER" != "$VISION_RUNNER" ]]; then
  echo "[INFO] Ignoring ambient KO_3D_VISION_RUNNER=$KO_3D_VISION_RUNNER."
fi
export HOST="$KO_UI_HOST"
export PORT="$KO_UI_PORT"
export KO_UI_BASE_URL="$KO_UI_URL"
export KO_3D_VISION_RUNNER="$VISION_RUNNER"

if [[ ! -x "$UI_DIR/.venv/bin/python" || ! -x "$VISION_DIR/.venv/bin/python" ]]; then
  echo "[ERROR] 통합 실행 환경이 아직 설치되지 않았습니다." >&2
  echo "        먼저 실행하세요: ./setup_integrated.sh" >&2
  exit 1
fi

for path in \
  "$UI_DIR/run.sh" \
  "$UI_DIR/run_ui_only.sh" \
  "$UI_DIR/run_vision_optional.sh" \
  "$VISION_RUNNER" \
  "$CALIBRATION_FILE" \
  "$CAMERA_ROLE_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] 통합 실행 필수 파일이 없습니다: $path" >&2
    exit 1
  fi
done

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] ROS 2 Humble을 찾을 수 없습니다: /opt/ros/humble/setup.bash" >&2
  exit 1
fi

for argument in "$@"; do
  case "${argument,,}" in
    *swap_webcams:=true*)
      echo "[ERROR] swap_webcams:=true는 사용할 수 없습니다." >&2
      echo "        캘리브레이션 당시 LEFT/RIGHT 역할과 USB 포트를 유지하세요." >&2
      exit 2
      ;;
    *calibration_path:=*|*camera_role_map:=*)
      echo "[ERROR] calibration_path와 camera_role_map은 통합본이 자동 지정합니다." >&2
      echo "        해당 파라미터를 제거하고 다시 실행하세요: $argument" >&2
      exit 2
      ;;
    --ros-args)
      echo "[ERROR] --ros-args는 통합 실행기가 자동으로 추가합니다." >&2
      echo "        예: ./run_integrated.sh -p pose_device:=cpu" >&2
      exit 2
      ;;
  esac
done

# UI and vision run concurrently below. Complete the one-time interactive key
# prompt in the foreground so a background UI process never reads /dev/null.
if [[ ! -s "$UI_DIR/.env" ]] || ! grep -q '^OPENAI_API_KEY=.' "$UI_DIR/.env"; then
  echo "[SETUP] 최초 실행용 OpenAI API 키를 먼저 설정합니다."
  "$UI_DIR/.venv/bin/python" "$UI_DIR/configure_api_key.py"
fi

without_robot="${KO_WITHOUT_ROBOT:-0}"
if [[ "$without_robot" == "1" ]]; then
  UI_SCRIPT="$UI_DIR/run_ui_only.sh"
  RUN_MODE="UI + 3D vision (robot disabled)"
else
  UI_SCRIPT="$UI_DIR/run.sh"
  RUN_MODE="UI + robot + 3D vision"
fi

child_pids=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if ((${#child_pids[@]})); then
    echo
    echo "[종료] 통합 프로세스를 정리합니다."
  fi
  local pid
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
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

echo "============================================================"
echo "Sandbag 3D integrated runtime"
echo "MODE        : $RUN_MODE"
echo "CALIBRATION : $CALIBRATION_FILE"
echo "CAMERA ROLES: $CAMERA_ROLE_FILE"
echo "KO UI       : $KO_UI_URL (local only)"
echo "주의        : 캘리 당시 C270 USB 포트를 유지하세요."
echo "종료        : Ctrl+C"
echo "============================================================"

bash "$UI_SCRIPT" &
UI_PID=$!
child_pids+=("$UI_PID")

bash "$UI_DIR/run_vision_optional.sh" \
  --ros-args \
  -p calibration_path:="$CALIBRATION_FILE" \
  -p camera_role_map:="$CAMERA_ROLE_FILE" \
  "$@" &
VISION_PID=$!
child_pids+=("$VISION_PID")

set +e
wait -n "$UI_PID" "$VISION_PID"
exit_code=$?
set -e

if ! kill -0 "$UI_PID" 2>/dev/null; then
  echo "[종료] UI 프로세스가 종료되었습니다." >&2
elif ! kill -0 "$VISION_PID" 2>/dev/null; then
  echo "[종료] 3D 비전 또는 KO UI 비전 브리지가 종료되었습니다." >&2
fi

exit "$exit_code"

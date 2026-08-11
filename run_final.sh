#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 실제 최종 모드는 Wakeword를 기본 활성화한다.
export KO_ENABLE_WAKEWORD="${KO_ENABLE_WAKEWORD:-1}"
# setup_ui.sh에서 base models를 미리 준비하므로 실운영 시작은 네트워크에 의존하지 않는다.
export WAKEWORD_DOWNLOAD_MODELS="${WAKEWORD_DOWNLOAD_MODELS:-0}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

# 중복 실행은 카메라/마이크/ROS 서비스 충돌을 만들 수 있으므로 차단한다.
exec 9>"$ROOT/.ko_final.lock"
if ! flock -n 9; then
  echo "[ERROR] KO 최종 시스템이 이미 실행 중입니다. 기존 프로세스를 먼저 종료하세요." >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "[ERROR] 메인 환경이 없습니다. 먼저 ./setup.sh --build-force 를 실행하세요." >&2
  exit 1
fi
if [[ "$KO_ENABLE_WAKEWORD" != "0" && ! -x "$ROOT/ui/.venv/bin/python" ]]; then
  echo "[ERROR] Wakeword 환경이 없습니다. 먼저 ./ui/setup_ui.sh 를 실행하세요." >&2
  exit 1
fi

force_ws="${BOXING_ROBOT_WS:-$ROOT/force_control/boxing_robot_ws}"
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
  echo "[KO] force workspace 소스/경로 변경 감지 → 최신 소스로 빌드합니다."
  BOXING_ROBOT_WS="$force_ws" "$ROOT/force_control/build_force_ws.sh"
fi

if [[ "${KO_SKIP_HARDWARE_PREFLIGHT:-0}" != "1" ]]; then
  preflight_args=(--hardware)
  if [[ "$KO_ENABLE_WAKEWORD" == "0" ]]; then preflight_args+=(--no-voice); fi
  "$ROOT/.venv/bin/python" "$ROOT/tools/preflight.py" "${preflight_args[@]}"
fi

# later mode arguments override the default --user-mode in run_integrated.sh.
exec "$ROOT/run_integrated.sh" --user-mode --force-control "$@"

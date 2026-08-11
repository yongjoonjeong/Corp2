#!/usr/bin/env bash
set -eo pipefail

ui_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host="${HOST:-127.0.0.1}"
port="${PORT:-5000}"
app_mode="${KO_APP_MODE:-user}"

if [[ "${KO_ENABLE_WAKEWORD:-0}" == "1" ]]; then
  if [[ ! -x "$ui_dir/.venv/bin/python" ]]; then
    echo "[ERROR] 음성 기능용 UI 환경이 없습니다. 먼저 ./ui/setup_ui.sh를 실행하세요." >&2
    exit 1
  fi
  exec "$ui_dir/.venv/bin/python" "$ui_dir/app.py" --host "$host" --port "$port" --app-mode "$app_mode"
fi

# 기본 통합 모드는 비전 UI만 실행하므로 추가 패키지나 API 키가 필요 없다.
exec python3 "$ui_dir/app.py" --host "$host" --port "$port" --app-mode "$app_mode" --no-wakeword

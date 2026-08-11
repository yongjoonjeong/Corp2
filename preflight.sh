#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "[ERROR] .venv가 없습니다. 먼저 ./setup.sh --build-force" >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" "$ROOT/tools/preflight.py" --hardware "$@"

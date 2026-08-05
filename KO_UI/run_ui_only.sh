#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[[ -d .venv ]] || { echo "먼저 ./setup.sh를 실행하세요."; exit 1; }
if [[ ! -s .env ]] || ! grep -q '^OPENAI_API_KEY=.' .env; then
  .venv/bin/python configure_api_key.py
fi
exec .venv/bin/python app.py

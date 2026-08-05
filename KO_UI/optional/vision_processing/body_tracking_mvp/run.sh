#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/body_tracking.py"

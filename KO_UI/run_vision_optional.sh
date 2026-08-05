#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/optional/vision_processing/body_tracking_mvp/run_with_ko_ui.sh" "$@"

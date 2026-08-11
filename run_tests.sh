#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m pytest -q -p no:cacheprovider "$project_dir/tests"
(
  cd "$project_dir/ui"
  "$python_bin" -m unittest discover -s tests -q
)

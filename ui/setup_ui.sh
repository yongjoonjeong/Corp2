#!/usr/bin/env bash
set -euo pipefail

ui_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This deployed openWakeWord/TFLite stack is validated on the project's
# Ubuntu 22.04 / Python 3.10 runtime. Fail early instead of creating a broken venv.
python3 - <<'PYVER'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"[ERROR] Wakeword 환경은 Python 3.10에서 준비하세요. 현재: {sys.version.split()[0]}"
    )
print(f"[OK] Wakeword Python={sys.version.split()[0]}")
PYVER

python3 -m venv "$ui_dir/.venv"
"$ui_dir/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ui_dir/.venv/bin/python" -m pip install -r "$ui_dir/requirements.txt"

# The custom TFLite wakeword uses openWakeWord feature/backbone files. Download
# and validate them once here so final robot startup does not require internet.
"$ui_dir/.venv/bin/python" - "$ui_dir/voice_processing/wake_up_ko.tflite" <<'PYMODEL'
import sys
import openwakeword
from openwakeword.model import Model
model_path = sys.argv[1]
print("[KO] openWakeWord base models 준비")
openwakeword.utils.download_models()
Model(wakeword_models=[model_path])
print("[OK] custom wakeword model init")
PYMODEL

echo "[OK] Wake Word/음성 UI 환경이 준비됐습니다."
echo "     최종 실행: ./run_final.sh"

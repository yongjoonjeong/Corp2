#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

hardware=0
if [[ "${1:-}" == "--hardware" ]]; then
  hardware=1
  shift
fi
if (($#)); then
  echo "Usage: ./test_final.sh [--hardware]" >&2
  exit 2
fi

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1

echo "[1/7] Static integration contracts"
"$PY" "$ROOT/tools/check_final_integration.py"
"$PY" "$ROOT/tools/preflight.py"

echo "[2/7] Python/Shell/JS/YAML/XML syntax"
"$PY" - <<'PY'
from pathlib import Path
import py_compile, xml.etree.ElementTree as ET
import yaml
root=Path('.')
py=[p for p in root.rglob('*.py') if not any(x in p.parts for x in ('build','install','.venv','__pycache__'))]
for p in py: py_compile.compile(str(p), doraise=True)
ys=[p for p in root.rglob('*.yaml') if not any(x in p.parts for x in ('build','install','.venv'))]
for p in ys:
    with p.open(encoding='utf-8') as f: yaml.safe_load(f)
xs=[p for p in root.rglob('package.xml') if not any(x in p.parts for x in ('build','install'))]
for p in xs: ET.parse(p)
print(f'[OK] Python={len(py)} YAML={len(ys)} package.xml={len(xs)}')
PY
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find . -type f -name '*.sh' -not -path '*/build/*' -not -path '*/install/*' -not -path '*/.venv/*' -print0)
while IFS= read -r -d '' f; do node --check "$f" >/dev/null; done < <(find ui/static -type f -name '*.js' -print0)
echo "[OK] shell + JavaScript syntax"

echo "[3/7] UI/API regression"
(
  cd "$ROOT/ui"
  PYTHONPATH=. "$PY" -m unittest discover -s tests -p 'test_*.py' -q
)

echo "[4/7] Force / rebound / mitt pure logic"
PYTHONPATH="$ROOT/force_control/boxing_robot_ws/src/boxing_integration:$ROOT/force_control/boxing_robot_ws/src/mitt_hit_system" \
"$PY" -m pytest -q -p no:cacheprovider \
  force_control/boxing_robot_ws/src/boxing_integration/test/test_user_mitt_calibration.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_collision_sensitivity_controller.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_compliance_controller.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_force_calibrator.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_hit_detector.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_hit_point_estimator.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_hit_record_logger.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_hit_score_calculator.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_hit_test_manager.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_impact_buffer.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_mitt_pose_planner.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_rebound_motion.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_return_to_reference.py \
  force_control/boxing_robot_ws/src/mitt_hit_system/test/test_wrench_frame_adapter.py

echo "[5/7] Vision pure logic"
PYTHONPATH="$ROOT" "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_admin_threecam_mode.py \
  tests/test_ekf_capture.py \
  tests/test_impact.py \
  tests/test_mitt.py \
  tests/test_pose_rotation.py \
  tests/test_snapshot.py \
  tests/test_triangulation.py

if [[ -f /opt/ros/humble/setup.bash ]]; then
  echo "[6/7] ROS-dependent vision import/preview tests"
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  PYTHONPATH="$ROOT" "$PY" -m pytest -q -p no:cacheprovider tests/test_front_preview.py tests/test_preview_overlay.py
else
  echo "[6/7] SKIP ROS-dependent preview tests (/opt/ros/humble/setup.bash 없음)"
fi

if ((hardware)); then
  echo "[7/7] Hardware preflight (NO robot motion)"
  "$PY" "$ROOT/tools/preflight.py" --hardware
else
  echo "[7/7] Hardware preflight skipped (run ./test_final.sh --hardware on target rig)"
fi

echo "============================================================"
echo "[OK] KO final regression suite passed${hardware:+}" 
if ((hardware)); then
  echo "[OK] Target camera/mic/wakeword/roboton preflight passed; no robot motion was issued."
else
  echo "[INFO] Physical camera/mic/roboton checks were not requested."
fi
echo "============================================================"

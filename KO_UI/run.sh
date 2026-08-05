#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "가상환경이 없습니다. ./setup.sh를 먼저 실행합니다."
  ./setup.sh
fi

if [[ ! -s .env ]] || ! grep -q '^OPENAI_API_KEY=.' .env; then
  echo "최초 실행: OpenAI API 키만 한 번 입력하면 .env가 자동 생성됩니다."
  .venv/bin/python configure_api_key.py
fi

mkdir -p logs

cleanup() {
  trap - EXIT INT TERM
  for pid in "${BRIDGE_PID:-}" "${ROBOT_PID:-}" "${UI_PID:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python app.py &
UI_PID=$!

for _ in $(seq 1 40); do
  curl -fsS http://127.0.0.1:5000/api/health >/dev/null 2>&1 && break
  sleep 0.25
done
if ! curl -fsS http://127.0.0.1:5000/api/health >/dev/null 2>&1; then
  echo "[오류] KO UI가 시작되지 않았습니다." >&2
  exit 1
fi

robot_control/run_bridge.sh &
BRIDGE_PID=$!
robot_control/run_robot_node.sh &
ROBOT_PID=$!

echo
echo "=============================================="
echo "KO UI + STT 위빙 통합 실행 중"
echo "브라우저: http://localhost:5000"
echo "터미널 1의 roboton은 계속 실행 상태여야 합니다."
echo "프로그램 시작 → HOME 이동 후 U자 위빙 자동 시작"
echo "Wake Word → 위빙 Soft Stop 후 위빙 준비 자세 복귀"
echo "STT 훈련 명령 → 준비 자세에서 실제 훈련 명령 전달"
echo "종료: Ctrl+C"
echo "=============================================="

set +e
wait -n "$UI_PID" "$BRIDGE_PID" "$ROBOT_PID"
EXIT_CODE=$?
set -e

if ! kill -0 "$ROBOT_PID" 2>/dev/null; then
  echo "[종료] 로봇 위빙 노드가 종료되었습니다. 위의 roboton/ROS 서비스 안내를 확인하세요." >&2
elif ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  echo "[종료] UI-ROS 브리지가 종료되었습니다." >&2
else
  echo "[종료] KO UI가 종료되었습니다."
fi
exit "$EXIT_CODE"

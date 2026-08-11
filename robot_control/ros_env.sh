#!/usr/bin/env bash
# KO 로봇 노드용 ROS 2 / Doosan Python 환경 탐색 도우미.
# 이 파일은 실행하지 않고 run_robot_node.sh / run_bridge.sh에서 source한다.

KO_ROS_PYTHON="${KO_ROS_PYTHON:-/usr/bin/python3}"

_ko_import_base_ok() {
  "$KO_ROS_PYTHON" -c 'import rclpy' >/dev/null 2>&1
}

_ko_import_robot_ok() {
  "$KO_ROS_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
for name in ("rclpy", "dsr_msgs2", "DR_init", "DSR_ROBOT2"):
    if importlib.util.find_spec(name) is None:
        sys.exit(1)
PY
}

_ko_source_setup() {
  local setup_file="$1"
  [[ -f "$setup_file" ]] || return 1
  # ROS setup 파일은 nounset(set -u)과 충돌할 수 있으므로 비활성화한다.
  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
}

ko_prepare_ros_base() {
  if _ko_import_base_ok; then
    return 0
  fi

  if [[ -f /opt/ros/humble/setup.bash ]]; then
    _ko_source_setup /opt/ros/humble/setup.bash
  fi

  if ! _ko_import_base_ok; then
    echo "[오류] 시스템 Python에서 rclpy를 찾지 못했습니다." >&2
    echo "확인: /usr/bin/python3 -c 'import rclpy; print(rclpy.__file__)'" >&2
    return 1
  fi
}

ko_prepare_robot_environment() {
  # 현재 터미널이 이미 올바른 Doosan overlay를 갖고 있으면 그대로 사용한다.
  if _ko_import_robot_ok; then
    echo "Doosan Python 환경: 현재 터미널 환경 사용"
    return 0
  fi

  ko_prepare_ros_base || return 1

  local candidates=()
  [[ -n "${DSR_SETUP_FILE:-}" ]] && candidates+=("$DSR_SETUP_FILE")
  [[ -n "${DSR_WORKSPACE:-}" ]] && candidates+=("$DSR_WORKSPACE/install/setup.bash")
  [[ -n "${COBOT_WS:-}" ]] && candidates+=("$COBOT_WS/install/setup.bash")
  candidates+=(
    "$HOME/cobot_ws/install/setup.bash"
    "$HOME/ros2_ws/install/setup.bash"
    "$HOME/doosan_ws/install/setup.bash"
    "$HOME/dsr_ws/install/setup.bash"
    "$HOME/DoosanRobotics/ros2_ws/install/setup.bash"
    "$HOME/doosan-robotics/ros2_ws/install/setup.bash"
  )

  local setup_file
  local -A seen=()
  for setup_file in "${candidates[@]}"; do
    [[ -n "$setup_file" ]] || continue
    [[ -z "${seen[$setup_file]:-}" ]] || continue
    seen[$setup_file]=1
    [[ -f "$setup_file" ]] || continue

    _ko_source_setup "$setup_file"
    if _ko_import_robot_ok; then
      echo "Doosan Python 환경 자동 선택: $setup_file"
      export KO_DSR_SETUP_FILE="$setup_file"
      return 0
    fi
  done

  # 표준 경로가 아닐 때 dsr_msgs2 설치 위치에서 상위 setup.bash를 찾아본다.
  local package_xml ancestor depth
  while IFS= read -r package_xml; do
    [[ -n "$package_xml" ]] || continue
    ancestor="$(dirname "$package_xml")"
    for depth in 1 2 3 4 5 6; do
      if [[ -f "$ancestor/setup.bash" ]]; then
        setup_file="$ancestor/setup.bash"
        if [[ -z "${seen[$setup_file]:-}" ]]; then
          seen[$setup_file]=1
          _ko_source_setup "$setup_file"
          if _ko_import_robot_ok; then
            echo "Doosan Python 환경 검색 성공: $setup_file"
            export KO_DSR_SETUP_FILE="$setup_file"
            return 0
          fi
        fi
      fi
      ancestor="$(dirname "$ancestor")"
    done
  done < <(find "$HOME" -maxdepth 8 -type f -path '*/share/dsr_msgs2/package.xml' 2>/dev/null | head -n 20)

  echo >&2
  echo "[오류] dsr_msgs2 / DR_init / DSR_ROBOT2 Python 모듈을 찾지 못했습니다." >&2
  echo "roboton 서비스는 보이지만, 터미널 2의 Python 환경에 Doosan overlay가 없습니다." >&2
  echo >&2
  echo "아래 명령으로 실제 설치 위치를 확인하세요:" >&2
  echo "  find \"$HOME\" -path '*/share/dsr_msgs2/package.xml' 2>/dev/null" >&2
  echo "  type roboton" >&2
  echo >&2
  echo "설치 workspace를 알고 있다면 실행 전에 한 번만 지정할 수 있습니다:" >&2
  echo "  DSR_WORKSPACE=~/실제_워크스페이스 ./run.sh" >&2
  echo "또는 setup 파일을 직접 지정:" >&2
  echo "  DSR_SETUP_FILE=~/실제_워크스페이스/install/setup.bash ./run.sh" >&2
  return 1
}

ko_print_robot_python_paths() {
  "$KO_ROS_PYTHON" - <<'PY'
import importlib.util
import sys
print(f"ROS Python: {sys.executable}")
for name in ("dsr_msgs2", "DR_init", "DSR_ROBOT2"):
    spec = importlib.util.find_spec(name)
    location = None if spec is None else (spec.origin or list(spec.submodule_search_locations or []))
    print(f"{name}: {location}")
PY
}

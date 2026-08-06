#!/usr/bin/env bash

source /opt/ros/humble/setup.bash

source_doosan_ros2() {
    if ros2 pkg prefix dsr_bringup2 >/dev/null 2>&1; then
        echo "[DOOSAN] dsr_bringup2 already visible"
        return 0
    fi
    local candidates=(
        "${DOOSAN_ROS2_SETUP:-}"
        "$HOME/ws_cobot_pjt/ws_dsr/install/setup.bash"
        "$HOME/ws_cobot_pjt/install/setup.bash"
        "$HOME/ws_dsr/install/setup.bash"
        "$HOME/doosan_ws/install/setup.bash"
        "$HOME/cobot_ws/install/setup.bash"
        "$HOME/ros2_ws/install/setup.bash"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" && -f "$candidate" ]] || continue
        # shellcheck disable=SC1090
        source "$candidate"
        if ros2 pkg prefix dsr_bringup2 >/dev/null 2>&1; then
            export DOOSAN_ROS2_SETUP="$candidate"
            echo "[DOOSAN] sourced: $candidate"
            return 0
        fi
    done
    echo "[ERROR] Doosan ROS 2 setup not found." >&2
    echo "Set it manually:" >&2
    echo "  export DOOSAN_ROS2_SETUP=/actual/path/install/setup.bash" >&2
    return 1
}

source_doosan_ros2

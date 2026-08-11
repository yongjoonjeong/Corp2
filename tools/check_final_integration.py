#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def python_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def yaml_list(text: str, key: str):
    match = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]+)\]", text, re.M)
    if not match:
        raise AssertionError(f"{key} not found")
    return [float(item.strip()) for item in match.group(1).split(",")]


def main() -> None:
    weaving = ROOT / "robot_control/robot_weaving_node.py"
    mitt_yaml = ROOT / "force_control/boxing_robot_ws/src/mitt_hit_bringup/config/mitt_positioner.params.yaml"
    mitt_positioner = (
        ROOT
        / "force_control/boxing_robot_ws/src/mitt_hit_system/mitt_hit_system/mitt_positioner_node.py"
    ).read_text(encoding="utf-8")
    ui_bridge = (ROOT / "robot_control/ui_robot_bridge.py").read_text(encoding="utf-8")
    session = (ROOT / "force_control/boxing_robot_ws/src/boxing_integration/boxing_integration/session_bridge.py").read_text(encoding="utf-8")
    js = (ROOT / "ui/static/js/app.js").read_text(encoding="utf-8")
    launch = (ROOT / "force_control/boxing_robot_ws/src/mitt_hit_bringup/launch/ko_integrated.launch.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_integrated.sh").read_text(encoding="utf-8")
    final_runner = (ROOT / "run_final.sh").read_text(encoding="utf-8")
    ui_runner = (ROOT / "ui/run_ui.sh").read_text(encoding="utf-8")
    vision_bridge = (ROOT / "ui/vision_bridge.py").read_text(encoding="utf-8")
    app = (ROOT / "ui/app.py").read_text(encoding="utf-8")
    preflight = (ROOT / "tools/preflight.py").read_text(encoding="utf-8")
    coach = (ROOT / "ui/vision_coach.py").read_text(encoding="utf-8")

    home = [float(v) for v in python_constant(weaving, "HOME_J_DEG")]
    ready = [float(v) for v in python_constant(weaving, "WEAVE_READY_J_DEG")]
    mitt_text = mitt_yaml.read_text(encoding="utf-8")
    punch_ready = yaml_list(mitt_text, "reference_joint_deg")
    assert home == yaml_list(mitt_text, "initial_joint_deg"), (home, yaml_list(mitt_text, "initial_joint_deg"))
    # Weaving and punching-ready are intentionally different postures.
    assert ready == [-180.0, 0.0, 90.0, 90.0, 90.0, 0.0], ready
    assert punch_ready == [-90.0, 60.0, 30.0, -90.0, -90.0, 0.0], punch_ready

    weaving_text = weaving.read_text(encoding="utf-8")
    assert "self.dsr.posx(dx, 0.0, dz" in weaving_text, "weaving must stay in BASE XZ"
    wakeword_handler = weaving_text.split("def _on_wakeword", 1)[1].split(
        "def _on_action_command", 1
    )[0]
    assert "request_stop" not in wakeword_handler, "wakeword must keep weaving"
    bridge_wakeword = ui_bridge.split('if command == "wakeword"', 1)[1].split(
        'if command in {"prepare", "start"}', 1
    )[0]
    assert "STOPPING_FOR_VOICE" not in bridge_wakeword
    assert "training_start 전까지 위빙 유지" in bridge_wakeword
    weave_vel = [float(v) for v in python_constant(weaving, "WEAVE_VEL")]
    weave_acc = [float(v) for v in python_constant(weaving, "WEAVE_ACC")]
    assert all(v > 0.0 for v in weave_vel) and all(a > 0.0 for a in weave_acc)
    round_trips = int(python_constant(weaving, "SMOOTH_ROUND_TRIPS_PER_MOVESX"))
    points_per = int(python_constant(weaving, "SMOOTH_POINTS_PER_ROUND_TRIP"))
    assert 4 + round_trips * points_per <= 100
    assert "/robot_boxing/training_request" in ui_bridge
    assert "/robot_boxing/session_command" in ui_bridge
    assert "create_client(StartHitTest" not in ui_bridge and "create_client(StopHitTest" not in ui_bridge
    assert 'self._post_json("/api/force/hit", payload)' in session
    # One HitResult must reach the UI by one route only.
    assert "/api/force/hit" not in vision_bridge
    assert "HitResult" not in vision_bridge
    assert "self._dedupe" in app and "client_session_id" in app
    assert "validated_training_request" in session
    assert "_begin_combination_transition" in session
    assert '"mitt_message": message.detail' in session
    assert 'SetBool, "/mitt/recalibrate_zero"' in session
    assert '"/mitt/recalibrate_zero"' in session
    assert "MITT_CALIBRATION_FAILED" in session
    assert "FORCE_CALIBRATION_REQUIRED_HITS = 5" in session
    assert "REACH_STEP_MM" not in session
    assert "REACH_MAXIMUM_APPROACH_MM" not in session
    assert 'Trigger, "/mitt/start_reach_approach"' in session
    assert 'Trigger, "/mitt/stop_motion"' in session
    assert "phase[\"state\"] = \"STOPPING_AT_CONTACT\"" in session
    assert "MoveStop" in mitt_positioner
    assert "Jog" in mitt_positioner
    assert "JOG_AXIS_TASK_Z = 8" in mitt_positioner
    assert '"/mitt/start_reach_approach"' in mitt_positioner
    assert "DR_SSTOP = 2" in mitt_positioner
    assert '"/mitt/stop_motion"' in mitt_positioner
    assert "MultiThreadedExecutor(num_threads=3)" in mitt_positioner
    assert 'move_stop_service: "/dsr01/motion/move_stop"' in mitt_text
    assert 'jog_service: "/dsr01/motion/jog"' in mitt_text
    assert "reach_jog_speed_percent: 1.0" in mitt_text
    assert "BETWEEN 0 AND 150" not in app.split(
        "CREATE TABLE IF NOT EXISTS user_reach_calibrations", 1
    )[1].split(");", 1)[0]
    assert "if correction_z < 0.0" in app
    rebound_config = (
        ROOT
        / "force_control/boxing_robot_ws/src/mitt_hit_bringup/config/punch_rebound.params.yaml"
    ).read_text(encoding="utf-8")
    analyzer_config = (
        ROOT
        / "force_control/boxing_robot_ws/src/mitt_hit_bringup/config/hit_analyzer.params.yaml"
    ).read_text(encoding="utf-8")
    assert "maximum_total_torque_nm: 30.0" in rebound_config
    assert "maximum_activation_total_torque_nm: 30.0" in rebound_config
    assert "rebound_service_timeout_sec: 1.0" in rebound_config
    assert "return_position_tolerance_mm: 1.5" in rebound_config
    assert "calibration_duration_ms: 500.0" in analyzer_config
    assert "minimum_calibration_samples: 20" in analyzer_config
    assert '"/mitt/motion_guard"' in session and '"/mitt/motion_rezero"' in session
    assert "_begin_reach_calibration" in session
    assert "_begin_force_calibration_correction" in session
    assert 'String, "/sandbag/vision/status", self._on_vision_status' in session
    fist_handler = session.split("def _on_fist_state", 1)[1].split(
        "def _on_vision_status", 1
    )[0]
    assert "_update_reach_fist_stability" not in fist_handler

    for key in ("client_session_id", "user_id", "dominant_hand", "height_cm", "left_punch_reach_cm", "right_punch_reach_cm", "training_type"):
        assert key in js, f"missing JS payload key: {key}"
    assert "MITT_CALIBRATION_ZEROING:" in js
    assert "MITT_CALIBRATION_PUNCH_READY:" in js
    assert "2차 미트 보정 · 힘 센서 영점 조정 중입니다." in js
    assert "2차 미트 보정 · 미트를 5회 펀치해 주세요." in js
    assert "speak(calibrationMessage)" in js
    wait_body = js[js.index('async function waitForRobotTrainingReady'):js.index('async function waitForForceSettle')]
    assert 'robotState === "TRAINING_READY"' in wait_body
    countdown_body = js[js.index('async function runCountdown()'):js.index('function resetTrainingState()')]
    assert 'await waitForRobotTrainingReady()' in countdown_body, 'countdown must never bypass robot-ready gate'
    assert 'waitForRobotTrainingStop' in js and 'waitForForceSettle' in js
    assert 'waitForRobotTrainingPaused' in js
    assert 'String(state.robot?.state || "") !== "WAITING_FOR_HIT"' in js

    for executable in ("rt_force_diagnostic", "hit_analyzer", "mitt_positioner", "session_bridge"):
        assert f'executable="{executable}"' in launch
    assert 'RegisterEventHandler' in launch and 'OnProcessExit' in launch and 'Shutdown' in launch
    assert launch.count('_shutdown_if_node_exits(') >= 2, 'force launch must fail-fast when a critical node exits'
    assert 'run_force_stack.sh" integrated' in runner
    assert 'ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"' in runner
    assert 'KO_ENABLE_WAKEWORD="${KO_ENABLE_WAKEWORD:-1}"' in final_runner
    assert 'WAKEWORD_DOWNLOAD_MODELS="${WAKEWORD_DOWNLOAD_MODELS:-0}"' in final_runner
    assert 'tools/preflight.py" "${preflight_args[@]}"' in final_runner
    assert 'KO_ENABLE_WAKEWORD' in ui_runner and '--no-wakeword' in ui_runner
    # Camera FPS is diagnostic only: low FPS must never block final startup.
    assert "FPS too low" not in preflight
    assert 'fail("카메라 FPS"' not in preflight
    assert "good < 20" not in preflight and "good < 7" not in preflight
    assert "good < 1" in preflight
    assert "England Boxing Level 1 Coaching Handbook" in coach
    assert "coaching_reference" in coach

    print("[OK] KO final integration contracts")
    print(f"  HOME       : {home}")
    print(f"  WEAVE READY: {ready}")
    print(f"  PUNCH READY: {punch_ready}")
    print("  weave plane: BASE XZ")
    print("  owner      : SessionBridge -> Start/StopHitTest")
    print("  launch     : RT force + analyzer + positioner + session bridge, fail-fast")
    print("  calibration: continuous Tool +Z reach + 8 N soft-stop -> 5-hit force centering -> post-move zero")
    print("  UI gate    : TRAINING_READY only after both calibrations")
    print("  pause      : SessionBridge Stop/StartHitTest, timer waits for ready")
    print("  hit route  : SessionBridge only + UI dedupe/session isolation")
    print("  wakeword   : enabled by run_final + offline model preflight")
    print("  camera FPS : diagnostic only; never blocks startup")
    print(f"  movesx     : {4 + round_trips * points_per} waypoints max")


if __name__ == "__main__":
    main()

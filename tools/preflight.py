#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "config/runtime.yaml"
ROLES = ROOT / "config/camera_roles.yaml"
ROBOT_WORLD = ROOT / "calibration/results/robot_world.yaml"
WAKE_MODEL = ROOT / "ui/voice_processing/wake_up_ko.tflite"
MAIN_VENV = ROOT / ".venv/bin/python"
VOICE_VENV = ROOT / "ui/.venv/bin/python"
FORCE_SETUP = ROOT / "force_control/boxing_robot_ws/install/setup.bash"


class PreflightError(RuntimeError):
    pass


def ok(label: str, detail: str = "") -> None:
    print(f"[OK] {label}{': ' + detail if detail else ''}")


def warn(label: str, detail: str = "") -> None:
    print(f"[WARN] {label}{': ' + detail if detail else ''}")


def fail(label: str, detail: str) -> None:
    raise PreflightError(f"{label}: {detail}")


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        fail("파일", f"없음: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        fail("YAML", f"객체 형식이 아님: {path}")
    return data


def static_checks() -> None:
    runtime = load_yaml(RUNTIME)
    roles = load_yaml(ROLES)
    world = load_yaml(ROBOT_WORLD)

    for section in ("target", "pose"):
        raw = runtime.get(section, {}).get("model")
        if not raw:
            fail("모델 설정", f"{section}.model 없음")
        model = (RUNTIME.parent / str(raw)).resolve()
        if not model.is_file():
            fail("모델", f"없음: {model}")
        ok(f"{section} model", model.name)

    camera_cfg = runtime.get("camera", {})
    if int(camera_cfg.get("width", 0)) != 640 or int(camera_cfg.get("height", 0)) != 480:
        fail("카메라 해상도", "캘리브레이션 기준 640x480과 다름")
    configured_fps = int(camera_cfg.get("fps", 0))
    if configured_fps != 30:
        warn("카메라 FPS 설정", f"{configured_fps} fps (권장/기본값 30 fps, 시작 차단 안 함)")
    ok("runtime camera", f"640x480 @ {configured_fps}fps target · FPS is not a startup gate")

    mapped = roles.get("c270_roles", {})
    left = str(mapped.get("left", "")).strip()
    right = str(mapped.get("right", "")).strip()
    if not left or not right or left == right:
        fail("C270 역할", "LEFT/RIGHT stable path가 비어 있거나 동일함. python3 tools/assign_cameras.py 실행")
    ok("C270 role map", f"LEFT={left} | RIGHT={right}")

    cameras = world.get("cameras", {})
    if set(cameras) != {"left", "front", "right"}:
        fail("외부 캘리브레이션", f"카메라 키 불일치: {sorted(cameras)}")
    for name in ("left", "front", "right"):
        item = cameras[name]
        if int(item.get("image_width", 0)) != 640 or int(item.get("image_height", 0)) != 480:
            fail("외부 캘리브레이션", f"{name} 해상도가 640x480이 아님")
        intr = ROOT / str(item.get("intrinsic_file", ""))
        if not intr.is_file():
            fail("내부 캘리브레이션", f"{name}: {intr} 없음")
    ok("3-camera calibration", "left/front/right + intrinsics present")

    if not WAKE_MODEL.is_file() or WAKE_MODEL.stat().st_size < 1024:
        fail("Wakeword model", f"없거나 비정상: {WAKE_MODEL}")
    ok("Wakeword model", WAKE_MODEL.name)

    for path in (ROOT / "run_final.sh", ROOT / "run_integrated.sh", ROOT / "force_control/run_force_stack.sh"):
        if not os.access(path, os.X_OK):
            fail("실행 권한", str(path))
    ok("launch scripts", "executable")


def _run(cmd: list[str], *, timeout: float = 15.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env)


def check_port() -> None:
    host, port = "127.0.0.1", 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            fail("UI port", f"{host}:{port}가 이미 사용 중. 기존 KO 프로세스를 종료하세요")
    ok("UI port", "5000 free")


def check_python_envs(require_voice: bool) -> None:
    if not MAIN_VENV.is_file():
        fail("메인 .venv", "없음. ./setup.sh --build-force 실행")
    result = _run([str(MAIN_VENV), "-c", "import cv2, numpy, yaml, pyrealsense2, mediapipe, ultralytics; print('imports-ok')"])
    if result.returncode != 0:
        fail("메인 Python imports", result.stdout.strip())
    ok("메인 Python imports")

    # rclpy comes from ROS 2 setup.bash rather than pip. Probe it in the same
    # sourced environment used by run_vision.sh.
    ros_import = (
        "set +u; source /opt/ros/humble/setup.bash; "
        + str(MAIN_VENV)
        + " -c 'import rclpy; print(rclpy.__file__)'"
    )
    result = _run(["bash", "-lc", ros_import], timeout=10, env=os.environ.copy())
    if result.returncode != 0:
        fail("ROS2 Python import", result.stdout.strip())
    ok("ROS2 Python import")

    if require_voice:
        if not VOICE_VENV.is_file():
            fail("음성 .venv", "없음. ./ui/setup_ui.sh 실행")
        result = _run([str(VOICE_VENV), "-c", "import openwakeword,pyaudio,numpy,scipy; print(openwakeword.__version__ if hasattr(openwakeword,'__version__') else 'openwakeword-ok')"], timeout=20)
        if result.returncode != 0:
            fail("Wakeword Python imports", result.stdout.strip())
        ok("Wakeword Python imports")


def check_c270_frames() -> None:
    roles = load_yaml(ROLES)["c270_roles"]
    code = r'''
import cv2, json, os, sys, time
paths=json.loads(sys.argv[1])
resolved={role: os.path.realpath(path) for role,path in paths.items()}
if len(set(resolved.values())) != len(resolved):
    raise SystemExit(f"LEFT/RIGHT resolve to the same V4L2 device: {resolved}")
results={}
for role,path in paths.items():
    if not os.path.exists(path):
        raise SystemExit(f"{role} path missing: {path}")
    cap=cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened():
        raise SystemExit(f"{role} open failed: {path}")
    good=0; frame=None
    started=time.monotonic()
    for _ in range(8):
        ok,frame=cap.read()
        if ok and frame is not None:
            good += 1
    elapsed=max(time.monotonic()-started,1e-6)
    actual_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    requested_fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    # Startup only requires that the camera opens and at least one valid frame arrives.
    # Frame count / FPS are diagnostics and never block startup.
    if good < 1 or frame is None:
        raise SystemExit(f"{role} no valid frame from {path}")
    dims=[int(frame.shape[1]),int(frame.shape[0])]
    if dims != [640,480] or [actual_w,actual_h] != [640,480]:
        raise SystemExit(f"{role} resolution mismatch: frame={dims}, device={[actual_w,actual_h]}")
    effective_fps=good/elapsed
    # FPS is diagnostic only. A low measured FPS must never block KO startup.
    results[role]={"size":dims,"frames_received":good,"device_fps":round(requested_fps,1),"effective_fps":round(effective_fps,1),"device":resolved[role]}
print(json.dumps(results))
'''
    result = _run([str(MAIN_VENV), "-c", code, json.dumps({"left": roles["left"], "right": roles["right"]})], timeout=15)
    if result.returncode != 0:
        fail("C270 frame", result.stdout.strip() or "프레임 읽기 실패")
    try:
        dims = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        fail("C270 frame", result.stdout.strip())
    if any(value.get("size") != [640, 480] for value in dims.values()):
        fail("C270 resolution", str(dims))
    for role, value in dims.items():
        effective = float(value.get("effective_fps", 0.0) or 0.0)
        if effective < 15.0:
            warn(f"C270 {role.upper()} FPS", f"{effective:.1f} fps · 프레임은 수신 중이므로 계속 시작")
    ok("C270 frames", f"LEFT/RIGHT connected: {dims}")

def check_realsense() -> None:
    code = r'''
import json, pyrealsense2 as rs, time
ctx=rs.context(); devices=list(ctx.query_devices())
if len(devices)!=1:
    raise SystemExit(f"RealSense exactly one required, found {len(devices)}: {[d.get_info(rs.camera_info.name) for d in devices]}")
serial=devices[0].get_info(rs.camera_info.serial_number)
name=devices[0].get_info(rs.camera_info.name)
p=rs.pipeline(); c=rs.config(); c.enable_device(serial); c.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30); c.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
p.start(c)
try:
    good=0; color=None; depth=None
    started=time.monotonic()
    for _ in range(3):
        frames=p.wait_for_frames(2500)
        color=frames.get_color_frame(); depth=frames.get_depth_frame()
        if color and depth:
            good += 1
    elapsed=max(time.monotonic()-started,1e-6)
    # Startup only requires one synchronized RGB+Depth frame pair.
    # Frame count / FPS are diagnostics and never block startup.
    if good < 1 or not color or not depth: raise SystemExit("RealSense RGB/Depth frame not received")
    if [color.get_width(),color.get_height()] != [640,480] or [depth.get_width(),depth.get_height()] != [640,480]:
        raise SystemExit("RealSense 640x480 stream mismatch")
    effective_fps=good/elapsed
    # FPS is diagnostic only. A low measured FPS must never block KO startup.
    print(json.dumps({"name":name,"serial":serial,"color":[color.get_width(),color.get_height()],"depth":[depth.get_width(),depth.get_height()],"frames_received":good,"effective_fps":round(effective_fps,1)}))
finally:
    p.stop()
'''
    result = _run([str(MAIN_VENV), "-c", code], timeout=12)
    if result.returncode != 0:
        fail("RealSense", result.stdout.strip() or "연결 실패")
    detail = result.stdout.strip().splitlines()[-1]
    try:
        info = json.loads(detail)
        effective = float(info.get("effective_fps", 0.0) or 0.0)
        if effective < 15.0:
            warn("RealSense FPS", f"{effective:.1f} fps · RGB/Depth 프레임은 수신 중이므로 계속 시작")
    except Exception:
        pass
    ok("RealSense RGB+Depth", detail)

def check_voice() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        env_path = ROOT / "ui/.env"
        env_text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
        if not any(line.strip().startswith("OPENAI_API_KEY=") and line.split("=",1)[1].strip() for line in env_text.splitlines()):
            fail("OpenAI STT", "OPENAI_API_KEY가 없음. python3 ui/configure_api_key.py 실행")
    ok("OpenAI STT config")

    code = r'''
import pyaudio, sys
p=pyaudio.PyAudio()
try:
    idx=int(p.get_default_input_device_info()["index"])
    info=p.get_device_info_by_index(idx)
    s=p.open(format=pyaudio.paInt16,channels=1,rate=48000,input=True,input_device_index=idx,frames_per_buffer=3840)
    try:
        data=s.read(3840,exception_on_overflow=False)
        if len(data)<100: raise SystemExit("microphone returned no PCM")
    finally:
        s.stop_stream(); s.close()
    print(f"{idx}:{info.get('name','mic')}")
finally:
    p.terminate()
'''
    result = _run([str(VOICE_VENV), "-c", code], timeout=8)
    if result.returncode != 0:
        fail("Microphone", result.stdout.strip() or "48kHz mono input open failed")
    ok("Microphone 48kHz", result.stdout.strip().splitlines()[-1])

    code = r'''
import sys
from openwakeword.model import Model
model_path=sys.argv[1]
# Base feature models are downloaded once by ui/setup_ui.sh. Final startup
# must not depend on internet access.
m=Model(wakeword_models=[model_path])
print("model-init-ok")
'''
    result = _run([str(VOICE_VENV), "-c", code, str(WAKE_MODEL)], timeout=90)
    if result.returncode != 0:
        fail("Wakeword model init", result.stdout.strip() or "Model init failed")
    ok("Wakeword model init")


def check_force_build() -> None:
    if not FORCE_SETUP.is_file():
        fail("boxing_robot_ws", "install/setup.bash 없음. ./force_control/build_force_ws.sh 실행")
    ok("boxing_robot_ws build")


def check_ros_services() -> None:
    critical = [
        "/dsr01/motion/move_stop",
        "/dsr01/motion/move_joint",
        "/dsr01/motion/move_line",
        "/dsr01/motion/fkin",
        "/dsr01/realtime/read_data_rt",
        "/dsr01/force/task_compliance_ctrl",
        "/dsr01/force/release_compliance_ctrl",
        "/dsr01/system/get_robot_speed_mode",
        "/dsr01/system/change_collision_sensitivity",
    ]
    ws = ROOT / "force_control/boxing_robot_ws"
    ros_env = ROOT / "robot_control/ros_env.sh"
    # Reuse the same Doosan overlay discovery logic as the actual weaving node.
    command = f'''set +u; source {ros_env}; ko_prepare_robot_environment >/dev/null; source {ws}/install/setup.bash; export ROS_DOMAIN_ID=${{ROS_DOMAIN_ID:-77}}; ros2 service list'''
    result = _run(["bash", "-lc", command], timeout=12, env=os.environ.copy())
    if result.returncode != 0:
        fail("ROS service list", result.stdout.strip())
    services = set(line.strip() for line in result.stdout.splitlines() if line.strip().startswith("/"))
    missing = [name for name in critical if name not in services]
    if missing:
        fail("roboton services", "missing: " + ", ".join(missing))
    ok("roboton services", f"{len(critical)}/{len(critical)} critical services")


def hardware_checks(require_voice: bool) -> None:
    check_port()
    check_python_envs(require_voice)
    check_force_build()
    check_c270_frames()
    check_realsense()
    if require_voice:
        check_voice()
    check_ros_services()


def main() -> int:
    parser = argparse.ArgumentParser(description="KO final static/hardware preflight")
    parser.add_argument("--hardware", action="store_true", help="실제 camera/mic/roboton까지 검사")
    parser.add_argument("--no-voice", action="store_true", help="Wakeword/STT 검사 제외")
    args = parser.parse_args()
    try:
        static_checks()
        if args.hardware:
            hardware_checks(require_voice=not args.no_voice)
    except (PreflightError, subprocess.TimeoutExpired) as error:
        print(f"[ERROR] PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    print("[OK] KO preflight complete" + (" · hardware" if args.hardware else " · static"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

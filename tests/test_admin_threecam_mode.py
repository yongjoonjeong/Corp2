from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "ui/static/js/app.js").read_text(encoding="utf-8")
RUN = (ROOT / "run_integrated.sh").read_text(encoding="utf-8")
RUNTIME = (ROOT / "config/runtime.yaml").read_text(encoding="utf-8")
NODE = (ROOT / "sandbag_vision/node.py").read_text(encoding="utf-8")


def test_admin_prefers_three_camera_composite():
    assert 'state.appMode === "admin"' in APP_JS
    assert 'source: "triptych"' in APP_JS
    assert 'path: "/api/vision/preview.jpg"' in APP_JS
    assert 'LEFT · FRONT · RIGHT 3카메라 인식' in APP_JS


def test_user_keeps_front_camera_view():
    marker = '// USER MODE: keep the clean front-camera view.'
    assert marker in APP_JS
    tail = APP_JS.split(marker, 1)[1].split('function refreshVisionPreview', 1)[0]
    assert 'path: "/api/vision/front.jpg"' in tail
    assert 'path: "/api/vision/preview.jpg"' not in tail


def test_integrated_runner_keeps_mode_switches_and_reports_threecam():
    assert '--admin-mode' in RUN
    assert '--user-mode' in RUN
    assert '3카메라 관리자 프리뷰' in RUN


def test_admin_triptych_applies_side_camera_rotation() -> None:
    assert "preview_rotations:" in RUNTIME
    preview_config = RUNTIME.split("preview_rotations:", 1)[1].split("status_topic:", 1)[0]
    assert "left: none" in preview_config
    assert "front: none" in preview_config
    assert "right: none" in preview_config
    assert 'self.preview_rotations.get(camera, "none")' in NODE

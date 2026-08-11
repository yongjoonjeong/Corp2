import numpy as np

from sandbag_vision.node import _append_fist_state_panel, _draw_runtime_status, _fist_state_overlay_line
from sandbag_vision.types import FistState


def state(valid: bool = True) -> FistState:
    return FistState(
        side="left",
        stamp_ns=1_000_000_000,
        position_base_mm=np.array([100.0, -250.0, 1200.0]),
        velocity_base_mm_s=np.array([400.0, 20.0, -30.0]),
        position_std_mm=18.0,
        measurement_age_ms=24.0,
        reprojection_error_px=2.0,
        confidence=0.91,
        camera_count=3,
        camera_mask=7,
        minimum_ray_angle_deg=12.0,
        depth_used=True,
        valid=valid,
    )


def test_fist_state_overlay_reports_base_position_velocity_and_quality() -> None:
    text, color = _fist_state_overlay_line("left", state())
    assert "LEFT  VALID" in text
    assert "P[" in text and "V[" in text
    assert "cams 3" in text and "conf 0.91" in text
    assert color == (80, 230, 80)


def test_fist_state_panel_is_appended_below_camera_canvas() -> None:
    canvas = np.zeros((360, 1440, 3), dtype=np.uint8)
    rendered = _append_fist_state_panel(canvas, {"left": state()})
    assert rendered.shape == (464, 1440, 3)
    assert np.array_equal(rendered[:360], canvas)
    assert np.any(rendered[360:])


def test_guard_gauge_is_green_without_permanent_red_impact_label() -> None:
    ready_canvas = np.zeros((360, 1440, 3), dtype=np.uint8)
    ready = _draw_runtime_status(ready_canvas, "LOCKED", "1/1", "READY", 4, 4, "0", 10.7)
    red_pixels = (ready[:, :, 2] > 200) & (ready[:, :, 1] < 80) & (ready[:, :, 0] < 80)
    green_pixels = (ready[:, :, 1] > 180) & (ready[:, :, 2] < 100)
    assert not np.any(red_pixels)
    assert np.count_nonzero(green_pixels) > 1000

    impact_canvas = np.zeros((360, 1440, 3), dtype=np.uint8)
    impact = _draw_runtime_status(impact_canvas, "LOCKED", "1/1", "IMPACT", 4, 4, "0", 10.7)
    impact_red_pixels = (impact[:, :, 2] > 200) & (impact[:, :, 1] < 80) & (impact[:, :, 0] < 80)
    assert np.any(impact_red_pixels)

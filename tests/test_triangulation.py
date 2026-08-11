import numpy as np

from sandbag_vision.calibration import CameraModel, ThreeCameraCalibration
from sandbag_vision.triangulation import PoseHistory, align_pose_histories, triangulate_robust
from sandbag_vision.types import Landmark2D, PoseSample


def camera(name: str, x_mm: float) -> CameraModel:
    transform = np.eye(4)
    transform[0, 3] = x_mm
    return CameraModel(
        name,
        np.asarray(((600.0, 0.0, 320.0), (0.0, 600.0, 240.0), (0.0, 0.0, 1.0))),
        np.zeros(5),
        transform,
        640,
        480,
    )


def synthetic_calibration() -> ThreeCameraCalibration:
    return ThreeCameraCalibration(
        {
            "left": camera("left", -500.0),
            "front": camera("front", 0.0),
            "right": camera("right", 500.0),
        }
    )


def test_weighted_triangulation_recovers_base_point() -> None:
    calibration = synthetic_calibration()
    expected = np.asarray((120.0, -80.0, 2200.0))
    observations = {
        name: Landmark2D(tuple(model.project_base(expected)), 0.95)
        for name, model in calibration.cameras.items()
    }
    result = triangulate_robust(calibration, observations, 3.0, 5.0)
    assert result is not None
    assert set(result.cameras) == {"left", "front", "right"}
    assert np.linalg.norm(result.point_base_mm - expected) < 1.0
    assert result.reprojection_rms_px < 0.1


def test_triangulation_survives_one_bad_camera() -> None:
    calibration = synthetic_calibration()
    expected = np.asarray((-100.0, 50.0, 1800.0))
    observations = {
        name: Landmark2D(tuple(model.project_base(expected)), 0.9)
        for name, model in calibration.cameras.items()
    }
    observations["right"] = Landmark2D(
        (observations["right"].pixel[0] + 100.0, observations["right"].pixel[1] - 80.0),
        0.9,
    )
    result = triangulate_robust(calibration, observations, 4.0, 5.0)
    assert result is not None
    assert len(result.cameras) == 2
    assert np.linalg.norm(result.point_base_mm - expected) < 2.0


def sample(camera_name: str, stamp_ns: int, x: float) -> PoseSample:
    return PoseSample(
        camera_name,
        stamp_ns,
        stamp_ns,
        (640, 480),
        (0, 0, 640, 480),
        {"left_wrist": Landmark2D((x, 100.0), 0.9)},
        1.0,
    )


def test_pose_histories_interpolate_to_one_capture_time() -> None:
    histories = {name: PoseHistory() for name in ("left", "front", "right")}
    histories["left"].add(sample("left", 1_000_000_000, 100.0))
    histories["left"].add(sample("left", 1_040_000_000, 140.0))
    histories["front"].add(sample("front", 1_010_000_000, 200.0))
    histories["front"].add(sample("front", 1_050_000_000, 240.0))
    histories["right"].add(sample("right", 1_020_000_000, 300.0))
    histories["right"].add(sample("right", 1_060_000_000, 340.0))
    aligned = align_pose_histories(histories, 1_065_000_000, 100_000_000, 80_000_000, 45_000_000)
    assert aligned is not None
    reference, samples = aligned
    assert reference == 1_040_000_000
    assert samples["front"].landmarks["left_wrist"].pixel[0] == 230.0
    assert samples["right"].landmarks["left_wrist"].pixel[0] == 320.0

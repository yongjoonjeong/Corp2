from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .common import (
    average_transforms,
    invert_transform,
    make_transform,
    project_to_rotation_matrix,
    transform_error,
)


@dataclass(frozen=True)
class ExternalSample:
    camera_name: str
    sample_id: str
    T_base_flange_mm: np.ndarray
    T_camera_board_mm: np.ndarray
    reprojection_error_px: float
    corner_count: int


@dataclass(frozen=True)
class CalibrationResult:
    T_flange_board_mm: np.ndarray
    T_base_camera_mm: dict[str, np.ndarray]
    samples_used: list[ExternalSample]
    samples_rejected: list[ExternalSample]
    metrics: dict[str, Any]


def _pack_transform(T: np.ndarray) -> np.ndarray:
    matrix = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]
    )


def _unpack_transform(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(6)
    return make_transform(Rotation.from_rotvec(vector[:3]).as_matrix(), vector[3:])


def _pairwise_motions(samples: list[ExternalSample]) -> list[tuple[np.ndarray, np.ndarray]]:
    motions: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            A = invert_transform(samples[i].T_base_flange_mm) @ samples[j].T_base_flange_mm
            B = invert_transform(samples[i].T_camera_board_mm) @ samples[j].T_camera_board_mm
            rotation_motion = max(
                np.linalg.norm(Rotation.from_matrix(A[:3, :3]).as_rotvec()),
                np.linalg.norm(Rotation.from_matrix(B[:3, :3]).as_rotvec()),
            )
            translation_motion = max(np.linalg.norm(A[:3, 3]), np.linalg.norm(B[:3, 3]))
            if rotation_motion >= np.deg2rad(2.0) or translation_motion >= 10.0:
                motions.append((A, B))
    return motions


def _solve_ax_xb_initial(samples: list[ExternalSample]) -> np.ndarray:
    """Initialize X=T_flange_board from pairwise A X = X B motions."""
    motions = _pairwise_motions(samples)
    if len(motions) < 3:
        return np.eye(4, dtype=np.float64)

    def rotation_residual(rotvec: np.ndarray) -> np.ndarray:
        R_x = Rotation.from_rotvec(rotvec).as_matrix()
        residuals: list[np.ndarray] = []
        for A, B in motions:
            error_rotation = A[:3, :3] @ R_x @ B[:3, :3].T @ R_x.T
            residuals.append(Rotation.from_matrix(project_to_rotation_matrix(error_rotation)).as_rotvec())
        return np.concatenate(residuals)

    seeds = [
        np.zeros(3),
        np.array([np.pi / 2, 0.0, 0.0]),
        np.array([-np.pi / 2, 0.0, 0.0]),
        np.array([0.0, np.pi / 2, 0.0]),
        np.array([0.0, -np.pi / 2, 0.0]),
        np.array([0.0, 0.0, np.pi / 2]),
        np.array([0.0, 0.0, -np.pi / 2]),
        np.array([np.pi, 0.0, 0.0]),
        np.array([0.0, np.pi, 0.0]),
        np.array([0.0, 0.0, np.pi]),
    ]
    best = None
    for seed in seeds:
        result = least_squares(rotation_residual, seed, method="trf", loss="soft_l1")
        score = float(np.mean(rotation_residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    assert best is not None
    R_x = Rotation.from_rotvec(best[1]).as_matrix()

    # (R_A - I)t_X = R_X t_B - t_A
    lhs_rows: list[np.ndarray] = []
    rhs_rows: list[np.ndarray] = []
    for A, B in motions:
        lhs_rows.append(A[:3, :3] - np.eye(3))
        rhs_rows.append(R_x @ B[:3, 3] - A[:3, 3])
    lhs = np.vstack(lhs_rows)
    rhs = np.concatenate(rhs_rows)
    t_x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return make_transform(R_x, t_x)


def _initial_camera_transform(samples: list[ExternalSample], T_flange_board: np.ndarray) -> np.ndarray:
    estimates = [
        sample.T_base_flange_mm
        @ T_flange_board
        @ invert_transform(sample.T_camera_board_mm)
        for sample in samples
    ]
    return average_transforms(estimates)


def _residual_for_sample(
    sample: ExternalSample,
    T_flange_board: np.ndarray,
    T_base_camera: np.ndarray,
    translation_scale_mm: float,
    rotation_scale_deg: float,
) -> np.ndarray:
    lhs = sample.T_base_flange_mm @ T_flange_board
    rhs = T_base_camera @ sample.T_camera_board_mm
    relative = invert_transform(lhs) @ rhs
    translation = relative[:3, 3] / translation_scale_mm
    rotation = Rotation.from_matrix(project_to_rotation_matrix(relative[:3, :3])).as_rotvec()
    rotation = rotation / np.deg2rad(rotation_scale_deg)
    return np.concatenate([translation, rotation])


def _sample_raw_errors(
    sample: ExternalSample,
    T_flange_board: np.ndarray,
    T_base_camera: np.ndarray,
) -> tuple[float, float]:
    lhs = sample.T_base_flange_mm @ T_flange_board
    rhs = T_base_camera @ sample.T_camera_board_mm
    return transform_error(lhs, rhs)


def _joint_refine(
    samples: list[ExternalSample],
    camera_names: list[str],
    T_flange_board_initial: np.ndarray,
    T_base_camera_initial: dict[str, np.ndarray],
    translation_scale_mm: float,
    rotation_scale_deg: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
    x0 = [_pack_transform(T_flange_board_initial)]
    for camera_name in camera_names:
        x0.append(_pack_transform(T_base_camera_initial[camera_name]))
    vector0 = np.concatenate(x0)
    camera_offsets = {name: 6 * (index + 1) for index, name in enumerate(camera_names)}

    def residual(values: np.ndarray) -> np.ndarray:
        T_flange_board = _unpack_transform(values[:6])
        cameras = {
            name: _unpack_transform(values[offset : offset + 6])
            for name, offset in camera_offsets.items()
        }
        residuals = [
            _residual_for_sample(
                sample,
                T_flange_board,
                cameras[sample.camera_name],
                translation_scale_mm,
                rotation_scale_deg,
            )
            for sample in samples
        ]
        return np.concatenate(residuals)

    result = least_squares(
        residual,
        vector0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )
    T_flange_board = _unpack_transform(result.x[:6])
    cameras = {
        name: _unpack_transform(result.x[offset : offset + 6])
        for name, offset in camera_offsets.items()
    }
    return T_flange_board, cameras, result


def _metrics(
    samples: Iterable[ExternalSample],
    T_flange_board: np.ndarray,
    T_base_camera: dict[str, np.ndarray],
) -> dict[str, Any]:
    by_camera: dict[str, dict[str, Any]] = {}
    all_translation: list[float] = []
    all_rotation: list[float] = []
    for sample in samples:
        t_error, r_error = _sample_raw_errors(
            sample, T_flange_board, T_base_camera[sample.camera_name]
        )
        all_translation.append(t_error)
        all_rotation.append(r_error)
        bucket = by_camera.setdefault(
            sample.camera_name,
            {"translation_mm": [], "rotation_deg": [], "reprojection_px": []},
        )
        bucket["translation_mm"].append(t_error)
        bucket["rotation_deg"].append(r_error)
        bucket["reprojection_px"].append(sample.reprojection_error_px)

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "max": float(np.max(array)),
            "p95": float(np.percentile(array, 95)),
        }

    camera_summary: dict[str, Any] = {}
    for name, values in by_camera.items():
        camera_summary[name] = {
            "sample_count": len(values["translation_mm"]),
            "translation_error_mm": summarize(values["translation_mm"]),
            "rotation_error_deg": summarize(values["rotation_deg"]),
            "board_reprojection_error_px": summarize(values["reprojection_px"]),
        }
    return {
        "all_cameras": {
            "sample_count": len(all_translation),
            "translation_error_mm": summarize(all_translation),
            "rotation_error_deg": summarize(all_rotation),
        },
        "per_camera": camera_summary,
    }


def solve_robot_world_calibration(
    samples: list[ExternalSample],
    minimum_samples_per_camera: int = 12,
    maximum_reprojection_error_px: float = 1.5,
    translation_scale_mm: float = 5.0,
    rotation_scale_deg: float = 2.0,
    outlier_translation_mm: float = 12.0,
    outlier_rotation_deg: float = 5.0,
) -> CalibrationResult:
    if not samples:
        raise ValueError("No external calibration samples")
    camera_names = sorted({sample.camera_name for sample in samples})
    accepted: list[ExternalSample] = []
    rejected: list[ExternalSample] = []
    for sample in samples:
        if (
            np.isfinite(sample.reprojection_error_px)
            and sample.reprojection_error_px <= maximum_reprojection_error_px
        ):
            accepted.append(sample)
        else:
            rejected.append(sample)
    grouped = {name: [sample for sample in accepted if sample.camera_name == name] for name in camera_names}
    for name, camera_samples in grouped.items():
        if len(camera_samples) < minimum_samples_per_camera:
            raise RuntimeError(
                f"Camera '{name}' has {len(camera_samples)} valid samples; "
                f"at least {minimum_samples_per_camera} are required"
            )

    initial_x_candidates = [_solve_ax_xb_initial(grouped[name]) for name in camera_names]
    T_flange_board_initial = average_transforms(initial_x_candidates)
    T_base_camera_initial = {
        name: _initial_camera_transform(grouped[name], T_flange_board_initial)
        for name in camera_names
    }
    T_flange_board, T_base_camera, first_result = _joint_refine(
        accepted,
        camera_names,
        T_flange_board_initial,
        T_base_camera_initial,
        translation_scale_mm,
        rotation_scale_deg,
    )

    refined_samples: list[ExternalSample] = []
    outliers: list[ExternalSample] = []
    for sample in accepted:
        t_error, r_error = _sample_raw_errors(
            sample, T_flange_board, T_base_camera[sample.camera_name]
        )
        if t_error > outlier_translation_mm or r_error > outlier_rotation_deg:
            outliers.append(sample)
        else:
            refined_samples.append(sample)
    rejected.extend(outliers)
    refined_grouped = {
        name: [sample for sample in refined_samples if sample.camera_name == name]
        for name in camera_names
    }
    for name, camera_samples in refined_grouped.items():
        if len(camera_samples) < minimum_samples_per_camera:
            raise RuntimeError(
                f"Outlier removal left only {len(camera_samples)} samples for '{name}'. "
                "Collect more diverse, sharper samples or relax thresholds."
            )

    T_flange_board, T_base_camera, final_result = _joint_refine(
        refined_samples,
        camera_names,
        T_flange_board,
        T_base_camera,
        translation_scale_mm,
        rotation_scale_deg,
    )
    metrics = _metrics(refined_samples, T_flange_board, T_base_camera)
    metrics["optimization"] = {
        "first_success": bool(first_result.success),
        "first_cost": float(first_result.cost),
        "final_success": bool(final_result.success),
        "final_cost": float(final_result.cost),
        "final_message": str(final_result.message),
        "rejected_sample_count": len(rejected),
    }
    metrics["frame_equation"] = (
        "T_base_flange * T_flange_board = "
        "T_base_camera[camera] * T_camera_board"
    )
    return CalibrationResult(
        T_flange_board_mm=T_flange_board,
        T_base_camera_mm=T_base_camera,
        samples_used=refined_samples,
        samples_rejected=rejected,
        metrics=metrics,
    )

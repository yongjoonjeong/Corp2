from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


EPS = 1e-12


@dataclass(frozen=True)
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    minimum_charuco_corners: int = 20
    legacy_pattern: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "BoardSpec":
        data = load_yaml(path)
        spec = cls(
            dictionary=str(data["dictionary"]),
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_length_mm=float(data["square_length_mm"]),
            marker_length_mm=float(data["marker_length_mm"]),
            minimum_charuco_corners=int(data.get("minimum_charuco_corners", 20)),
            legacy_pattern=bool(data.get("legacy_pattern", False)),
        )
        if spec.squares_x < 2 or spec.squares_y < 2:
            raise ValueError("ChArUco squares_x/squares_y must be at least 2")
        if not (0 < spec.marker_length_mm < spec.square_length_mm):
            raise ValueError("marker_length_mm must be smaller than square_length_mm")
        return spec


@dataclass(frozen=True)
class IntrinsicCalibration:
    camera_name: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float
    mean_reprojection_error_px: float
    board: BoardSpec

    @classmethod
    def load(cls, path: str | Path) -> "IntrinsicCalibration":
        data = load_yaml(path)
        board_data = data["board"]
        board = BoardSpec(
            dictionary=str(board_data["dictionary"]),
            squares_x=int(board_data["squares_x"]),
            squares_y=int(board_data["squares_y"]),
            square_length_mm=float(board_data["square_length_mm"]),
            marker_length_mm=float(board_data["marker_length_mm"]),
            minimum_charuco_corners=int(board_data.get("minimum_charuco_corners", 20)),
            legacy_pattern=bool(board_data.get("legacy_pattern", False)),
        )
        return cls(
            camera_name=str(data["camera_name"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3),
            distortion=np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1),
            rms_px=float(data["rms_reprojection_error_px"]),
            mean_reprojection_error_px=float(data["mean_reprojection_error_px"]),
            board=board,
        )

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "camera_name": self.camera_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion.reshape(-1).tolist(),
            "rms_reprojection_error_px": float(self.rms_px),
            "mean_reprojection_error_px": float(self.mean_reprojection_error_px),
            "board": {
                "dictionary": self.board.dictionary,
                "squares_x": self.board.squares_x,
                "squares_y": self.board.squares_y,
                "square_length_mm": self.board.square_length_mm,
                "marker_length_mm": self.board.marker_length_mm,
                "minimum_charuco_corners": self.board.minimum_charuco_corners,
                "legacy_pattern": self.board.legacy_pattern,
            },
        }
        if extra:
            payload.update(extra)
        save_yaml(path, payload)


@dataclass(frozen=True)
class CharucoObservation:
    corners_px: np.ndarray  # N x 2
    ids: np.ndarray  # N
    object_points_mm: np.ndarray  # N x 3
    marker_count: int


@dataclass(frozen=True)
class PoseEstimate:
    T_camera_board_mm: np.ndarray
    reprojection_error_px: float
    corner_count: int


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = project_to_rotation_matrix(R)
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    matrix = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = R.T
    result[:3, 3] = -R.T @ t
    return result


def transform_point(T_target_source: np.ndarray, point_source: Sequence[float]) -> np.ndarray:
    T = np.asarray(T_target_source, dtype=np.float64).reshape(4, 4)
    point = np.asarray(point_source, dtype=np.float64).reshape(3)
    return T[:3, :3] @ point + T[:3, 3]


def transform_vector(T_target_source: np.ndarray, vector_source: Sequence[float]) -> np.ndarray:
    T = np.asarray(T_target_source, dtype=np.float64).reshape(4, 4)
    vector = np.asarray(vector_source, dtype=np.float64).reshape(3)
    return T[:3, :3] @ vector


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return value / norm


def project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(value)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def transform_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    relative = invert_transform(T_a) @ np.asarray(T_b, dtype=np.float64).reshape(4, 4)
    translation_mm = float(np.linalg.norm(relative[:3, 3]))
    rotation_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec())))
    return translation_mm, rotation_deg


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    matrices = [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in transforms]
    if not matrices:
        raise ValueError("No transforms to average")
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in matrices]))
    try:
        mean_rotation = rotations.mean().as_matrix()
    except AttributeError:
        quaternions = rotations.as_quat()
        accumulator = np.zeros((4, 4), dtype=np.float64)
        for q in quaternions:
            accumulator += np.outer(q, q)
        _, eigenvectors = np.linalg.eigh(accumulator)
        mean_rotation = Rotation.from_quat(eigenvectors[:, -1]).as_matrix()
    translation = np.median(np.stack([item[:3, 3] for item in matrices]), axis=0)
    return make_transform(mean_rotation, translation)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = project_to_rotation_matrix(np.asarray(R_a).reshape(3, 3).T @ np.asarray(R_b).reshape(3, 3))
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))


def doosan_zyz_degrees_to_matrix(abc_deg: Sequence[float]) -> np.ndarray:
    a, b, c = np.radians(np.asarray(abc_deg, dtype=np.float64).reshape(3))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    rz_c = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    return project_to_rotation_matrix(rz_a @ ry_b @ rz_c)


def doosan_posx_to_transform(posx_mm_deg: Sequence[float]) -> np.ndarray:
    values = np.asarray(posx_mm_deg, dtype=np.float64).reshape(-1)
    if values.size < 6:
        raise ValueError("Doosan posx must contain at least 6 values")
    return make_transform(doosan_zyz_degrees_to_matrix(values[3:6]), values[:3])


def matrix_to_doosan_zyz_degrees(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to one valid intrinsic Z-Y-Z Euler representation.

    Euler angles are not unique. Compare rotation matrices or normal vectors, not
    component-wise ABC differences.
    """
    Rm = project_to_rotation_matrix(rotation)
    with np.errstate(invalid="ignore"):
        b = math.acos(float(np.clip(Rm[2, 2], -1.0, 1.0)))
    sb = math.sin(b)
    if abs(sb) > 1e-9:
        a = math.atan2(Rm[1, 2], Rm[0, 2])
        c = math.atan2(Rm[2, 1], -Rm[2, 0])
    else:
        # At a ZYZ singularity only a+c (b=0) or a-c (b=pi) is observable.
        if Rm[2, 2] > 0:
            a = math.atan2(Rm[1, 0], Rm[0, 0])
            c = 0.0
        else:
            a = math.atan2(-Rm[1, 0], -Rm[0, 0])
            c = 0.0
    result = np.degrees([a, b, c])
    return (result + 180.0) % 360.0 - 180.0


def board_objects(spec: BoardSpec) -> tuple[Any, Any]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not found. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, spec.dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {spec.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, spec.dictionary)))
    size = (spec.squares_x, spec.squares_y)
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            board = cv2.aruco.CharucoBoard(size, spec.square_length_mm, spec.marker_length_mm, dictionary)
        except TypeError:
            board = cv2.aruco.CharucoBoard_create(
                spec.squares_x,
                spec.squares_y,
                spec.square_length_mm,
                spec.marker_length_mm,
                dictionary,
            )
    else:
        board = cv2.aruco.CharucoBoard_create(
            spec.squares_x,
            spec.squares_y,
            spec.square_length_mm,
            spec.marker_length_mm,
            dictionary,
        )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(spec.legacy_pattern))
    return dictionary, board


def board_chessboard_corners(board: Any) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        points = board.getChessboardCorners()
    else:
        points = board.chessboardCorners
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def create_detector_parameters() -> Any:
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(params, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect_charuco(
    image: np.ndarray,
    spec: BoardSpec,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> tuple[CharucoObservation | None, np.ndarray]:
    dictionary, board = board_objects(spec)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    parameters = create_detector_parameters()
    annotated = image.copy()

    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_parameters = cv2.aruco.CharucoParameters()
        if camera_matrix is not None:
            charuco_parameters.cameraMatrix = np.asarray(camera_matrix, dtype=np.float64)
        if distortion is not None:
            charuco_parameters.distCoeffs = np.asarray(distortion, dtype=np.float64)
        detector = cv2.aruco.CharucoDetector(
            board, charuco_parameters, parameters
        )
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        if marker_ids is None or len(marker_ids) == 0:
            return None, annotated
        interpolation_kwargs: dict[str, Any] = {}
        if camera_matrix is not None:
            interpolation_kwargs["cameraMatrix"] = np.asarray(camera_matrix, dtype=np.float64)
        if distortion is not None:
            interpolation_kwargs["distCoeffs"] = np.asarray(distortion, dtype=np.float64)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board, **interpolation_kwargs
        )

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
        return None, annotated
    corners_px = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    all_object_points = board_chessboard_corners(board)
    if int(np.max(ids)) >= len(all_object_points):
        raise RuntimeError("Detected ChArUco ID exceeds board corner count")
    object_points = all_object_points[ids]
    cv2.aruco.drawDetectedCornersCharuco(
        annotated,
        corners_px.reshape(-1, 1, 2).astype(np.float32),
        ids.reshape(-1, 1),
    )
    marker_count = 0 if marker_ids is None else len(marker_ids)
    return (
        CharucoObservation(
            corners_px=corners_px,
            ids=ids,
            object_points_mm=object_points,
            marker_count=marker_count,
        ),
        annotated,
    )


def estimate_board_pose(
    image: np.ndarray,
    intrinsic: IntrinsicCalibration,
    minimum_corners: int | None = None,
) -> tuple[PoseEstimate | None, np.ndarray]:
    minimum = minimum_corners or intrinsic.board.minimum_charuco_corners
    observation, annotated = detect_charuco(
        image,
        intrinsic.board,
        camera_matrix=intrinsic.camera_matrix,
        distortion=intrinsic.distortion,
    )
    if observation is None or len(observation.ids) < minimum:
        return None, annotated
    object_points = observation.object_points_mm.astype(np.float64)
    image_points = observation.corners_px.astype(np.float64)
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None

    # ChArUco is planar. IPPE exposes both planar pose candidates; select the
    # positive-depth candidate with the lowest reprojection error, then refine.
    if hasattr(cv2, "SOLVEPNP_IPPE") and hasattr(cv2, "solvePnPGeneric"):
        try:
            generic = cv2.solvePnPGeneric(
                object_points,
                image_points,
                intrinsic.camera_matrix,
                intrinsic.distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            success = bool(generic[0])
            candidate_rvecs = generic[1] if len(generic) > 1 else []
            candidate_tvecs = generic[2] if len(generic) > 2 else []
            best_score = float("inf")
            if success:
                for candidate_rvec, candidate_tvec in zip(candidate_rvecs, candidate_tvecs):
                    candidate_t = np.asarray(candidate_tvec, dtype=np.float64).reshape(3)
                    if candidate_t[2] <= 0:
                        continue
                    candidate_projected, _ = cv2.projectPoints(
                        object_points,
                        candidate_rvec,
                        candidate_tvec,
                        intrinsic.camera_matrix,
                        intrinsic.distortion,
                    )
                    candidate_projected = np.asarray(candidate_projected).reshape(-1, 2)
                    score = float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (candidate_projected - image_points.reshape(-1, 2)) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    )
                    if score < best_score:
                        best_score = score
                        rvec = np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
                        tvec = np.asarray(candidate_tvec, dtype=np.float64).reshape(3, 1)
        except cv2.error:
            rvec = None
            tvec = None

    if rvec is None or tvec is None:
        ok, fallback_rvec, fallback_tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic.camera_matrix,
            intrinsic.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, annotated
        rvec = np.asarray(fallback_rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(fallback_tvec, dtype=np.float64).reshape(3, 1)

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            intrinsic.camera_matrix,
            intrinsic.distortion,
            rvec,
            tvec,
        )
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic.camera_matrix,
        intrinsic.distortion,
    )
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - observation.corners_px) ** 2, axis=1))))
    R_cam_board, _ = cv2.Rodrigues(rvec)
    T_camera_board = make_transform(R_cam_board, np.asarray(tvec).reshape(3))
    try:
        cv2.drawFrameAxes(
            annotated,
            intrinsic.camera_matrix,
            intrinsic.distortion,
            rvec,
            tvec,
            intrinsic.board.square_length_mm * 2.0,
            2,
        )
    except TypeError:
        cv2.drawFrameAxes(
            annotated,
            intrinsic.camera_matrix,
            intrinsic.distortion,
            rvec,
            tvec,
            intrinsic.board.square_length_mm * 2.0,
        )
    return PoseEstimate(T_camera_board, error, len(observation.ids)), annotated


def put_status_lines(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    output = image.copy()
    y = 28
    for line in lines:
        cv2.putText(
            output,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return output


def validate_rotation_matrix(matrix: np.ndarray, tolerance: float = 1e-3) -> bool:
    R = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(R)):
        return False
    if np.linalg.norm(R) < 0.5:
        return False
    return bool(
        np.linalg.norm(R.T @ R - np.eye(3)) < tolerance
        and abs(np.linalg.det(R) - 1.0) < tolerance
    )

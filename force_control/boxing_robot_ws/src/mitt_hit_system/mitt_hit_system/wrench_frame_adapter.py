"""Coordinate-frame correction for Doosan wrench samples."""

import math
from typing import Sequence


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Wrench6 = tuple[float, float, float, float, float, float]


def rotation_from_zyz_degrees(angles: Sequence[float]) -> Matrix3:
    """Return BASE-from-TOOL rotation for Doosan Euler ZYZ angles."""
    if len(angles) != 3:
        raise ValueError("ZYZ angles must contain three values")
    values = tuple(float(value) for value in angles)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ZYZ angles must be finite")
    alpha, beta, gamma = (math.radians(value) for value in values)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return normalize_rotation_matrix(
        (
            (ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb),
            (sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb),
            (-sb * cg, sb * sg, cb),
        )
    )


def normalize_rotation_matrix(rows: Sequence[Sequence[float]]) -> Matrix3:
    """Validate a BASE-from-TOOL rotation matrix returned by Doosan."""
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("rotation matrix must be 3x3")
    matrix = tuple(tuple(float(value) for value in row) for row in rows)
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("rotation matrix values must be finite")

    # Reject stale/corrupt responses without demanding unrealistic float precision.
    for row in matrix:
        norm = sum(value * value for value in row)
        if not math.isclose(norm, 1.0, abs_tol=0.02):
            raise ValueError("rotation matrix row is not unit length")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        dot = sum(matrix[first][i] * matrix[second][i] for i in range(3))
        if not math.isclose(dot, 0.0, abs_tol=0.02):
            raise ValueError("rotation matrix rows are not orthogonal")
    return matrix  # type: ignore[return-value]


def rotation_distance_degrees(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    """Return the shortest physical angle between two rotation matrices."""
    first_matrix = normalize_rotation_matrix(first)
    second_matrix = normalize_rotation_matrix(second)
    relative_trace = sum(
        first_matrix[row][column] * second_matrix[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def rotate_base_vector_to_tool(vector: Sequence[float], rotation: Matrix3) -> Vector3:
    """Apply R transpose, matching Doosan's BASE-to-TOOL force conversion."""
    if len(vector) != 3:
        raise ValueError("vector must contain three values")
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vector values must be finite")
    return tuple(
        sum(rotation[row][column] * values[row] for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def correct_base_wrench_to_tool(
    wrench: Sequence[float], rotation: Sequence[Sequence[float]]
) -> Wrench6:
    """Rotate both force and moment from BASE into the TOOL coordinate frame."""
    if len(wrench) != 6:
        raise ValueError("wrench must contain six values")
    values = tuple(float(value) for value in wrench)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wrench values must be finite")
    matrix = normalize_rotation_matrix(rotation)
    force = rotate_base_vector_to_tool(values[:3], matrix)
    moment = rotate_base_vector_to_tool(values[3:], matrix)
    return (*force, *moment)


def correct_doosan_rt_wrench_to_tool(
    wrench: Sequence[float], rotation: Sequence[Sequence[float]]
) -> Wrench6:
    """Normalize the mixed frame used by this controller's RT force field.

    The measured ``external_tcp_force`` force triplet is BASE-referenced, while
    its moment triplet is already TOOL-referenced.  Rotate only force so the
    resulting six values consistently describe the mounted mitt in TOOL.
    """
    if len(wrench) != 6:
        raise ValueError("wrench must contain six values")
    values = tuple(float(value) for value in wrench)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wrench values must be finite")
    matrix = normalize_rotation_matrix(rotation)
    force = rotate_base_vector_to_tool(values[:3], matrix)
    return (*force, *values[3:])


def solve_base_wrench_from_joint_torque(
    jacobian: Sequence[Sequence[float]], joint_torque: Sequence[float]
) -> Wrench6:
    """Solve ``J.T * wrench = external_joint_torque`` for a BASE wrench."""
    if len(jacobian) != 6 or any(len(row) != 6 for row in jacobian):
        raise ValueError("jacobian must be 6x6")
    if len(joint_torque) != 6:
        raise ValueError("joint torque must contain six values")
    rows = [[float(value) for value in row] for row in jacobian]
    torque = [float(value) for value in joint_torque]
    if not all(math.isfinite(value) for row in rows for value in row) or not all(
        math.isfinite(value) for value in torque
    ):
        raise ValueError("jacobian and joint torque values must be finite")

    # Partial-pivot Gaussian elimination on J transpose.  Keeping this small
    # solver local avoids adding a new runtime dependency to the 250 Hz node.
    augmented = [
        [rows[column][row] for column in range(6)] + [torque[row]]
        for row in range(6)
    ]
    for pivot_column in range(6):
        pivot_row = max(
            range(pivot_column, 6),
            key=lambda row: abs(augmented[row][pivot_column]),
        )
        pivot = augmented[pivot_row][pivot_column]
        if abs(pivot) < 1e-8:
            raise ValueError("jacobian transpose is singular")
        augmented[pivot_column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_column],
        )
        pivot = augmented[pivot_column][pivot_column]
        for column in range(pivot_column, 7):
            augmented[pivot_column][column] /= pivot
        for row in range(6):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            for column in range(pivot_column, 7):
                augmented[row][column] -= factor * augmented[pivot_column][column]

    result = tuple(augmented[row][6] for row in range(6))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("solved wrench is not finite")
    return result  # type: ignore[return-value]


def fuse_doosan_rt_wrench_to_tool(
    external_tcp_force: Sequence[float],
    external_joint_torque: Sequence[float],
    jacobian: Sequence[Sequence[float]],
    rotation: Sequence[Sequence[float]],
) -> Wrench6:
    """Keep controller force while recovering observable TOOL moments."""
    corrected = correct_doosan_rt_wrench_to_tool(external_tcp_force, rotation)
    solved_base = solve_base_wrench_from_joint_torque(
        jacobian, external_joint_torque
    )
    matrix = normalize_rotation_matrix(rotation)
    moment_tool = rotate_base_vector_to_tool(solved_base[3:], matrix)
    return (*corrected[:3], *moment_tool)

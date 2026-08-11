"""Plan a user-adjusted mitt pose from a photographed robot posture."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from mitt_hit_system.wrench_frame_adapter import (
    normalize_rotation_matrix,
    rotation_from_zyz_degrees,
)


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Pose6 = tuple[float, float, float, float, float, float]


class PunchType(str, Enum):
    STRAIGHT = "STRAIGHT"
    LEFT_HOOK = "LEFT_HOOK"
    RIGHT_HOOK = "RIGHT_HOOK"
    UPPERCUT = "UPPERCUT"


@dataclass(frozen=True)
class PersonMeasurement:
    height_mm: float
    arm_length_mm: float

    def validate(self) -> None:
        if not all(math.isfinite(v) for v in (self.height_mm, self.arm_length_mm)):
            raise ValueError("person measurement values must be finite")
        if self.height_mm <= 0.0:
            raise ValueError("person height must be positive")
        if self.arm_length_mm <= 0.0:
            raise ValueError("person arm length must be positive")


@dataclass(frozen=True)
class MittPosePlan:
    tcp_pose_mm_deg: Pose6
    strike_range_mm: float
    target_height_mm: float
    person_distance_mm: float
    face_normal_base: Vector3


class MittPosePlanner:
    """Keep the final photographed posture and apply small body offsets."""

    def __init__(
        self,
        arm_extension_ratio: float = 1.0,
        hook_arm_extension_ratio: float = 0.70,
        uppercut_arm_extension_ratio: float = 0.55,
        shoulder_height_ratio: float = 0.818,
        uppercut_height_ratio: float = 0.72,
        hook_face_angle_deg: float = 55.0,
        reference_tcp_pose_mm_deg: Sequence[float] = (
            216.30,
            -711.58,
            328.37,
            175.10,
            89.74,
            83.08,
        ),
        reference_person_height_mm: float = 1730.0,
        reference_person_arm_length_mm: float = 666.0,
    ) -> None:
        self.arm_extension_ratio = float(arm_extension_ratio)
        self.hook_arm_extension_ratio = float(hook_arm_extension_ratio)
        self.uppercut_arm_extension_ratio = float(uppercut_arm_extension_ratio)
        self.shoulder_height_ratio = float(shoulder_height_ratio)
        self.uppercut_height_ratio = float(uppercut_height_ratio)
        self.hook_face_angle_deg = float(hook_face_angle_deg)
        self.reference_tcp_pose_mm_deg = _pose6(reference_tcp_pose_mm_deg)
        self.reference_person_height_mm = float(reference_person_height_mm)
        self.reference_person_arm_length_mm = float(reference_person_arm_length_mm)
        for label, value in (
            ("arm extension ratio", self.arm_extension_ratio),
            ("hook arm extension ratio", self.hook_arm_extension_ratio),
            ("uppercut arm extension ratio", self.uppercut_arm_extension_ratio),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")
        for label, value in (
            ("shoulder height ratio", self.shoulder_height_ratio),
            ("uppercut height ratio", self.uppercut_height_ratio),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{label} must be in (0, 1)")
        if not math.isfinite(self.hook_face_angle_deg) or not 0.0 < self.hook_face_angle_deg <= 90.0:
            raise ValueError("hook face angle must be in (0, 90] degrees")
        if not math.isfinite(self.reference_person_height_mm) or self.reference_person_height_mm <= 0.0:
            raise ValueError("reference person height must be positive")
        if not math.isfinite(self.reference_person_arm_length_mm) or self.reference_person_arm_length_mm <= 0.0:
            raise ValueError("reference person arm length must be positive")

    def plan(
        self, person: PersonMeasurement, punch_type: PunchType = PunchType.STRAIGHT
    ) -> MittPosePlan:
        person.validate()
        selected_type = PunchType(punch_type)
        extension_ratio = {
            PunchType.STRAIGHT: self.arm_extension_ratio,
            PunchType.LEFT_HOOK: self.hook_arm_extension_ratio,
            PunchType.RIGHT_HOOK: self.hook_arm_extension_ratio,
            PunchType.UPPERCUT: self.uppercut_arm_extension_ratio,
        }[selected_type]
        height_ratio = self.uppercut_height_ratio if selected_type is PunchType.UPPERCUT else self.shoulder_height_ratio
        strike_range = person.arm_length_mm * extension_ratio
        reference_position = self.reference_tcp_pose_mm_deg[:3]
        reference_angles = self.reference_tcp_pose_mm_deg[3:]
        reference_rotation = rotation_from_zyz_degrees(reference_angles)
        flange_forward = tuple(
            reference_rotation[row][2] for row in range(3)
        )
        person_direction = _normalize(
            (flange_forward[0], flange_forward[1], 0.0)
        )
        reference_strike = self.reference_person_arm_length_mm * self.arm_extension_ratio
        # Keep uppercuts at the user's calibrated horizontal mitt position.
        # Applying the shortened uppercut arm-extension ratio as a forward TCP
        # offset pushed real targets close to the M0609 workspace boundary, so
        # MoveL was rejected and the mitt remained in its front-facing pose.
        # The uppercut-specific reach is still retained as plan metadata; only
        # its TCP distance adjustment is suppressed.
        distance_offset = (
            0.0
            if selected_type is PunchType.UPPERCUT
            else reference_strike - strike_range
        )
        target_position = (
            reference_position[0] + distance_offset * person_direction[0],
            reference_position[1] + distance_offset * person_direction[1],
            reference_position[2] + person.height_mm * height_ratio
            - self.reference_person_height_mm * self.shoulder_height_ratio,
        )
        target_rotation = _punch_rotation(
            reference_rotation,
            selected_type,
            self.hook_face_angle_deg,
        )
        target_angles = (
            reference_angles
            if selected_type is PunchType.STRAIGHT
            else _zyz_degrees_near(target_rotation, reference_angles)
        )
        face_normal = tuple(target_rotation[row][2] for row in range(3))
        return MittPosePlan(
            tcp_pose_mm_deg=(*target_position, *target_angles),
            strike_range_mm=strike_range,
            target_height_mm=target_position[2],
            person_distance_mm=reference_strike,
            face_normal_base=face_normal,  # type: ignore[arg-type]
        )


def _punch_rotation(
    reference: Matrix3,
    punch_type: PunchType,
    hook_face_angle_deg: float,
) -> Matrix3:
    if punch_type is PunchType.STRAIGHT:
        return reference
    if punch_type in {PunchType.LEFT_HOOK, PunchType.RIGHT_HOOK}:
        # The punch side names describe the boxer's hand, viewed from the
        # boxer.  The previous signs were mirrored from the robot viewpoint.
        sign = -1.0 if punch_type is PunchType.LEFT_HOOK else 1.0
        turn = _axis_angle_rotation((0.0, 0.0, 1.0), sign * hook_face_angle_deg)
        return normalize_rotation_matrix(_matrix_multiply(turn, reference))

    # The Tool +Z axis is the outward mitt-face normal.  An uppercut arrives
    # from below, so rotate the face normal from the boxer-facing horizontal
    # direction to BASE -Z while preserving the shortest physical rotation.
    reference_normal = tuple(reference[row][2] for row in range(3))
    target_normal = (0.0, 0.0, -1.0)
    axis = _cross(reference_normal, target_normal)
    dot = max(-1.0, min(1.0, _dot(reference_normal, target_normal)))
    angle_deg = math.degrees(math.acos(dot))
    turn = _axis_angle_rotation(_normalize(axis), angle_deg)
    return normalize_rotation_matrix(_matrix_multiply(turn, reference))


def _zyz_degrees_near(rotation: Matrix3, reference: Sequence[float]) -> Vector3:
    matrix = normalize_rotation_matrix(rotation)
    beta = math.acos(max(-1.0, min(1.0, matrix[2][2])))
    if abs(math.sin(beta)) > 1e-9:
        alpha = math.atan2(matrix[1][2], matrix[0][2])
        gamma = math.atan2(matrix[2][1], -matrix[2][0])
    else:
        alpha = math.atan2(matrix[1][0], matrix[0][0])
        gamma = 0.0
    canonical = tuple(math.degrees(value) for value in (alpha, beta, gamma))
    alternate = (canonical[0] + 180.0, -canonical[1], canonical[2] + 180.0)
    reference_angles = tuple(float(value) for value in reference)
    candidates = []
    for base in (canonical, alternate):
        candidate = tuple(
            value + 360.0 * round((reference_angles[index] - value) / 360.0)
            for index, value in enumerate(base)
        )
        candidates.append(candidate)
    return min(
        candidates,
        key=lambda values: sum(
            (values[index] - reference_angles[index]) ** 2 for index in range(3)
        ),
    )  # type: ignore[return-value]


def _axis_angle_rotation(axis: Vector3, angle_deg: float) -> Matrix3:
    x, y, z = _normalize(axis)
    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return (
        (cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine),
        (y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine),
        (z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement),
    )


def _matrix_multiply(first: Matrix3, second: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(first[row][index] * second[index][column] for index in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
def _pose6(values: Sequence[float]) -> Pose6:
    pose = tuple(float(value) for value in values)
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise ValueError("reference TCP pose must contain six finite values")
    return pose  # type: ignore[return-value]
def _normalize(vector: Vector3) -> Vector3:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]

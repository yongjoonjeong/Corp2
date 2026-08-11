"""Wrench zero calibration without any ROS dependency."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
import tempfile
from typing import Iterable, Sequence

import yaml


WRENCH_KEYS = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")


@dataclass(frozen=True)
class WrenchZeroCalibration:
    created_at: str
    pose_name: str
    sample_count: int
    offset: tuple[float, float, float, float, float, float]
    stddev: tuple[float, float, float, float, float, float]


class ForceCalibrationError(ValueError):
    """Raised when wrench samples are unsuitable for zero calibration."""


class ForceCalibrator:
    def __init__(
        self,
        *,
        maximum_force_stddev_n: float = 0.5,
        maximum_moment_stddev_nm: float = 0.05,
        minimum_samples: int = 10,
    ) -> None:
        if not math.isfinite(maximum_force_stddev_n) or maximum_force_stddev_n <= 0:
            raise ValueError("maximum_force_stddev_n must be finite and positive")
        if (
            not math.isfinite(maximum_moment_stddev_nm)
            or maximum_moment_stddev_nm <= 0
        ):
            raise ValueError("maximum_moment_stddev_nm must be finite and positive")
        if minimum_samples <= 1:
            raise ValueError("minimum_samples must exceed one")
        self.maximum_force_stddev_n = maximum_force_stddev_n
        self.maximum_moment_stddev_nm = maximum_moment_stddev_nm
        self.minimum_samples = minimum_samples

    def calculate(
        self,
        samples: Iterable[Sequence[float]],
        *,
        pose_name: str = "MOCK",
        created_at: str | None = None,
    ) -> WrenchZeroCalibration:
        normalized = [self._normalize_wrench(sample) for sample in samples]
        if len(normalized) < self.minimum_samples:
            raise ForceCalibrationError(
                f"at least {self.minimum_samples} samples are required"
            )

        axes = list(zip(*normalized))
        offset = tuple(fmean(axis) for axis in axes)
        stddev = tuple(pstdev(axis) for axis in axes)
        if any(value > self.maximum_force_stddev_n for value in stddev[:3]):
            force_detail = ", ".join(
                f"{axis}={value:.6f}"
                for axis, value in zip(("Fx", "Fy", "Fz"), stddev[:3])
            )
            raise ForceCalibrationError(
                "force samples are not stable: "
                f"stddev [{force_detail}] N; "
                f"limit={self.maximum_force_stddev_n:.6f} N"
            )
        if any(value > self.maximum_moment_stddev_nm for value in stddev[3:]):
            moment_detail = ", ".join(
                f"{axis}={value:.6f}"
                for axis, value in zip(("Mx", "My", "Mz"), stddev[3:])
            )
            raise ForceCalibrationError(
                "moment samples are not stable: "
                f"stddev [{moment_detail}] Nm; "
                f"limit={self.maximum_moment_stddev_nm:.6f} Nm"
            )

        timestamp = created_at or datetime.now(timezone.utc).astimezone().isoformat()
        return WrenchZeroCalibration(
            created_at=timestamp,
            pose_name=pose_name.strip() or "UNKNOWN",
            sample_count=len(normalized),
            offset=offset,  # type: ignore[arg-type]
            stddev=stddev,  # type: ignore[arg-type]
        )

    @staticmethod
    def apply(
        wrench: Sequence[float], calibration: WrenchZeroCalibration
    ) -> tuple[float, float, float, float, float, float]:
        normalized = ForceCalibrator._normalize_wrench(wrench)
        return tuple(
            value - offset for value, offset in zip(normalized, calibration.offset)
        )  # type: ignore[return-value]

    @staticmethod
    def save(calibration: WrenchZeroCalibration, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "created_at": calibration.created_at,
            "pose_name": calibration.pose_name,
            "sample_count": calibration.sample_count,
            "offset": dict(zip(WRENCH_KEYS, calibration.offset)),
            "stddev": dict(zip(WRENCH_KEYS, calibration.stddev)),
        }

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                yaml.safe_dump(document, temporary, sort_keys=False)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def load(path: str | Path) -> WrenchZeroCalibration:
        source = Path(path).expanduser()
        with source.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ForceCalibrationError("calibration YAML must contain a mapping")
        try:
            offset_map = document["offset"]
            stddev_map = document["stddev"]
            calibration = WrenchZeroCalibration(
                created_at=str(document["created_at"]),
                pose_name=str(document.get("pose_name", "UNKNOWN")),
                sample_count=int(document["sample_count"]),
                offset=tuple(float(offset_map[key]) for key in WRENCH_KEYS),
                stddev=tuple(float(stddev_map[key]) for key in WRENCH_KEYS),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ForceCalibrationError(f"invalid calibration YAML: {error}") from error
        ForceCalibrator._normalize_wrench(calibration.offset)
        ForceCalibrator._normalize_wrench(calibration.stddev)
        if calibration.sample_count <= 0:
            raise ForceCalibrationError("sample_count must be positive")
        return calibration

    @staticmethod
    def _normalize_wrench(
        wrench: Sequence[float],
    ) -> tuple[float, float, float, float, float, float]:
        if len(wrench) != 6:
            raise ForceCalibrationError("wrench must contain six values")
        values = tuple(float(value) for value in wrench)
        if not all(math.isfinite(value) for value in values):
            raise ForceCalibrationError("wrench values must be finite")
        return values  # type: ignore[return-value]

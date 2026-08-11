"""Failure-isolated JSON summary and optional CSV wrench logging."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from mitt_hit_system.impact_buffer import BufferedWrenchSample
from mitt_hit_system.return_to_reference import ReturnObservation


class HitRecordLogger:
    CSV_FIELDS = (
        "timestamp_ns",
        "raw_fx_n",
        "raw_fy_n",
        "raw_fz_n",
        "raw_mx_nm",
        "raw_my_nm",
        "raw_mz_nm",
        "filtered_fx_n",
        "filtered_fy_n",
        "filtered_fz_n",
        "filtered_mx_nm",
        "filtered_my_nm",
        "filtered_mz_nm",
    )
    RETURN_CSV_FIELDS = (
        "timestamp_ns",
        "tcp_x_mm",
        "tcp_y_mm",
        "tcp_z_mm",
        "tcp_vx_mm_s",
        "tcp_vy_mm_s",
        "tcp_vz_mm_s",
        "displacement_mm",
        "translation_speed_mm_s",
        "normal_force_n",
        "robot_state",
    )

    def __init__(self, record_dir: str | Path, *, save_raw_hit_data: bool = True) -> None:
        self.record_dir = Path(record_dir).expanduser()
        self.save_raw_hit_data = save_raw_hit_data
        self.session_id = ""
        self._document: dict[str, Any] = {}

    def start_session(
        self,
        *,
        mode: str = "MANUAL",
        pose_name: str = "MOCK",
        session_id: str | None = None,
    ) -> str:
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._document = {
            "session_id": self.session_id,
            "mode": mode,
            "pose_name": pose_name,
            "hits": [],
        }
        return self.session_id

    def log_hit(
        self,
        result: dict[str, Any],
        samples: Iterable[BufferedWrenchSample],
    ) -> tuple[bool, str]:
        """Persist one hit without propagating storage failures to control code."""
        try:
            if not self.session_id:
                raise RuntimeError("session has not started")
            self.record_dir.mkdir(parents=True, exist_ok=True)
            hit = dict(result)
            self._document["hits"].append(hit)
            self._write_json_atomic()
            if self.save_raw_hit_data:
                self._write_csv(int(hit["hit_id"]), tuple(samples))
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            return False, str(error)
        return True, ""

    def log_return(
        self,
        hit_id: int,
        result: dict[str, Any],
        samples: Iterable[ReturnObservation],
    ) -> tuple[bool, str]:
        """Attach return outcome to a hit and persist its bounded RT trace."""
        try:
            if not self.session_id:
                raise RuntimeError("session has not started")
            hit = next(
                item
                for item in self._document["hits"]
                if int(item["hit_id"]) == int(hit_id)
            )
            observations = tuple(samples)
            hit["return_to_reference"] = dict(result)
            self._write_json_atomic()
            if self.save_raw_hit_data:
                self._write_return_csv(int(hit_id), observations)
        except (OSError, RuntimeError, StopIteration, TypeError, ValueError, KeyError) as error:
            return False, str(error)
        return True, ""

    def log_compliance_baseline(
        self, baseline: dict[str, Any]
    ) -> tuple[bool, str]:
        """Persist the frozen post-activation TCP/wrench session reference."""
        try:
            if not self.session_id:
                raise RuntimeError("session has not started")
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._document["compliance_baseline"] = dict(baseline)
            self._write_json_atomic()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return False, str(error)
        return True, ""

    def log_compliance_summary(
        self, summary: dict[str, Any]
    ) -> tuple[bool, str]:
        """Persist session-wide compliance extrema and terminal outcome."""
        try:
            if not self.session_id:
                raise RuntimeError("session has not started")
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._document["compliance_summary"] = dict(summary)
            self._write_json_atomic()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return False, str(error)
        return True, ""

    def end_session(self) -> None:
        """Make accidental writes after Stop impossible."""
        self.session_id = ""
        self._document = {}

    @property
    def json_path(self) -> Path:
        if not self.session_id:
            raise RuntimeError("session has not started")
        return self.record_dir / f"{self.session_id}_session.json"

    def _write_json_atomic(self) -> None:
        destination = self.json_path
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.record_dir,
                prefix=destination.name + ".",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(self._document, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _write_csv(
        self, hit_id: int, samples: tuple[BufferedWrenchSample, ...]
    ) -> None:
        destination = self.record_dir / f"{self.session_id}_hit_{hit_id:04d}.csv"
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.CSV_FIELDS)
            for sample in samples:
                writer.writerow(
                    (sample.timestamp_ns, *sample.raw_wrench, *sample.filtered_wrench)
                )

    def _write_return_csv(
        self, hit_id: int, samples: tuple[ReturnObservation, ...]
    ) -> None:
        destination = (
            self.record_dir / f"{self.session_id}_hit_{hit_id:04d}_return.csv"
        )
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.RETURN_CSV_FIELDS)
            for sample in samples:
                writer.writerow(
                    (
                        sample.timestamp_ns,
                        *sample.tcp_position_mm,
                        *sample.tcp_velocity_mm_s,
                        sample.displacement_mm,
                        sample.translation_speed_mm_s,
                        sample.normal_force_n,
                        sample.robot_state,
                    )
                )

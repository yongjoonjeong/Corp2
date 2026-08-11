import csv
import json
from pathlib import Path

from mitt_hit_system.hit_record_logger import HitRecordLogger
from mitt_hit_system.impact_buffer import BufferedWrenchSample
from mitt_hit_system.return_to_reference import ReturnObservation


def test_json_and_csv_are_written(tmp_path: Path) -> None:
    logger = HitRecordLogger(tmp_path / "records", save_raw_hit_data=True)
    logger.start_session(session_id="test_session")
    wrench = (0.0, 0.0, 18.0, 0.0, -0.9, 0.0)
    samples = [BufferedWrenchSample(123, wrench, wrench)]

    success, error = logger.log_hit(
        {
            "hit_id": 1,
            "valid_hit": True,
            "hit_direction": "RIGHT",
            "hit_x_mm": 50.0,
            "total_score": 6.3,
        },
        samples,
    )

    assert success, error
    with logger.json_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    assert document["hits"][0]["hit_direction"] == "RIGHT"

    csv_path = tmp_path / "records" / "test_session_hit_0001.csv"
    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][0] == "timestamp_ns"
    assert rows[1][0] == "123"


def test_storage_failure_is_returned_not_raised(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("occupied", encoding="utf-8")
    logger = HitRecordLogger(not_a_directory)
    logger.start_session(session_id="failure")

    success, error = logger.log_hit({"hit_id": 1}, [])

    assert not success
    assert error


def test_return_summary_and_synchronized_trace_are_written(tmp_path: Path) -> None:
    logger = HitRecordLogger(tmp_path / "records", save_raw_hit_data=True)
    logger.start_session(session_id="return_session")
    assert logger.log_hit({"hit_id": 1, "valid_hit": True}, [])[0]
    observation = ReturnObservation(
        timestamp_ns=456,
        tcp_position_mm=(1.0, 2.0, 3.0),
        tcp_velocity_mm_s=(0.1, 0.2, 0.3),
        displacement_mm=0.4,
        translation_speed_mm_s=0.5,
        normal_force_n=0.6,
        robot_state=1,
    )

    success, error = logger.log_return(
        1,
        {"outcome": "SETTLED", "sample_count": 1},
        [observation],
    )

    assert success, error
    with logger.json_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    assert document["hits"][0]["return_to_reference"]["outcome"] == "SETTLED"

    csv_path = tmp_path / "records" / "return_session_hit_0001_return.csv"
    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][-1] == "robot_state"
    assert rows[1] == [
        "456",
        "1.0",
        "2.0",
        "3.0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "1",
    ]


def test_compliance_baseline_is_saved_before_the_first_hit(tmp_path: Path) -> None:
    logger = HitRecordLogger(tmp_path)
    logger.start_session(session_id="baseline")

    success, message = logger.log_compliance_baseline(
        {
            "tcp_reference_mm": [1.0, 2.0, 3.0],
            "wrench_offset": [4.0, 5.0, 6.0, 0.1, 0.2, 0.3],
            "wrench_stddev": [0.1] * 6,
        }
    )

    assert success, message
    document = json.loads((tmp_path / "baseline_session.json").read_text())
    assert document["compliance_baseline"]["tcp_reference_mm"] == [
        1.0,
        2.0,
        3.0,
    ]
    assert document["hits"] == []


def test_compliance_session_summary_is_persisted_before_end(tmp_path: Path) -> None:
    logger = HitRecordLogger(tmp_path)
    logger.start_session(session_id="summary")

    success, message = logger.log_compliance_summary(
        {
            "outcome": "TEST_COMPLETE",
            "hit_count": 3,
            "maximum_tcp_displacement_mm": 0.55,
            "maximum_tcp_angular_displacement_deg": 0.04,
            "maximum_total_force_n": 13.7,
            "maximum_total_torque_nm": 0.4,
        }
    )

    assert success, message
    document = json.loads((tmp_path / "summary_session.json").read_text())
    assert document["compliance_summary"]["outcome"] == "TEST_COMPLETE"
    assert document["compliance_summary"]["hit_count"] == 3
    assert document["compliance_summary"][
        "maximum_tcp_angular_displacement_deg"
    ] == 0.04

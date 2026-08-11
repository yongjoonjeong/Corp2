from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import mimetypes
import os
import re
import sqlite3
import threading
import time
from collections import deque
from email.parser import BytesParser
from email.policy import default as email_policy
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from reporting import METRICS, build_progress, compact_event, derive_metrics, event_score, infer_tracked_metric, issue_tags, select_best_worst
from vision_coach import (
    DEFAULT_VISION_COACH_MODEL,
    VisionCoachError,
    analyze_boxing_images,
    select_representative_images,
)
from voice_processing.openai_transcriber import TranscriptionError, transcribe_audio_bytes
from voice_processing.wakeword_service import WakeWordService

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIR / "instance" / "ko.sqlite3"
INDEX_FILE = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"
EVIDENCE_DIR = BASE_DIR / "instance" / "evidence"
ENV_FILE = BASE_DIR / ".env"
MAX_AUDIO_BYTES = 15 * 1024 * 1024
WAKEWORD_MODEL_DEFAULT = "wake_up_ko.tflite"


def normalize_app_mode(value: str | None) -> str:
    mode = str(value or "user").strip().lower().replace("-", "_")
    if mode in {"admin", "admin_mode"}:
        return "admin"
    return "user"


def load_local_env(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs without exposing them to the browser."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


load_local_env()

WAKEWORD_MODEL_FILE = (os.environ.get("WAKEWORD_MODEL_FILE", WAKEWORD_MODEL_DEFAULT).strip() or WAKEWORD_MODEL_DEFAULT)
WAKEWORD_MODEL = BASE_DIR / "voice_processing" / Path(WAKEWORD_MODEL_FILE).name
WAKEWORD_DISPLAY_NAME = (os.environ.get("WAKEWORD_DISPLAY_NAME", "웨이크 업 케이오").strip() or "웨이크 업 케이오")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                height_cm REAL NOT NULL CHECK(height_cm BETWEEN 100 AND 230),
                dominant_hand TEXT NOT NULL CHECK(dominant_hand IN ('right', 'left')),
                wingspan_cm REAL,
                left_punch_reach_cm REAL,
                right_punch_reach_cm REAL,
                recommended_distance_cm REAL,
                measurement_confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_training_at TEXT
            );

            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                training_type TEXT NOT NULL,
                hand TEXT NOT NULL CHECK(hand IN ('right', 'left', 'both')),
                duration_sec INTEGER NOT NULL,
                punch_count INTEGER NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 0,
                avg_reaction_ms REAL,
                posture_score REAL,
                feedback TEXT,
                client_session_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_mitt_calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                punch_role TEXT NOT NULL CHECK(punch_role IN ('jab', 'straight')),
                hand TEXT NOT NULL CHECK(hand IN ('left', 'right')),
                correction_x_mm REAL NOT NULL,
                correction_y_mm REAL NOT NULL,
                raw_center_x_mm REAL NOT NULL,
                raw_center_y_mm REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                accepted_sample_count INTEGER NOT NULL,
                dispersion_mm REAL NOT NULL,
                correction_limited INTEGER NOT NULL DEFAULT 0,
                base_pose_json TEXT,
                calibrated_pose_json TEXT,
                raw_samples_json TEXT NOT NULL,
                vision_summary_json TEXT,
                calibration_version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, punch_role)
            );

            CREATE TABLE IF NOT EXISTS user_reach_calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                hand TEXT NOT NULL CHECK(hand IN ('left', 'right')),
                correction_z_mm REAL NOT NULL CHECK(correction_z_mm >= 0),
                baseline_normal_force_n REAL NOT NULL,
                contact_delta_force_n REAL NOT NULL,
                base_pose_json TEXT NOT NULL,
                contact_pose_json TEXT NOT NULL,
                calibration_version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vision_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
                total_punches INTEGER,
                successful_punches INTEGER,
                accuracy_percent REAL,
                average_reaction_sec REAL,
                average_guard_return_sec REAL,
                guard_drop_count INTEGER,
                slow_guard_return_count INTEGER,
                arm_extension_score REAL,
                guard_score REAL,
                torso_balance_score REAL,
                representative_images_json TEXT,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE REFERENCES training_sessions(id) ON DELETE CASCADE,
                summary TEXT,
                strengths_json TEXT,
                improvements_json TEXT,
                next_training TEXT,
                coach_message TEXT,
                model TEXT,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS force_hit_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES training_sessions(id) ON DELETE CASCADE,
                received_version INTEGER NOT NULL,
                hit_id INTEGER,
                stamp_ns INTEGER,
                valid_hit INTEGER,
                invalid_reason TEXT,
                hit_direction TEXT,
                hit_x_mm REAL,
                hit_y_mm REAL,
                center_error_mm REAL,
                peak_force_n REAL,
                peak_normal_force_n REAL,
                impulse_ns REAL,
                contact_duration_ms REAL,
                accuracy_score REAL,
                power_score REAL,
                total_score REAL,
                force_warning INTEGER,
                safety_stop INTEGER,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, received_version)
            );

            CREATE TABLE IF NOT EXISTS punch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
                punch_index INTEGER NOT NULL,
                vision_punch_id INTEGER,
                punch_side TEXT,
                punch_type TEXT,
                event_score REAL NOT NULL,
                passed INTEGER,
                violations_json TEXT NOT NULL,
                issue_tags_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                force_result_id INTEGER REFERENCES force_hit_results(id) ON DELETE SET NULL,
                evidence_version INTEGER,
                evidence_path TEXT,
                is_best INTEGER NOT NULL DEFAULT 0,
                is_worst INTEGER NOT NULL DEFAULT 0,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, punch_index)
            );

            CREATE TABLE IF NOT EXISTS feedback_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source_session_id INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
                metric_key TEXT NOT NULL,
                metric_label TEXT NOT NULL,
                baseline_value REAL,
                advice TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(training_sessions)").fetchall()}
        if "client_session_id" not in columns:
            db.execute("ALTER TABLE training_sessions ADD COLUMN client_session_id TEXT")
        reach_table_sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'user_reach_calibrations'"
        ).fetchone()
        if reach_table_sql and "BETWEEN 0 AND 150" in str(reach_table_sql[0]).upper():
            db.executescript(
                """
                ALTER TABLE user_reach_calibrations
                    RENAME TO user_reach_calibrations_limited;
                CREATE TABLE user_reach_calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    hand TEXT NOT NULL CHECK(hand IN ('left', 'right')),
                    correction_z_mm REAL NOT NULL CHECK(correction_z_mm >= 0),
                    baseline_normal_force_n REAL NOT NULL,
                    contact_delta_force_n REAL NOT NULL,
                    base_pose_json TEXT NOT NULL,
                    contact_pose_json TEXT NOT NULL,
                    calibration_version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO user_reach_calibrations
                SELECT * FROM user_reach_calibrations_limited;
                DROP TABLE user_reach_calibrations_limited;
                """
            )
        db.commit()
    finally:
        db.close()


@contextmanager
def connect(database: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
    finally:
        db.close()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class EventBroker:
    def __init__(self, max_events: int = 100):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._next_id = 1
        self._lock = threading.Lock()

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            event = {
                "id": self._next_id,
                "type": event_type,
                "payload": payload or {},
                "created_at": utc_now(),
            }
            self._next_id += 1
            self._events.append(event)
            return dict(event)

    def after(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events if int(event["id"]) > event_id]



def normalize_voice_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(text or "").lower())


def is_training_voice_command(text: str) -> bool:
    normalized = normalize_voice_text(text)
    if not normalized:
        return False
    explicit = ("훈련시작", "운동시작", "연습할게", "훈련할게", "운동할게")
    intent_words = ("훈련", "운동", "연습", "시작")
    punch_words = ("스트레이트", "잽", "훅", "어퍼", "어퍼컷")
    has_punch = any(word in normalized for word in punch_words)
    has_intent = any(word in normalized for word in intent_words)
    has_duration = bool(re.search(r"\d+(?:분|초)", normalized))
    return any(word in normalized for word in explicit) or (has_punch and (has_intent or has_duration))


class RobotHub:
    """Browser-independent command queue between the UI server and ROS bridge."""

    def __init__(self) -> None:
        self.events = EventBroker(max_events=200)
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "connected": False,
            "state": "WAITING_BRIDGE",
            "message": "ROS 브리지 연결 대기",
            "last_seen_at": None,
            "last_command": None,
        }
        self._last_command_key = ""
        self._last_command_at = 0.0

    def enqueue(self, command: str, *, source: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        command = str(command).strip()
        now = time.monotonic()
        dedupe_key = f"{command}:{source}"
        with self._lock:
            if dedupe_key == self._last_command_key and now - self._last_command_at < 0.8:
                return {"accepted": True, "deduplicated": True, "command": command}
            self._last_command_key = dedupe_key
            self._last_command_at = now
            self._status["last_command"] = command
        event_payload = {"command": command, "source": source}
        if payload:
            event_payload["payload"] = payload
        event = self.events.publish("robot_command", event_payload)
        return {"accepted": True, "queued": True, "command": command, "event_id": event["id"]}

    def commands_after(self, event_id: int) -> list[dict[str, Any]]:
        return self.events.after(event_id)

    def update_status(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._status.update(payload)
            self._status["connected"] = True
            self._status["last_seen_at"] = utc_now()

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
        return result


class VisionHub:
    def __init__(self) -> None:
        self.event_broker = EventBroker(max_events=500)
        self._lock = threading.Lock()
        self._last_seen_monotonic = 0.0
        self._last_seen_at: str | None = None
        self._preview: bytes | None = None
        self._preview_type = "image/jpeg"
        self._preview_version = 0
        self._front: bytes | None = None
        self._front_type = "image/jpeg"
        self._front_version = 0
        self._evidence: bytes | None = None
        self._evidence_type = "image/jpeg"
        self._evidence_version = 0
        self._evidence_history: deque[tuple[int, bytes, str]] = deque(maxlen=60)
        self._evidence_meta: dict[int, dict[str, Any]] = {}
        self._last_punch: dict[str, Any] | None = None
        self._heartbeat: dict[str, Any] = {}
        self._live_status: dict[str, Any] = {}

    def touch(self) -> None:
        with self._lock:
            self._last_seen_monotonic = __import__("time").monotonic()
            self._last_seen_at = utc_now()

    def heartbeat(self, payload: dict[str, Any]) -> None:
        self.touch()
        with self._lock:
            self._heartbeat = dict(payload)

    def update_status(self, payload: dict[str, Any]) -> None:
        self.touch()
        with self._lock:
            self._live_status = dict(payload)

    def publish_punch(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.touch()
        with self._lock:
            self._last_punch = dict(payload)
        return self.event_broker.publish("punch", payload)

    def set_image(self, kind: str, data: bytes, content_type: str, metadata: dict[str, Any] | None = None) -> int:
        self.touch()
        with self._lock:
            if kind == "preview":
                self._preview = data
                self._preview_type = content_type
                self._preview_version += 1
                return self._preview_version
            if kind == "front":
                self._front = data
                self._front_type = content_type
                self._front_version += 1
                return self._front_version
            if kind == "evidence":
                self._evidence = data
                self._evidence_type = content_type
                self._evidence_version += 1
                self._evidence_history.append((self._evidence_version, bytes(data), content_type))
                self._evidence_meta[self._evidence_version] = dict(metadata or {})
                oldest = self._evidence_version - 80
                for key in [key for key in self._evidence_meta if key < oldest]:
                    self._evidence_meta.pop(key, None)
                return self._evidence_version
            raise ValueError(f"알 수 없는 비전 이미지 종류입니다: {kind}")

    def image(self, kind: str) -> tuple[bytes | None, str, int]:
        with self._lock:
            if kind == "preview":
                return self._preview, self._preview_type, self._preview_version
            if kind == "front":
                return self._front, self._front_type, self._front_version
            if kind == "evidence":
                return self._evidence, self._evidence_type, self._evidence_version
            raise ValueError(f"알 수 없는 비전 이미지 종류입니다: {kind}")

    def evidence_after(self, version: int) -> list[tuple[int, bytes, str]]:
        with self._lock:
            return [item for item in self._evidence_history if item[0] > version]

    def evidence_records_after(self, version: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"version": item[0], "data": bytes(item[1]), "content_type": item[2], "metadata": dict(self._evidence_meta.get(item[0], {}))}
                for item in self._evidence_history if item[0] > version
            ]

    def status(self) -> dict[str, Any]:
        import time
        with self._lock:
            age = None
            if self._last_seen_monotonic:
                age = max(0.0, time.monotonic() - self._last_seen_monotonic)
            return {
                "connected": age is not None and age < 5.0,
                "last_seen_at": self._last_seen_at,
                "age_sec": round(age, 2) if age is not None else None,
                "preview_available": self._preview is not None,
                "preview_version": self._preview_version,
                "front_available": self._front is not None,
                "front_version": self._front_version,
                "evidence_available": self._evidence is not None,
                "evidence_version": self._evidence_version,
                "last_punch": dict(self._last_punch) if self._last_punch else None,
                "heartbeat": dict(self._heartbeat),
                "live_status": dict(self._live_status),
            }


class ForceHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=500)
        self._last_seen_at: str | None = None
        self._dedupe: dict[tuple[str, int, int], int] = {}

    @staticmethod
    def _key(payload: dict[str, Any]) -> tuple[str, int, int] | None:
        session = str(payload.get("client_session_id", "")).strip()
        hit_id = payload.get("hit_id")
        stamp_ns = payload.get("stamp_ns")
        if not session or hit_id is None or stamp_ns is None:
            return None
        try:
            return session, int(hit_id), int(stamp_ns)
        except (TypeError, ValueError):
            return None

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            key = self._key(payload)
            if key is not None:
                existing_version = self._dedupe.get(key)
                if existing_version is not None:
                    for item in reversed(self._history):
                        if int(item["version"]) == existing_version:
                            return dict(item)
            self._version += 1
            item = {"version": self._version, "payload": dict(payload), "received_at": utc_now()}
            self._history.append(item)
            if key is not None:
                self._dedupe[key] = self._version
                valid_versions = {int(item["version"]) for item in self._history}
                self._dedupe = {k: v for k, v in self._dedupe.items() if v in valid_versions}
            self._last_seen_at = item["received_at"]
            return dict(item)

    def after(self, version: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._history if int(item["version"]) > int(version)]

    def status(self) -> dict[str, Any]:
        with self._lock:
            last_payload = dict(self._history[-1]["payload"]) if self._history else None
            return {
                "available": bool(self._history),
                "version": self._version,
                "last_seen_at": self._last_seen_at,
                "last_hit": last_payload,
            }


class KoRequestHandler(BaseHTTPRequestHandler):
    server_version = "KOUI/1.0"

    @property
    def database(self) -> Path:
        return self.server.database  # type: ignore[attr-defined]

    @property
    def wakeword_service(self) -> WakeWordService | None:
        return self.server.wakeword_service  # type: ignore[attr-defined]

    @property
    def force_hub(self) -> ForceHub:
        return self.server.force_hub  # type: ignore[attr-defined]

    @property
    def event_broker(self) -> EventBroker:
        return self.server.event_broker  # type: ignore[attr-defined]

    @property
    def vision_hub(self) -> VisionHub:
        return self.server.vision_hub  # type: ignore[attr-defined]

    @property
    def robot_hub(self) -> RobotHub:
        return self.server.robot_hub  # type: ignore[attr-defined]

    @property
    def app_mode(self) -> str:
        return self.server.app_mode  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, content_type: str, *, cache_control: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8" if (content_type or "").startswith("text/") else content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > 2 * 1024 * 1024:
            raise ValueError("요청 데이터가 너무 큽니다.")
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON 형식이 올바르지 않습니다.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON 객체가 필요합니다.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self.send_file(INDEX_FILE)
            return
        if path.startswith("/static/"):
            relative = Path(path.removeprefix("/static/"))
            requested = (STATIC_DIR / relative).resolve()
            try:
                requested.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.send_file(requested)
            return
        if path == "/api/app/config":
            supported_punches = [
                value.strip().lower()
                for value in os.environ.get(
                    "KO_ROBOT_SUPPORTED_PUNCHES",
                    "jab,straight,hook,uppercut",
                ).split(",")
                if value.strip().lower() in {"jab", "straight", "hook", "uppercut"}
            ]
            self.send_json({
                "mode": self.app_mode,
                "is_admin": self.app_mode == "admin",
                "show_system_settings": self.app_mode == "admin",
                "show_vision_debug": self.app_mode == "admin",
                "voice_activation_policy": "one_command_per_wake",
                "robot_supported_punches": supported_punches,
            })
            return
        if path == "/api/health":
            self.send_json({"ok": True, "service": "ko-boxing-ui", "time": utc_now(), "mode": self.app_mode})
            return
        if path == "/api/database/status":
            self.database_status()
            return
        if path == "/api/stt/status":
            self.send_json({
                "configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                "model": os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
                "provider": "OpenAI Audio Transcriptions",
            })
            return
        if path == "/api/ai/vision-coach/status":
            self.send_json({
                "configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                "model": os.environ.get("OPENAI_VISION_COACH_MODEL", DEFAULT_VISION_COACH_MODEL),
                "provider": "OpenAI Responses API",
            })
            return
        if path == "/api/wakeword/status":
            status = self.wakeword_service.status() if self.wakeword_service else {
                "available": False,
                "running": False,
                "enabled": False,
                "state": "disabled",
                "message": "웨이크업 서비스가 시작되지 않았습니다.",
                "model": WAKEWORD_MODEL.name,
                "display_name": WAKEWORD_DISPLAY_NAME,
            }
            status["stt_configured"] = bool(os.environ.get("OPENAI_API_KEY", "").strip())
            self.send_json(status)
            return
        if path == "/api/wakeword/events":
            query = parse_qs(urlparse(self.path).query)
            try:
                after_id = max(0, int(query.get("after", ["0"])[0]))
            except (TypeError, ValueError):
                after_id = 0
            self.send_json({"events": self.event_broker.after(after_id)})
            return
        if path == "/api/vision/status":
            self.send_json(self.vision_hub.status())
            return
        if path == "/api/vision/events":
            query = parse_qs(urlparse(self.path).query)
            try:
                after_id = max(0, int(query.get("after", ["0"])[0]))
            except (TypeError, ValueError):
                after_id = 0
            self.send_json({"events": self.vision_hub.event_broker.after(after_id)})
            return
        if path == "/api/vision/preview.jpg":
            data, content_type, _ = self.vision_hub.image("preview")
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(data, content_type)
            return
        if path == "/api/vision/front.jpg":
            data, content_type, _ = self.vision_hub.image("front")
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(data, content_type)
            return
        if path == "/api/vision/evidence.jpg":
            data, content_type, _ = self.vision_hub.image("evidence")
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(data, content_type)
            return
        if path == "/api/robot/status":
            self.send_json(self.robot_hub.status())
            return
        if path == "/api/robot/commands":
            query = parse_qs(urlparse(self.path).query)
            try:
                after_id = max(0, int(query.get("after", ["0"])[0]))
            except (TypeError, ValueError):
                after_id = 0
            self.send_json({"events": self.robot_hub.commands_after(after_id)})
            return
        if path == "/api/users":
            self.list_users()
            return
        user_match = re.fullmatch(r"/api/users/(\d+)", path)
        if user_match:
            self.get_user(int(user_match.group(1)))
            return
        mitt_calibration_match = re.fullmatch(r"/api/users/(\d+)/mitt-calibrations", path)
        if mitt_calibration_match:
            self.list_mitt_calibrations(int(mitt_calibration_match.group(1)))
            return
        reach_calibration_match = re.fullmatch(r"/api/users/(\d+)/reach-calibration", path)
        if reach_calibration_match:
            self.get_reach_calibration(int(reach_calibration_match.group(1)))
            return
        if path == "/api/force/status":
            self.send_json(self.force_hub.status())
            return
        evidence_match = re.fullmatch(r"/api/punch-events/(\d+)/evidence\.jpg", path)
        if evidence_match:
            self.serve_punch_evidence(int(evidence_match.group(1)))
            return
        sessions_match = re.fullmatch(r"/api/users/(\d+)/sessions", path)
        if sessions_match:
            self.list_sessions(int(sessions_match.group(1)))
            return
        detail_match = re.fullmatch(r"/api/sessions/(\d+)/details", path)
        if detail_match:
            self.session_details(int(detail_match.group(1)))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/transcribe":
                self.transcribe_audio()
                return
            payload = self.read_json()
            if path == "/api/wakeword/control":
                self.control_wakeword(payload)
                return
            if path == "/api/wakeword/suppress":
                self.suppress_wakeword(payload)
                return
            if path == "/api/wakeword/session":
                self.control_voice_session(payload)
                return
            if path == "/api/vision/heartbeat":
                self.vision_hub.heartbeat(payload)
                self.send_json({"ok": True})
                return
            if path == "/api/vision/status_update":
                self.vision_hub.update_status(payload)
                self.send_json({"ok": True})
                return
            if path == "/api/vision/punch":
                self.receive_vision_punch(payload)
                return
            if path == "/api/vision/preview":
                self.receive_vision_image(payload, "preview")
                return
            if path == "/api/vision/front":
                self.receive_vision_image(payload, "front")
                return
            if path == "/api/vision/evidence":
                self.receive_vision_image(payload, "evidence")
                return
            if path == "/api/force/hit":
                self.receive_force_hit(payload)
                return
            if path == "/api/punch-events/sessionize":
                self.sessionize_punch_events(payload)
                return
            if path == "/api/users":
                self.create_user(payload)
                return
            mitt_calibration_match = re.fullmatch(r"/api/users/(\d+)/mitt-calibrations", path)
            if mitt_calibration_match:
                self.save_mitt_calibration(int(mitt_calibration_match.group(1)), payload)
                return
            reach_calibration_match = re.fullmatch(r"/api/users/(\d+)/reach-calibration", path)
            if reach_calibration_match:
                self.save_reach_calibration(int(reach_calibration_match.group(1)), payload)
                return
            if path == "/api/sessions":
                self.save_session(payload)
                return
            if path == "/api/vision/results":
                self.save_vision_result(payload)
                return
            if path == "/api/ai/vision-coach":
                self.generate_vision_coach(payload)
                return
            if path == "/api/ai/reports":
                self.save_ai_report(payload)
                return
            if path == "/api/robot/status_update":
                self.robot_hub.update_status(payload)
                self.send_json({"ok": True})
                return
            if path == "/api/robot/command":
                self.robot_command(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except sqlite3.IntegrityError:
            self.send_json({"error": "저장 값의 형식이나 범위를 확인해 주세요."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - final safety boundary
            print("Unhandled POST error:", repr(exc))
            self.send_json({"error": "서버에서 요청을 처리하지 못했습니다."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PATCH(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/users/(\d+)/measurement", path)
        if not match:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.save_measurement(int(match.group(1)), self.read_json())
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except sqlite3.IntegrityError:
            self.send_json({"error": "측정값 범위를 확인해 주세요."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            print("Unhandled PATCH error:", repr(exc))
            self.send_json({"error": "서버에서 요청을 처리하지 못했습니다."}, HTTPStatus.INTERNAL_SERVER_ERROR)


    def read_audio_upload(self) -> tuple[bytes, str, str]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("음성 파일은 multipart/form-data 형식으로 전송해 주세요.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("음성 파일 크기를 확인할 수 없습니다.") from exc
        if length <= 0:
            raise ValueError("녹음된 음성 파일이 없습니다.")
        if length > MAX_AUDIO_BYTES:
            raise ValueError("음성 파일은 15MB 이하만 사용할 수 있습니다.")

        raw = self.rfile.read(length)
        message = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8")
            + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
        )
        if not message.is_multipart():
            raise ValueError("음성 업로드 형식을 해석하지 못했습니다.")

        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            if part.get_param("name", header="content-disposition") != "audio":
                continue
            data = part.get_payload(decode=True) or b""
            if len(data) < 500:
                raise ValueError("녹음이 너무 짧습니다. 다시 말해 주세요.")
            filename = Path(part.get_filename() or "ko-command.webm").name
            media_type = part.get_content_type() or "application/octet-stream"
            return data, filename, media_type
        raise ValueError("audio 필드에서 음성 파일을 찾지 못했습니다.")

    def transcribe_audio(self) -> None:
        try:
            audio_bytes, filename, media_type = self.read_audio_upload()
            self.send_json(transcribe_audio_bytes(audio_bytes, filename, media_type))
        except TranscriptionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)

    def receive_vision_punch(self, payload: dict[str, Any]) -> None:
        required = ("punch_id", "punch_type", "punch_side", "total_score")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"비전 펀치 데이터 필드가 없습니다: {', '.join(missing)}")
        event = self.vision_hub.publish_punch(payload)
        self.send_json({"ok": True, "event_id": event["id"]}, HTTPStatus.CREATED)

    def receive_vision_image(self, payload: dict[str, Any], kind: str) -> None:
        raw = payload.get("data_base64")
        if not isinstance(raw, str) or not raw:
            raise ValueError("비전 이미지 data_base64가 필요합니다.")
        try:
            data = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("비전 이미지 Base64 형식이 올바르지 않습니다.") from exc
        if not data or len(data) > 3 * 1024 * 1024:
            raise ValueError("비전 이미지는 3MB 이하만 허용합니다.")
        fmt = str(payload.get("format", "jpeg")).lower()
        content_type = "image/png" if "png" in fmt else "image/jpeg"
        metadata = {key: value for key, value in payload.items() if key != "data_base64"}
        version = self.vision_hub.set_image(kind, data, content_type, metadata)
        self.send_json({"ok": True, "version": version}, HTTPStatus.CREATED)

    def receive_force_hit(self, payload: dict[str, Any]) -> None:
        item = self.force_hub.publish(payload)
        self.send_json({"ok": True, "version": item["version"]}, HTTPStatus.CREATED)

    def _persist_force_hits(self, db: sqlite3.Connection, session_id: int, after_version: int) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        session_row = db.execute("SELECT client_session_id FROM training_sessions WHERE id=?", (session_id,)).fetchone()
        expected_client_session_id = str(session_row["client_session_id"] or "").strip() if session_row else ""
        for item in self.force_hub.after(after_version):
            payload = dict(item.get("payload") or {})
            payload_client_session_id = str(payload.get("client_session_id", "")).strip()
            # When a training session has a UUID, accept only force hits carrying
            # that exact UUID. Missing IDs fail closed so a stale/foreign hit can
            # never be attached to the current session.
            if expected_client_session_id and payload_client_session_id != expected_client_session_id:
                continue
            values = {
                "received_version": int(item["version"]),
                "hit_id": payload.get("hit_id"),
                "stamp_ns": payload.get("stamp_ns"),
                "valid_hit": payload.get("valid_hit"),
                "invalid_reason": str(payload.get("invalid_reason", ""))[:300],
                "hit_direction": str(payload.get("hit_direction", ""))[:80],
                "hit_x_mm": payload.get("hit_x_mm"), "hit_y_mm": payload.get("hit_y_mm"),
                "center_error_mm": payload.get("center_error_mm"),
                "peak_force_n": payload.get("peak_force_n"),
                "peak_normal_force_n": payload.get("peak_normal_force_n"),
                "impulse_ns": payload.get("impulse_ns"),
                "contact_duration_ms": payload.get("contact_duration_ms"),
                "accuracy_score": payload.get("accuracy_score"),
                "power_score": payload.get("power_score"),
                "total_score": payload.get("total_score"),
                "force_warning": payload.get("force_warning"),
                "safety_stop": payload.get("safety_stop"),
            }
            cursor = db.execute(
                """INSERT OR IGNORE INTO force_hit_results(
                    session_id, received_version, hit_id, stamp_ns, valid_hit, invalid_reason, hit_direction,
                    hit_x_mm, hit_y_mm, center_error_mm, peak_force_n, peak_normal_force_n, impulse_ns,
                    contact_duration_ms, accuracy_score, power_score, total_score, force_warning, safety_stop, raw_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, values["received_version"], values["hit_id"], values["stamp_ns"],
                 None if values["valid_hit"] is None else int(bool(values["valid_hit"])), values["invalid_reason"], values["hit_direction"],
                 values["hit_x_mm"], values["hit_y_mm"], values["center_error_mm"], values["peak_force_n"], values["peak_normal_force_n"],
                 values["impulse_ns"], values["contact_duration_ms"], values["accuracy_score"], values["power_score"], values["total_score"],
                 None if values["force_warning"] is None else int(bool(values["force_warning"])),
                 None if values["safety_stop"] is None else int(bool(values["safety_stop"])),
                 json.dumps(payload, ensure_ascii=False), utc_now()),
            )
            row = db.execute("SELECT * FROM force_hit_results WHERE session_id=? AND received_version=?", (session_id, values["received_version"])).fetchone()
            if row is not None:
                saved.append(dict(row))
        return saved

    @staticmethod
    def _match_force(vision_event: dict[str, Any], force_rows: list[dict[str, Any]], used: set[int]) -> dict[str, Any] | None:
        raw = vision_event.get("raw_event") or {}
        vision_stamp = raw.get("impact_stamp_ns")
        best: tuple[float, int, dict[str, Any]] | None = None
        for row in force_rows:
            row_id = int(row["id"])
            if row_id in used:
                continue
            force_stamp = row.get("stamp_ns")
            if vision_stamp is not None and force_stamp is not None:
                distance = abs(int(vision_stamp) - int(force_stamp)) / 1e6
                if distance > 500.0:
                    continue
            else:
                distance = float(row.get("received_version") or 0)
            candidate = (distance, row_id, row)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            return None
        used.add(best[1])
        return best[2]

    def sessionize_punch_events(self, payload: dict[str, Any]) -> None:
        try:
            session_id = int(payload.get("session_id"))
            after_evidence_version = max(0, int(payload.get("after_evidence_version", 0)))
            after_force_version = max(0, int(payload.get("after_force_version", 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id/evidence/force version을 확인해 주세요.") from exc
        events = payload.get("punch_events") or []
        if not isinstance(events, list):
            raise ValueError("punch_events는 배열이어야 합니다.")
        evidence_records = self.vision_hub.evidence_records_after(after_evidence_version)
        # Snapshot publication is asynchronous; give the final impact images a short chance to arrive.
        if events and len(evidence_records) < len(events):
            deadline = time.monotonic() + 1.8
            while len(evidence_records) < len(events) and time.monotonic() < deadline:
                time.sleep(0.05)
                evidence_records = self.vision_hub.evidence_records_after(after_evidence_version)
        evidence_by_id = {
            int(record["metadata"].get("impact_id")): record
            for record in evidence_records
            if str(record["metadata"].get("impact_id", "")).isdigit()
        }
        session_dir = EVIDENCE_DIR / f"session_{session_id:06d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        stored_events: list[dict[str, Any]] = []
        with connect(self.database) as db:
            if db.execute("SELECT 1 FROM training_sessions WHERE id=?", (session_id,)).fetchone() is None:
                self.send_json({"error": "훈련 세션을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            force_rows = self._persist_force_hits(db, session_id, after_force_version)
            used_force: set[int] = set()
            db.execute("DELETE FROM punch_events WHERE session_id=?", (session_id,))
            for index, source in enumerate(events, start=1):
                if not isinstance(source, dict):
                    continue
                event = dict(source)
                vision_id = int(event.get("punch_id") or index)
                force = self._match_force(event, force_rows, used_force)
                score = event_score(event, force)
                evidence = evidence_by_id.get(vision_id)
                if evidence is None and index - 1 < len(evidence_records):
                    evidence = evidence_records[index - 1]
                evidence_path = None
                evidence_version = None
                if evidence is not None:
                    evidence_version = int(evidence["version"])
                    evidence_file = session_dir / f"punch_{index:03d}_impact_{vision_id:05d}.jpg"
                    evidence_file.write_bytes(evidence["data"])
                    evidence_path = str(evidence_file)
                tags = issue_tags({**event, "force": force})
                cursor = db.execute(
                    """INSERT INTO punch_events(
                        session_id,punch_index,vision_punch_id,punch_side,punch_type,event_score,passed,violations_json,issue_tags_json,quality_json,
                        force_result_id,evidence_version,evidence_path,is_best,is_worst,raw_json,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,?)""",
                    (session_id,index,vision_id,str(event.get("punch_side","")),str(event.get("punch_type","impact")),score,
                     int(bool(event.get("passed", True))),json.dumps(event.get("violations") or [],ensure_ascii=False),
                     json.dumps(tags,ensure_ascii=False),json.dumps(event.get("quality") or {},ensure_ascii=False),
                     force.get("id") if force else None,evidence_version,evidence_path,json.dumps(event,ensure_ascii=False),utc_now()),
                )
                stored_events.append({
                    "id": int(cursor.lastrowid), "punch_index": index, "vision_punch_id": vision_id,
                    "punch_side": event.get("punch_side"), "punch_type": event.get("punch_type"), "event_score": score,
                    "issue_tags": tags, "evidence_url": f"/api/punch-events/{int(cursor.lastrowid)}/evidence.jpg" if evidence_path else None,
                    "force": force,
                })
            best, worst = select_best_worst(stored_events)
            if best:
                db.execute("UPDATE punch_events SET is_best=1 WHERE id=?", (best["id"],))
            if worst:
                db.execute("UPDATE punch_events SET is_worst=1 WHERE id=?", (worst["id"],))
            db.commit()
        self.send_json({"ok": True, "count": len(stored_events), "best": compact_event(best), "worst": compact_event(worst), "force_count": len(force_rows)}, HTTPStatus.CREATED)

    def serve_punch_evidence(self, event_id: int) -> None:
        with connect(self.database) as db:
            row = db.execute("SELECT evidence_path FROM punch_events WHERE id=?", (event_id,)).fetchone()
        if row is None or not row["evidence_path"]:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = Path(str(row["evidence_path"]))
        try:
            resolved = path.resolve()
            root = EVIDENCE_DIR.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def control_wakeword(self, payload: dict[str, Any]) -> None:
        if self.wakeword_service is None:
            self.send_json({"error": "웨이크업 서비스가 실행 중이 아닙니다."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        enabled = bool(payload.get("enabled", True))
        self.wakeword_service.set_enabled(enabled)
        self.send_json(self.wakeword_service.status())

    def suppress_wakeword(self, payload: dict[str, Any]) -> None:
        if self.wakeword_service is None:
            self.send_json({"ok": True, "active": False})
            return
        try:
            duration_ms = int(payload.get("duration_ms", 2500))
        except (TypeError, ValueError):
            duration_ms = 2500
        self.wakeword_service.suppress(duration_ms)
        self.send_json({"ok": True, "duration_ms": max(0, min(30000, duration_ms))})

    def control_voice_session(self, payload: dict[str, Any]) -> None:
        if self.wakeword_service is None:
            self.send_json({"error": "음성 서비스가 실행 중이 아닙니다."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        action = str(payload.get("action", "extend")).strip().lower()
        try:
            duration_sec = float(payload.get("duration_sec", 30))
        except (TypeError, ValueError):
            duration_sec = 30.0
        if action == "start":
            status = self.wakeword_service.start_session(duration_sec, source="ui")
        elif action in {"extend", "set"}:
            status = self.wakeword_service.extend_session(duration_sec, source="ui")
        elif action == "end":
            status = self.wakeword_service.end_session(str(payload.get("reason", "ui")))
        else:
            self.send_json({"error": "지원하지 않는 음성 세션 명령입니다."}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json(status)

    def database_status(self) -> None:
        try:
            with connect(self.database) as db:
                user_count = int(db.execute("SELECT COUNT(*) FROM users").fetchone()[0])
                session_count = int(db.execute("SELECT COUNT(*) FROM training_sessions").fetchone()[0])
                vision_count = int(db.execute("SELECT COUNT(*) FROM vision_results").fetchone()[0])
                report_count = int(db.execute("SELECT COUNT(*) FROM ai_reports").fetchone()[0])
                punch_event_count = int(db.execute("SELECT COUNT(*) FROM punch_events").fetchone()[0])
                force_hit_count = int(db.execute("SELECT COUNT(*) FROM force_hit_results").fetchone()[0])
                feedback_goal_count = int(db.execute("SELECT COUNT(*) FROM feedback_goals").fetchone()[0])
            display_path = str(self.database.relative_to(BASE_DIR)) if self.database.is_relative_to(BASE_DIR) else str(self.database)
            self.send_json({
                "ok": True,
                "engine": "SQLite",
                "path": display_path,
                "users": user_count,
                "sessions": session_count,
                "vision_results": vision_count,
                "ai_reports": report_count,
                "punch_events": punch_event_count,
                "force_hit_results": force_hit_count,
                "feedback_goals": feedback_goal_count,
            })
        except sqlite3.Error as exc:
            self.send_json({"ok": False, "engine": "SQLite", "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def list_users(self) -> None:
        with connect(self.database) as db:
            rows = db.execute(
                """
                SELECT id, name, height_cm, dominant_hand, wingspan_cm,
                       left_punch_reach_cm, right_punch_reach_cm,
                       recommended_distance_cm, measurement_confidence,
                       created_at, updated_at, last_training_at
                FROM users
                ORDER BY COALESCE(last_training_at, updated_at) DESC, id DESC
                """
            ).fetchall()
        self.send_json([dict(row) for row in rows])

    def get_user(self, user_id: int) -> None:
        with connect(self.database) as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            self.send_json({"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(dict(row))

    def create_user(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name", "")).strip()
        dominant_hand = str(payload.get("dominant_hand", "right")).lower()
        try:
            height_cm = float(payload.get("height_cm"))
        except (TypeError, ValueError) as exc:
            raise ValueError("키를 숫자로 입력해 주세요.") from exc
        if not name or len(name) > 30:
            raise ValueError("이름은 1~30자로 입력해 주세요.")
        if not 100 <= height_cm <= 230:
            raise ValueError("키는 100~230cm 범위로 입력해 주세요.")
        if dominant_hand not in {"right", "left"}:
            raise ValueError("주 사용 손 값이 올바르지 않습니다.")

        now = utc_now()
        with connect(self.database) as db:
            cursor = db.execute(
                "INSERT INTO users(name, height_cm, dominant_hand, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (name, height_cm, dominant_hand, now, now),
            )
            db.commit()
            row = db.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        self.send_json(dict(row), HTTPStatus.CREATED)

    def save_measurement(self, user_id: int, payload: dict[str, Any]) -> None:
        fields = [
            "wingspan_cm",
            "left_punch_reach_cm",
            "right_punch_reach_cm",
            "recommended_distance_cm",
            "measurement_confidence",
        ]
        values: dict[str, float | None] = {}
        for field in fields:
            raw = payload.get(field)
            if raw in (None, ""):
                values[field] = None
                continue
            try:
                values[field] = round(float(raw), 2)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} 값이 올바르지 않습니다.") from exc
        wingspan = values["wingspan_cm"]
        confidence = values["measurement_confidence"]
        if wingspan is not None and not 80 <= wingspan <= 260:
            raise ValueError("리치 측정값이 허용 범위를 벗어났습니다.")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("측정 신뢰도는 0~1 범위여야 합니다.")

        with connect(self.database) as db:
            exists = db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            if exists is None:
                self.send_json({"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            db.execute(
                """
                UPDATE users
                   SET wingspan_cm = ?, left_punch_reach_cm = ?, right_punch_reach_cm = ?,
                       recommended_distance_cm = ?, measurement_confidence = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    values["wingspan_cm"], values["left_punch_reach_cm"],
                    values["right_punch_reach_cm"], values["recommended_distance_cm"],
                    values["measurement_confidence"], utc_now(), user_id,
                ),
            )
            db.commit()
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        self.send_json(dict(row))

    def save_session(self, payload: dict[str, Any]) -> None:
        try:
            user_id = int(payload.get("user_id"))
            duration_sec = int(payload.get("duration_sec", 60))
            punch_count = int(payload.get("punch_count", 0))
            success_rate = float(payload.get("success_rate", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("훈련 기록 값이 올바르지 않습니다.") from exc
        training_type = str(payload.get("training_type", "straight"))[:30]
        hand = str(payload.get("hand", "right"))
        if hand not in {"right", "left", "both"}:
            raise ValueError("훈련 손 정보가 올바르지 않습니다.")
        if not 1 <= duration_sec <= 3600 or punch_count < 0 or not 0 <= success_rate <= 100:
            raise ValueError("훈련 기록 범위를 확인해 주세요.")

        def optional_float(key: str) -> float | None:
            raw = payload.get(key)
            if raw in (None, ""):
                return None
            return float(raw)

        try:
            avg_reaction_ms = optional_float("avg_reaction_ms")
            posture_score = optional_float("posture_score")
        except (TypeError, ValueError) as exc:
            raise ValueError("반응시간 또는 자세 점수 값이 올바르지 않습니다.") from exc

        feedback = str(payload.get("feedback", ""))[:500]
        client_session_id = str(payload.get("client_session_id", ""))[:100] or None
        now = utc_now()
        with connect(self.database) as db:
            exists = db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            if exists is None:
                self.send_json({"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            cursor = db.execute(
                """
                INSERT INTO training_sessions(
                    user_id, training_type, hand, duration_sec, punch_count,
                    success_rate, avg_reaction_ms, posture_score, feedback, client_session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, training_type, hand, duration_sec, punch_count,
                    round(success_rate, 2), avg_reaction_ms, posture_score, feedback, client_session_id, now,
                ),
            )
            session_id = int(cursor.lastrowid)
            initial_report = {
                "headline": "이번 훈련 코칭 요약",
                "coach_message": feedback,
                "strengths": [],
                "improvements": [],
                "force_analysis": "",
                "next_training": None,
                "source": "local-rules",
            }
            db.execute(
                """
                INSERT INTO ai_reports(
                    session_id, summary, strengths_json, improvements_json,
                    next_training, coach_message, model, raw_json, created_at, updated_at
                ) VALUES (?, ?, '[]', '[]', '', ?, 'local-rules', ?, ?, ?)
                """,
                (
                    session_id,
                    feedback,
                    feedback,
                    json.dumps(initial_report, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            db.execute("UPDATE users SET last_training_at = ?, updated_at = ? WHERE id = ?", (now, now, user_id))
            db.commit()
        self.send_json({"id": session_id, "saved": True}, HTTPStatus.CREATED)

    def list_mitt_calibrations(self, user_id: int) -> None:
        with connect(self.database) as db:
            if db.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone() is None:
                self.send_json(
                    {"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND
                )
                return
            rows = db.execute(
                """
                SELECT id, user_id, punch_role, hand, correction_x_mm,
                       correction_y_mm, raw_center_x_mm, raw_center_y_mm,
                       sample_count, accepted_sample_count, dispersion_mm,
                       correction_limited, base_pose_json,
                       calibrated_pose_json, vision_summary_json,
                       calibration_version, created_at, updated_at
                  FROM user_mitt_calibrations
                 WHERE user_id = ?
                 ORDER BY CASE punch_role WHEN 'jab' THEN 0 ELSE 1 END
                """,
                (user_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            for key in (
                "base_pose_json",
                "calibrated_pose_json",
                "vision_summary_json",
            ):
                item[key.removesuffix("_json")] = (
                    json.loads(item.pop(key)) if item[key] else None
                )
            item["correction_limited"] = bool(item["correction_limited"])
            output.append(item)
        self.send_json(output)

    def get_reach_calibration(self, user_id: int) -> None:
        with connect(self.database) as db:
            if db.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone() is None:
                self.send_json(
                    {"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND
                )
                return
            row = db.execute(
                "SELECT * FROM user_reach_calibrations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            self.send_json(None)
            return
        item = dict(row)
        item["base_pose"] = json.loads(item.pop("base_pose_json"))
        item["contact_pose"] = json.loads(item.pop("contact_pose_json"))
        self.send_json(item)

    def save_reach_calibration(
        self, user_id: int, payload: dict[str, Any]
    ) -> None:
        hand = str(payload.get("hand", "")).strip().lower()
        if hand not in {"left", "right"}:
            raise ValueError("리치 보정 손 정보가 올바르지 않습니다.")
        try:
            correction_z = float(payload["correction_z_mm"])
            baseline_force = float(payload["baseline_normal_force_n"])
            contact_force = float(payload["contact_delta_force_n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("리치 보정 외력 값이 올바르지 않습니다.") from exc
        if not all(
            math.isfinite(value)
            for value in (correction_z, baseline_force, contact_force)
        ):
            raise ValueError("리치 보정 값은 유한해야 합니다.")
        if correction_z < 0.0:
            raise ValueError("리치 앞뒤 보정량은 0mm 이상이어야 합니다.")
        if contact_force <= 0.0:
            raise ValueError("접촉 외력은 양수여야 합니다.")

        def pose(key: str) -> list[float]:
            raw = payload.get(key)
            if not isinstance(raw, list) or len(raw) != 6:
                raise ValueError(f"{key}는 6축 TCP 자세여야 합니다.")
            values = [float(value) for value in raw]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{key} 값은 유한해야 합니다.")
            return values

        base_pose = pose("base_pose")
        contact_pose = pose("contact_pose")
        with connect(self.database) as db:
            user = db.execute(
                "SELECT dominant_hand FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if user is None:
                self.send_json(
                    {"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND
                )
                return
            expected = "left" if str(user["dominant_hand"]) == "right" else "right"
            if hand != expected:
                raise ValueError("리치 보정은 주손 반대쪽 팔로 진행해야 합니다.")
            now = utc_now()
            db.execute(
                """
                INSERT INTO user_reach_calibrations(
                    user_id, hand, correction_z_mm, baseline_normal_force_n,
                    contact_delta_force_n, base_pose_json, contact_pose_json,
                    calibration_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    hand = excluded.hand,
                    correction_z_mm = excluded.correction_z_mm,
                    baseline_normal_force_n = excluded.baseline_normal_force_n,
                    contact_delta_force_n = excluded.contact_delta_force_n,
                    base_pose_json = excluded.base_pose_json,
                    contact_pose_json = excluded.contact_pose_json,
                    calibration_version = user_reach_calibrations.calibration_version + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id, hand, correction_z, baseline_force, contact_force,
                    json.dumps(base_pose), json.dumps(contact_pose), now, now,
                ),
            )
            db.commit()
            row = db.execute(
                "SELECT * FROM user_reach_calibrations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        self.send_json(
            {
                "saved": True,
                "id": int(row["id"]),
                "user_id": user_id,
                "calibration_version": int(row["calibration_version"]),
            },
            HTTPStatus.CREATED,
        )

    def save_mitt_calibration(
        self, user_id: int, payload: dict[str, Any]
    ) -> None:
        role = str(payload.get("punch_role", "")).strip().lower()
        hand = str(payload.get("hand", "")).strip().lower()
        if role not in {"jab", "straight"}:
            raise ValueError("보정 펀치는 잽 또는 스트레이트여야 합니다.")
        if hand not in {"left", "right"}:
            raise ValueError("보정 손 정보가 올바르지 않습니다.")

        numeric_keys = (
            "correction_x_mm",
            "correction_y_mm",
            "raw_center_x_mm",
            "raw_center_y_mm",
            "dispersion_mm",
        )
        try:
            values = {key: float(payload[key]) for key in numeric_keys}
            sample_count = int(payload["sample_count"])
            accepted_sample_count = int(payload["accepted_sample_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("미트 보정 결과 값이 올바르지 않습니다.") from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("미트 보정 결과는 유한한 값이어야 합니다.")
        if not -50.0 <= values["correction_x_mm"] <= 50.0 or not -50.0 <= values["correction_y_mm"] <= 50.0:
            raise ValueError("미트 보정량은 축별 ±50mm 이하여야 합니다.")
        if sample_count < 5 or not 0 < accepted_sample_count <= sample_count:
            raise ValueError("5회의 유효한 보정 표본이 필요합니다.")
        if values["dispersion_mm"] < 0.0:
            raise ValueError("보정 분산은 음수일 수 없습니다.")

        def pose_value(key: str) -> list[float] | None:
            raw = payload.get(key)
            if raw in (None, ""):
                return None
            if not isinstance(raw, list) or len(raw) != 6:
                raise ValueError(f"{key}는 6축 TCP 자세여야 합니다.")
            pose = [float(value) for value in raw]
            if not all(math.isfinite(value) for value in pose):
                raise ValueError(f"{key} 값은 유한해야 합니다.")
            return pose

        base_pose = pose_value("base_pose")
        calibrated_pose = pose_value("calibrated_pose")
        raw_samples = payload.get("raw_samples")
        if not isinstance(raw_samples, list) or len(raw_samples) < 5:
            raise ValueError("보정 원본 타격 5회가 필요합니다.")
        vision_summary = payload.get("vision_summary")

        with connect(self.database) as db:
            user = db.execute(
                "SELECT dominant_hand FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if user is None:
                self.send_json(
                    {"error": "사용자를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND
                )
                return
            dominant = str(user["dominant_hand"])
            expected_hand = dominant if role == "straight" else (
                "left" if dominant == "right" else "right"
            )
            if hand != expected_hand:
                raise ValueError(
                    f"{dominant}손잡이의 {role} 보정 손은 {expected_hand}손입니다."
                )
            now = utc_now()
            db.execute(
                """
                INSERT INTO user_mitt_calibrations(
                    user_id, punch_role, hand, correction_x_mm,
                    correction_y_mm, raw_center_x_mm, raw_center_y_mm,
                    sample_count, accepted_sample_count, dispersion_mm,
                    correction_limited, base_pose_json, calibrated_pose_json,
                    raw_samples_json, vision_summary_json,
                    calibration_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(user_id, punch_role) DO UPDATE SET
                    hand = excluded.hand,
                    correction_x_mm = excluded.correction_x_mm,
                    correction_y_mm = excluded.correction_y_mm,
                    raw_center_x_mm = excluded.raw_center_x_mm,
                    raw_center_y_mm = excluded.raw_center_y_mm,
                    sample_count = excluded.sample_count,
                    accepted_sample_count = excluded.accepted_sample_count,
                    dispersion_mm = excluded.dispersion_mm,
                    correction_limited = excluded.correction_limited,
                    base_pose_json = excluded.base_pose_json,
                    calibrated_pose_json = excluded.calibrated_pose_json,
                    raw_samples_json = excluded.raw_samples_json,
                    vision_summary_json = excluded.vision_summary_json,
                    calibration_version = user_mitt_calibrations.calibration_version + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    role,
                    hand,
                    values["correction_x_mm"],
                    values["correction_y_mm"],
                    values["raw_center_x_mm"],
                    values["raw_center_y_mm"],
                    sample_count,
                    accepted_sample_count,
                    values["dispersion_mm"],
                    int(bool(payload.get("correction_limited"))),
                    json.dumps(base_pose) if base_pose is not None else None,
                    json.dumps(calibrated_pose) if calibrated_pose is not None else None,
                    json.dumps(raw_samples, ensure_ascii=False),
                    json.dumps(vision_summary, ensure_ascii=False)
                    if vision_summary is not None
                    else None,
                    now,
                    now,
                ),
            )
            db.commit()
            row = db.execute(
                "SELECT * FROM user_mitt_calibrations WHERE user_id = ? AND punch_role = ?",
                (user_id, role),
            ).fetchone()
        self.send_json(
            {
                "saved": True,
                "id": int(row["id"]),
                "user_id": user_id,
                "punch_role": role,
                "hand": hand,
                "calibration_version": int(row["calibration_version"]),
            },
            HTTPStatus.CREATED,
        )

    def list_sessions(self, user_id: int) -> None:
        with connect(self.database) as db:
            rows = db.execute(
                "SELECT * FROM training_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                (user_id,),
            ).fetchall()
        self.send_json([dict(row) for row in rows])

    def save_vision_result(self, payload: dict[str, Any]) -> None:
        try:
            session_id = int(payload.get("session_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id가 필요합니다.") from exc
        numeric_fields = [
            "total_punches", "successful_punches", "accuracy_percent",
            "average_reaction_sec", "average_guard_return_sec", "guard_drop_count",
            "slow_guard_return_count", "arm_extension_score", "guard_score",
            "torso_balance_score",
        ]
        values: dict[str, Any] = {}
        for field in numeric_fields:
            raw = payload.get(field)
            values[field] = None if raw in (None, "") else float(raw)
        images = payload.get("representative_images", [])
        if not isinstance(images, list):
            raise ValueError("representative_images는 배열이어야 합니다.")
        now = utc_now()
        with connect(self.database) as db:
            if db.execute("SELECT 1 FROM training_sessions WHERE id = ?", (session_id,)).fetchone() is None:
                self.send_json({"error": "훈련 세션을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            cursor = db.execute(
                """
                INSERT INTO vision_results(
                    session_id, total_punches, successful_punches, accuracy_percent,
                    average_reaction_sec, average_guard_return_sec, guard_drop_count,
                    slow_guard_return_count, arm_extension_score, guard_score,
                    torso_balance_score, representative_images_json, raw_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, values["total_punches"], values["successful_punches"],
                    values["accuracy_percent"], values["average_reaction_sec"],
                    values["average_guard_return_sec"], values["guard_drop_count"],
                    values["slow_guard_return_count"], values["arm_extension_score"],
                    values["guard_score"], values["torso_balance_score"],
                    json.dumps(images, ensure_ascii=False), json.dumps(payload, ensure_ascii=False), now,
                ),
            )
            db.commit()
        self.send_json({"id": int(cursor.lastrowid), "saved": True}, HTTPStatus.CREATED)

    @staticmethod
    def _force_summary(db: sqlite3.Connection, session_id: int) -> dict[str, Any]:
        row = db.execute(
            """SELECT COUNT(*) AS hit_count,
                      SUM(CASE WHEN valid_hit=1 THEN 1 ELSE 0 END) AS valid_hit_count,
                      MAX(peak_force_n) AS peak_force_n,
                      AVG(CASE WHEN valid_hit=1 THEN peak_force_n END) AS average_peak_force_n,
                      AVG(CASE WHEN valid_hit=1 THEN peak_normal_force_n END) AS average_peak_normal_force_n,
                      AVG(CASE WHEN valid_hit=1 THEN accuracy_score END) AS average_accuracy_score,
                      AVG(CASE WHEN valid_hit=1 THEN center_error_mm END) AS average_center_error_mm,
                      AVG(CASE WHEN valid_hit=1 THEN impulse_ns END) AS average_impulse_ns,
                      AVG(CASE WHEN valid_hit=1 THEN contact_duration_ms END) AS average_contact_duration_ms,
                      SUM(CASE WHEN force_warning=1 THEN 1 ELSE 0 END) AS force_warning_count
                 FROM force_hit_results WHERE session_id=?""",
            (session_id,),
        ).fetchone()
        summary = dict(row) if row else {}
        direction_rows = db.execute(
            """SELECT hit_direction, COUNT(*) AS count FROM force_hit_results
                 WHERE session_id=? AND valid_hit=1 AND hit_direction IS NOT NULL AND hit_direction!=''
                 GROUP BY hit_direction ORDER BY count DESC""",
            (session_id,),
        ).fetchall()
        summary["direction_counts"] = {str(item["hit_direction"]): int(item["count"]) for item in direction_rows}
        return summary

    @staticmethod
    def _event_summary(db: sqlite3.Connection, session_id: int, which: str) -> dict[str, Any] | None:
        column = "is_best" if which == "best" else "is_worst"
        row = db.execute(f"SELECT * FROM punch_events WHERE session_id=? AND {column}=1 LIMIT 1", (session_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["issue_tags"] = json.loads(item.get("issue_tags_json") or "[]")
        item["evidence_url"] = f"/api/punch-events/{item['id']}/evidence.jpg" if item.get("evidence_path") else None
        if item.get("force_result_id"):
            force = db.execute("SELECT * FROM force_hit_results WHERE id=?", (item["force_result_id"],)).fetchone()
            item["force"] = dict(force) if force else None
        else:
            item["force"] = None
        return item

    def _progress_context(self, db: sqlite3.Connection, session: sqlite3.Row) -> dict[str, Any]:
        current_vision = db.execute("SELECT * FROM vision_results WHERE session_id=? ORDER BY id DESC LIMIT 1", (session["id"],)).fetchone()
        current_force = self._force_summary(db, int(session["id"]))
        current_metrics = derive_metrics(dict(session), dict(current_vision) if current_vision else None, current_force)
        previous = db.execute(
            """SELECT * FROM training_sessions
                 WHERE user_id=? AND id<? AND training_type=? AND hand=?
                 ORDER BY id DESC LIMIT 1""",
            (session["user_id"], session["id"], session["training_type"], session["hand"]),
        ).fetchone()
        goal = db.execute(
            """SELECT * FROM feedback_goals
                 WHERE user_id=? AND source_session_id<?
                 ORDER BY id DESC LIMIT 1""",
            (session["user_id"], session["id"]),
        ).fetchone()
        if previous is None:
            return {
                "has_previous": False, "previous_session_id": None,
                "previous_feedback": None, "tracked_goal": dict(goal) if goal else None,
                "current_metrics": current_metrics, "comparisons": [], "tracked_result": None,
            }
        previous_vision = db.execute("SELECT * FROM vision_results WHERE session_id=? ORDER BY id DESC LIMIT 1", (previous["id"],)).fetchone()
        previous_force = self._force_summary(db, int(previous["id"]))
        previous_metrics = derive_metrics(dict(previous), dict(previous_vision) if previous_vision else None, previous_force)
        tracked_metric = str(goal["metric_key"]) if goal else infer_tracked_metric(previous["feedback"], previous_metrics, current_metrics)
        progress = build_progress(previous_metrics, current_metrics, tracked_metric)
        progress.update({
            "previous_session_id": int(previous["id"]),
            "previous_feedback": previous["feedback"],
            "tracked_goal": dict(goal) if goal else None,
            "previous_metrics": previous_metrics,
            "current_metrics": current_metrics,
        })
        return progress

    def generate_vision_coach(self, payload: dict[str, Any]) -> None:
        try:
            session_id = int(payload.get("session_id"))
            after_version = max(0, int(payload.get("after_evidence_version", 0)))
            expected_image_count = max(0, min(60, int(payload.get("expected_image_count", 0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id, after_evidence_version, expected_image_count를 확인해 주세요.") from exc
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("metrics는 JSON 객체여야 합니다.")
        fallback = re.sub(r"\s+", " ", str(payload.get("fallback_feedback", "훈련 데이터를 저장했습니다. 다음 훈련에서도 자세를 안정적으로 유지하세요."))).strip()[:500]

        with connect(self.database) as db:
            session = db.execute("SELECT * FROM training_sessions WHERE id=?", (session_id,)).fetchone()
            if session is None:
                self.send_json({"error": "훈련 세션을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            best = self._event_summary(db, session_id, "best")
            worst = self._event_summary(db, session_id, "worst")
            progress = self._progress_context(db, session)
            force_summary = self._force_summary(db, session_id)

        # B option: BEST PUNCH + CHECK POINT are the primary evidence images.
        selected: list[tuple[int, bytes, str]] = []
        image_roles: list[dict[str, Any]] = []
        for role, event in (("best_punch", best), ("check_point", worst)):
            if event and event.get("evidence_path"):
                path = Path(str(event["evidence_path"]))
                if path.is_file():
                    selected.append((int(event.get("evidence_version") or event["id"]), path.read_bytes(), "image/jpeg"))
                    image_roles.append({
                        "role": role, "punch_index": event.get("punch_index"),
                        "event_score": event.get("event_score"), "issue_tags": event.get("issue_tags", []),
                    })
        if not selected:
            evidence = self.vision_hub.evidence_after(after_version)
            if bool(os.environ.get("OPENAI_API_KEY", "").strip()) and expected_image_count > len(evidence):
                deadline = time.monotonic() + 1.5
                while len(evidence) < expected_image_count and time.monotonic() < deadline:
                    time.sleep(0.05)
                    evidence = self.vision_hub.evidence_after(after_version)
            selected = select_representative_images(evidence, 2)
            image_roles = [{"role": "representative", "index": index + 1} for index in range(len(selected))]

        analysis_metrics = {
            "training_type": session["training_type"], "hand": session["hand"],
            "duration_sec": session["duration_sec"], "punch_count": session["punch_count"],
            "success_rate": session["success_rate"], "average_reaction_ms": session["avg_reaction_ms"],
            "progress": progress, "force_summary": force_summary,
            "best_punch": compact_event(best), "check_point": compact_event(worst),
            "image_roles": image_roles, **metrics,
        }
        configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        if not configured or not selected:
            reason = "api_key_missing" if not configured else "no_session_images"
            self.send_json({
                "ok": True, "used_ai": False, "coach_message": fallback, "reason": reason,
                "image_count": len(selected), "progress": progress, "force_summary": force_summary, "best": compact_event(best), "worst": compact_event(worst),
            })
            return
        try:
            result = analyze_boxing_images(selected, analysis_metrics)
        except VisionCoachError as exc:
            self.send_json({
                "ok": True, "used_ai": False, "coach_message": fallback, "reason": "openai_error",
                "message": str(exc), "image_count": len(selected), "progress": progress, "force_summary": force_summary,
                "best": compact_event(best), "worst": compact_event(worst),
            })
            return

        coach_message = str(result.get("coach_message", fallback)).strip()[:500] or fallback
        strength = str(result.get("observed_strength", "")).strip()[:1000]
        improvement = str(result.get("improvement", "")).strip()[:1000]
        strengths = result.get("strengths") if isinstance(result.get("strengths"), list) else ([strength] if strength else [])
        improvements = result.get("improvements") if isinstance(result.get("improvements"), list) else ([improvement] if improvement else [])
        next_focus = str(result.get("next_focus", "")).strip()[:1000]
        next_training = result.get("next_training") if isinstance(result.get("next_training"), dict) else {}
        next_focus_metric = str(result.get("next_focus_metric", "none"))
        now = utc_now()
        stored = {**result, "evidence_versions": [item[0] for item in selected], "metrics": analysis_metrics}
        with connect(self.database) as db:
            db.execute("UPDATE training_sessions SET feedback=? WHERE id=?", (coach_message, session_id))
            db.execute(
                """INSERT INTO ai_reports(session_id,summary,strengths_json,improvements_json,next_training,coach_message,model,raw_json,created_at,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary,strengths_json=excluded.strengths_json,
                     improvements_json=excluded.improvements_json,next_training=excluded.next_training,coach_message=excluded.coach_message,
                     model=excluded.model,raw_json=excluded.raw_json,updated_at=excluded.updated_at""",
                (session_id, coach_message, json.dumps(strengths[:3], ensure_ascii=False),
                 json.dumps(improvements[:3], ensure_ascii=False),
                 json.dumps(next_training, ensure_ascii=False) if next_training else next_focus, coach_message,
                 str(result.get("model", ""))[:100], json.dumps(stored, ensure_ascii=False), now, now),
            )
            if next_focus_metric in METRICS and next_focus_metric in progress.get("current_metrics", {}) and next_focus:
                db.execute("DELETE FROM feedback_goals WHERE source_session_id=?", (session_id,))
                db.execute(
                    "INSERT INTO feedback_goals(user_id,source_session_id,metric_key,metric_label,baseline_value,advice,created_at) VALUES(?,?,?,?,?,?,?)",
                    (session["user_id"], session_id, next_focus_metric, METRICS[next_focus_metric]["label"],
                     progress["current_metrics"][next_focus_metric], next_focus, now),
                )
            db.commit()
        self.send_json({
            "ok": True, "used_ai": True,
            "headline": result.get("headline"),
            "coach_message": coach_message, "progress_message": result.get("progress_message"),
            "strengths": strengths[:3], "improvements": improvements[:3],
            "force_analysis": result.get("force_analysis"),
            "best_punch_comment": result.get("best_punch_comment"), "check_point_comment": result.get("check_point_comment"),
            "next_focus": next_focus, "next_focus_metric": next_focus_metric, "next_training": next_training, "model": result.get("model"),
            "image_count": len(selected), "visual_confidence": result.get("visual_confidence"),
            "progress": progress, "force_summary": force_summary, "best": compact_event(best), "worst": compact_event(worst),
        })

    def save_ai_report(self, payload: dict[str, Any]) -> None:
        try:
            session_id = int(payload.get("session_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id가 필요합니다.") from exc
        summary = str(payload.get("summary", ""))[:3000]
        strengths = payload.get("strengths", [])
        improvements = payload.get("improvements", [])
        if not isinstance(strengths, list) or not isinstance(improvements, list):
            raise ValueError("strengths와 improvements는 배열이어야 합니다.")
        now = utc_now()
        raw_json = json.dumps(payload, ensure_ascii=False)
        next_training_value = payload.get("next_training", "")
        next_training = (
            json.dumps(next_training_value, ensure_ascii=False)
            if isinstance(next_training_value, (dict, list))
            else str(next_training_value)
        )[:2000]
        with connect(self.database) as db:
            if db.execute("SELECT 1 FROM training_sessions WHERE id = ?", (session_id,)).fetchone() is None:
                self.send_json({"error": "훈련 세션을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            db.execute(
                """
                INSERT INTO ai_reports(
                    session_id, summary, strengths_json, improvements_json,
                    next_training, coach_message, model, raw_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    strengths_json = excluded.strengths_json,
                    improvements_json = excluded.improvements_json,
                    next_training = excluded.next_training,
                    coach_message = excluded.coach_message,
                    model = excluded.model,
                    raw_json = excluded.raw_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id, summary,
                    json.dumps(strengths, ensure_ascii=False),
                    json.dumps(improvements, ensure_ascii=False),
                    next_training,
                    str(payload.get("coach_message", ""))[:2000],
                    str(payload.get("model", ""))[:100], raw_json, now, now,
                ),
            )
            db.commit()
        self.send_json({"session_id": session_id, "saved": True}, HTTPStatus.CREATED)

    def session_details(self, session_id: int) -> None:
        with connect(self.database) as db:
            session = db.execute("SELECT * FROM training_sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                self.send_json({"error": "훈련 세션을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            vision = db.execute("SELECT * FROM vision_results WHERE session_id = ? ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
            report = db.execute("SELECT * FROM ai_reports WHERE session_id = ?", (session_id,)).fetchone()
            punches = db.execute("SELECT * FROM punch_events WHERE session_id=? ORDER BY punch_index", (session_id,)).fetchall()
            force_rows = db.execute("SELECT * FROM force_hit_results WHERE session_id=? ORDER BY received_version", (session_id,)).fetchall()
            best = self._event_summary(db, session_id, "best")
            worst = self._event_summary(db, session_id, "worst")
            force_summary = self._force_summary(db, session_id)
            progress = self._progress_context(db, session)
        payload: dict[str, Any] = {
            "session": dict(session), "vision_result": dict(vision) if vision else None, "ai_report": dict(report) if report else None,
            "punch_events": [dict(row) for row in punches], "force_hits": [dict(row) for row in force_rows],
            "force_summary": force_summary, "progress": progress,
            "best_punch": compact_event(best), "check_point": compact_event(worst),
        }
        for section, key in ((payload.get("vision_result"), "representative_images_json"), (payload.get("ai_report"), "strengths_json"), (payload.get("ai_report"), "improvements_json")):
            if section and section.get(key):
                try:
                    section[key.removesuffix("_json")] = json.loads(section[key])
                except json.JSONDecodeError:
                    section[key.removesuffix("_json")] = []
        if payload.get("ai_report") and payload["ai_report"].get("raw_json"):
            try:
                payload["ai_report"]["raw"] = json.loads(payload["ai_report"]["raw_json"])
            except json.JSONDecodeError:
                payload["ai_report"]["raw"] = {}
        self.send_json(payload)

    def robot_command(self, payload: dict[str, Any]) -> None:
        command = str(payload.get("command", "")).strip()
        allowed = {
            "wakeword", "prepare", "start", "training_start", "training_go", "training_end",
            "pause", "resume", "stop", "home", "emergency_stop",
        }
        if command not in allowed:
            raise ValueError("지원하지 않는 명령입니다.")
        # The HTTP envelope is {command, payload}. Queue only the inner payload;
        # otherwise ROS receives a second nested {command, payload} object and
        # cannot validate height/reach/session fields.
        command_payload = payload.get("payload")
        if command_payload is not None and not isinstance(command_payload, dict):
            raise ValueError("로봇 명령 payload는 객체여야 합니다.")
        result = self.robot_hub.enqueue(
            command,
            source="ui",
            payload=command_payload if isinstance(command_payload, dict) else None,
        )
        self.send_json(result)


class KoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        database: Path,
        *,
        start_wakeword: bool = False,
        app_mode: str | None = None,
    ):
        self.database = database
        self.app_mode = normalize_app_mode(app_mode or os.environ.get("KO_APP_MODE", "user"))
        self.event_broker = EventBroker()
        self.vision_hub = VisionHub()
        self.force_hub = ForceHub()
        self.robot_hub = RobotHub()
        self.wakeword_service: WakeWordService | None = None
        init_db(database)
        super().__init__(server_address, KoRequestHandler)
        if start_wakeword:
            def handle_voice_event(event_type: str, payload: dict[str, Any]) -> None:
                self.event_broker.publish(event_type, payload)
                if event_type == "wake_detected":
                    # Wake detection is safety-critical for stopping the active
                    # weave, so it is queued server-side immediately.
                    self.robot_hub.enqueue("wakeword", source="wakeword", payload=payload)
                # Do not enqueue raw STT transcripts as robot training commands.
                # The browser owns voice parsing because it has the selected user,
                # measured reach, duration and combination context required to build
                # the fully validated training payload. Enqueuing here as well would
                # create a second, incomplete training_start.

            self.wakeword_service = WakeWordService(WAKEWORD_MODEL, handle_voice_event)
            self.wakeword_service.start()


def run(
    host: str = "0.0.0.0",
    port: int = 5000,
    database: Path = DEFAULT_DATABASE,
    *,
    start_wakeword: bool = True,
    app_mode: str | None = None,
) -> None:
    server = KoServer(
        (host, port),
        database,
        start_wakeword=start_wakeword,
        app_mode=app_mode,
    )
    print(f"KO UI running: http://localhost:{port}")
    print(f"UI mode: {server.app_mode.upper()}")
    print(f"웨이크업 단어: {WAKEWORD_DISPLAY_NAME} (openWakeWord 로컬 감지)")
    print("종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKO UI stopped")
    finally:
        if server.wakeword_service:
            server.wakeword_service.stop()
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KO AI Boxing Coach UI")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--app-mode",
        choices=("user", "admin"),
        default=normalize_app_mode(os.environ.get("KO_APP_MODE", "user")),
        help="user: 사용자/시연 화면, admin: 개발·진단 화면",
    )
    parser.add_argument("--no-wakeword", action="store_true", help="UI/DB 테스트 시 마이크와 openWakeWord를 시작하지 않습니다.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.host,
        args.port,
        args.database,
        start_wakeword=not args.no_wakeword,
        app_mode=args.app_mode,
    )

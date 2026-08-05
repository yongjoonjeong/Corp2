from __future__ import annotations

import argparse
import base64
import binascii
import json
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

from voice_processing.openai_transcriber import TranscriptionError, transcribe_audio_bytes
from voice_processing.wakeword_service import WakeWordService

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIR / "instance" / "ko.sqlite3"
INDEX_FILE = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"
ENV_FILE = BASE_DIR / ".env"
MAX_AUDIO_BYTES = 15 * 1024 * 1024
WAKEWORD_MODEL_DEFAULT = "wake_up_ko.tflite"


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
                created_at TEXT NOT NULL
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
        self._evidence: bytes | None = None
        self._evidence_type = "image/jpeg"
        self._evidence_version = 0
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

    def set_image(self, kind: str, data: bytes, content_type: str) -> int:
        self.touch()
        with self._lock:
            if kind == "preview":
                self._preview = data
                self._preview_type = content_type
                self._preview_version += 1
                return self._preview_version
            self._evidence = data
            self._evidence_type = content_type
            self._evidence_version += 1
            return self._evidence_version

    def image(self, kind: str) -> tuple[bytes | None, str, int]:
        with self._lock:
            if kind == "preview":
                return self._preview, self._preview_type, self._preview_version
            return self._evidence, self._evidence_type, self._evidence_version

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
                "evidence_available": self._evidence is not None,
                "evidence_version": self._evidence_version,
                "last_punch": dict(self._last_punch) if self._last_punch else None,
                "heartbeat": dict(self._heartbeat),
                "live_status": dict(self._live_status),
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
    def event_broker(self) -> EventBroker:
        return self.server.event_broker  # type: ignore[attr-defined]

    @property
    def vision_hub(self) -> VisionHub:
        return self.server.vision_hub  # type: ignore[attr-defined]

    @property
    def robot_hub(self) -> RobotHub:
        return self.server.robot_hub  # type: ignore[attr-defined]

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
        if path == "/api/health":
            self.send_json({"ok": True, "service": "ko-boxing-ui", "time": utc_now()})
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
            if path == "/api/vision/evidence":
                self.receive_vision_image(payload, "evidence")
                return
            if path == "/api/users":
                self.create_user(payload)
                return
            if path == "/api/sessions":
                self.save_session(payload)
                return
            if path == "/api/vision/results":
                self.save_vision_result(payload)
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
        version = self.vision_hub.set_image(kind, data, content_type)
        self.send_json({"ok": True, "version": version}, HTTPStatus.CREATED)

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
            display_path = str(self.database.relative_to(BASE_DIR)) if self.database.is_relative_to(BASE_DIR) else str(self.database)
            self.send_json({
                "ok": True,
                "engine": "SQLite",
                "path": display_path,
                "users": user_count,
                "sessions": session_count,
                "vision_results": vision_count,
                "ai_reports": report_count,
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
                    success_rate, avg_reaction_ms, posture_score, feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, training_type, hand, duration_sec, punch_count,
                    round(success_rate, 2), avg_reaction_ms, posture_score, feedback, now,
                ),
            )
            db.execute("UPDATE users SET last_training_at = ?, updated_at = ? WHERE id = ?", (now, now, user_id))
            db.commit()
        self.send_json({"id": int(cursor.lastrowid), "saved": True}, HTTPStatus.CREATED)

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
                    str(payload.get("next_training", ""))[:2000],
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
        payload: dict[str, Any] = {"session": dict(session), "vision_result": dict(vision) if vision else None, "ai_report": dict(report) if report else None}
        for section, key in ((payload.get("vision_result"), "representative_images_json"), (payload.get("ai_report"), "strengths_json"), (payload.get("ai_report"), "improvements_json")):
            if section and section.get(key):
                try:
                    section[key.removesuffix("_json")] = json.loads(section[key])
                except json.JSONDecodeError:
                    section[key.removesuffix("_json")] = []
        self.send_json(payload)

    def robot_command(self, payload: dict[str, Any]) -> None:
        command = str(payload.get("command", "")).strip()
        allowed = {
            "wakeword", "prepare", "start", "training_start",
            "pause", "resume", "stop", "home", "emergency_stop",
        }
        if command not in allowed:
            raise ValueError("지원하지 않는 명령입니다.")
        result = self.robot_hub.enqueue(command, source="ui", payload=payload)
        self.send_json(result)


class KoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        database: Path,
        *,
        start_wakeword: bool = False,
    ):
        self.database = database
        self.event_broker = EventBroker()
        self.vision_hub = VisionHub()
        self.robot_hub = RobotHub()
        self.wakeword_service: WakeWordService | None = None
        init_db(database)
        super().__init__(server_address, KoRequestHandler)
        if start_wakeword:
            def handle_voice_event(event_type: str, payload: dict[str, Any]) -> None:
                self.event_broker.publish(event_type, payload)
                if event_type == "wake_detected":
                    self.robot_hub.enqueue("wakeword", source="wakeword", payload=payload)
                elif event_type == "transcript":
                    transcript = str(payload.get("text", ""))
                    if is_training_voice_command(transcript):
                        self.robot_hub.enqueue(
                            "training_start",
                            source="stt",
                            payload={"text": transcript},
                        )

            self.wakeword_service = WakeWordService(WAKEWORD_MODEL, handle_voice_event)
            self.wakeword_service.start()


def run(host: str = "0.0.0.0", port: int = 5000, database: Path = DEFAULT_DATABASE, *, start_wakeword: bool = True) -> None:
    server = KoServer((host, port), database, start_wakeword=start_wakeword)
    print(f"KO UI running: http://localhost:{port}")
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
    parser.add_argument("--no-wakeword", action="store_true", help="UI/DB 테스트 시 마이크와 openWakeWord를 시작하지 않습니다.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.host, args.port, args.database, start_wakeword=not args.no_wakeword)

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import KoServer, is_training_voice_command


class KoApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        cls.db_path = Path(tmp.name)
        cls.server = KoServer(("127.0.0.1", 0), cls.db_path)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.db_path.unlink(missing_ok=True)

    def request(self, path, method="GET", payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def multipart_audio_request(self, audio=b"fake-audio-data" * 100):
        boundary = "----KoTestBoundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="command.webm"\r\n'
            "Content-Type: audio/webm\r\n\r\n"
        ).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.port}/api/transcribe",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health_and_user_flow(self):
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])

        status, stt = self.request("/api/stt/status")
        self.assertEqual(status, 200)
        self.assertEqual(stt["provider"], "OpenAI Audio Transcriptions")
        self.assertIn("configured", stt)

        status, wake = self.request("/api/wakeword/status")
        self.assertEqual(status, 200)
        self.assertEqual(wake["model"], "wake_up_ko.tflite")

        self.server.event_broker.publish("wake_detected", {"confidence": 0.8})
        status, events = self.request("/api/wakeword/events?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"][-1]["type"], "wake_detected")

        status, user = self.request(
            "/api/users",
            "POST",
            {"name": "테스트", "height_cm": 175, "dominant_hand": "right"},
        )
        self.assertEqual(status, 201)

        status, measured = self.request(
            f"/api/users/{user['id']}/measurement",
            "PATCH",
            {
                "wingspan_cm": 177,
                "left_punch_reach_cm": 70,
                "right_punch_reach_cm": 72,
                "recommended_distance_cm": 107,
                "measurement_confidence": 0.9,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(measured["wingspan_cm"], 177)

        status, saved = self.request(
            "/api/sessions",
            "POST",
            {
                "user_id": user["id"],
                "training_type": "straight",
                "hand": "right",
                "duration_sec": 30,
                "punch_count": 12,
                "success_rate": 80,
                "avg_reaction_ms": 620,
                "posture_score": 78,
                "feedback": "좋습니다.",
            },
        )
        self.assertEqual(status, 201)
        self.assertTrue(saved["saved"])

        status, sessions = self.request(f"/api/users/{user['id']}/sessions")
        self.assertEqual(status, 200)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["punch_count"], 12)

        status, db_status = self.request("/api/database/status")
        self.assertEqual(status, 200)
        self.assertTrue(db_status["ok"])
        self.assertGreaterEqual(db_status["users"], 1)
        self.assertGreaterEqual(db_status["sessions"], 1)

        session_id = saved["id"]
        status, vision = self.request(
            "/api/vision/results",
            "POST",
            {
                "session_id": session_id,
                "total_punches": 12,
                "successful_punches": 10,
                "accuracy_percent": 83.3,
                "average_reaction_sec": 0.62,
                "guard_drop_count": 2,
                "arm_extension_score": 86,
                "representative_images": ["images/best.jpg"],
            },
        )
        self.assertEqual(status, 201)
        self.assertTrue(vision["saved"])

        status, report = self.request(
            "/api/ai/reports",
            "POST",
            {
                "session_id": session_id,
                "summary": "정확한 타격이 좋았습니다.",
                "strengths": ["팔 신전"],
                "improvements": ["가드 복귀"],
                "next_training": "가드 유지 스트레이트",
                "coach_message": "좋은 훈련이었습니다.",
                "model": "test-model",
            },
        )
        self.assertEqual(status, 201)
        self.assertTrue(report["saved"])

        status, details = self.request(f"/api/sessions/{session_id}/details")
        self.assertEqual(status, 200)
        self.assertEqual(details["vision_result"]["successful_punches"], 10.0)
        self.assertEqual(details["ai_report"]["strengths"], ["팔 신전"])


    def test_live_vision_bridge_endpoints(self):
        status, heartbeat = self.request(
            "/api/vision/heartbeat",
            "POST",
            {"node": "test-bridge"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(heartbeat["ok"])

        status, live = self.request(
            "/api/vision/status_update",
            "POST",
            {"pose_detected": True, "centered": True, "detector_state": "READY"},
        )
        self.assertEqual(status, 200)

        status, punch = self.request(
            "/api/vision/punch",
            "POST",
            {
                "punch_id": 1,
                "punch_type": "straight",
                "punch_side": "right",
                "total_score": 84.2,
                "passed": True,
                "violations": [],
            },
        )
        self.assertEqual(status, 201)
        self.assertGreater(punch["event_id"], 0)

        status, vision_status = self.request("/api/vision/status")
        self.assertEqual(status, 200)
        self.assertTrue(vision_status["connected"])
        self.assertEqual(vision_status["live_status"]["detector_state"], "READY")

        status, events = self.request("/api/vision/events?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"][-1]["type"], "punch")

    def test_transcription_with_mocked_openai(self):
        class FakeOpenAIResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"text": "웨이크 업 케이오 오른손 스트레이트 시작"}).encode("utf-8")

        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
        try:
            with patch("voice_processing.openai_transcriber.urlopen", return_value=FakeOpenAIResponse()) as mocked:
                status, result = self.multipart_audio_request()
            self.assertEqual(status, 200)
            self.assertIn("오른손 스트레이트", result["text"])
            outbound_request = mocked.call_args.args[0]
            self.assertTrue(outbound_request.full_url.endswith("/audio/transcriptions"))
            self.assertNotIn("sk-test-key-not-real", outbound_request.data.decode("latin1", errors="ignore"))
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key


    def test_robot_command_queue(self):
        status, queued = self.request(
            "/api/robot/command",
            "POST",
            {"command": "wakeword"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(queued["accepted"])

        status, commands = self.request("/api/robot/commands?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(commands["events"][-1]["payload"]["command"], "wakeword")

        status, updated = self.request(
            "/api/robot/status_update",
            "POST",
            {"state": "WEAVING", "message": "위빙 중"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(updated["ok"])

        status, robot_status = self.request("/api/robot/status")
        self.assertEqual(status, 200)
        self.assertTrue(robot_status["connected"])
        self.assertEqual(robot_status["state"], "WEAVING")


    def test_training_voice_intent(self):
        self.assertTrue(is_training_voice_command("오른손 스트레이트 1분 훈련"))
        self.assertTrue(is_training_voice_command("왼손 잽 연습할게"))
        self.assertFalse(is_training_voice_command("스트레이트가 뭐야"))

    def test_validation(self):
        with self.assertRaises(HTTPError) as ctx:
            self.request(
                "/api/users",
                "POST",
                {"name": "", "height_cm": 20, "dominant_hand": "right"},
            )
        self.assertEqual(ctx.exception.code, 400)

        with self.assertRaises(HTTPError) as ctx:
            self.request("/api/robot/command", "POST", {"command": "move_random"})
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class VoiceSessionStateTest(unittest.TestCase):
    def test_continuous_voice_session_lifecycle(self):
        from voice_processing.wakeword_service import WakeWordService

        events = []
        service = WakeWordService(Path("/tmp/not-started-wakeword.tflite"), lambda event_type, payload: events.append((event_type, payload)))

        started = service.start_session(30, source="test")
        self.assertTrue(started["session_active"])
        self.assertGreater(started["session_remaining_sec"], 0)

        extended = service.extend_session(120, source="test")
        self.assertTrue(extended["session_active"])
        self.assertGreaterEqual(extended["session_remaining_sec"], 119)

        ended = service.end_session("test_complete")
        self.assertFalse(ended["session_active"])
        self.assertEqual(ended["state"], "waiting_wakeword")

        event_types = [event_type for event_type, _ in events]
        self.assertIn("session_started", event_types)
        self.assertIn("session_extended", event_types)
        self.assertIn("session_ended", event_types)

    def test_voice_session_exit_phrases(self):
        from voice_processing.wakeword_service import WakeWordService

        self.assertTrue(WakeWordService._is_exit_command("대화 종료"))
        self.assertTrue(WakeWordService._is_exit_command("음성 대기 모드로 돌아가"))
        self.assertFalse(WakeWordService._is_exit_command("오른손 스트레이트 1분 훈련"))

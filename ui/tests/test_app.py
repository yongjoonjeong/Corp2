from __future__ import annotations

import base64
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
from vision_coach import analyze_boxing_images, select_representative_images


UI_ROOT = Path(__file__).resolve().parents[1]
INTEGRATED_ROOT = UI_ROOT.parent


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

    def test_local_pose_assets_are_served_with_browser_compatible_types(self):
        assets = {
            "/static/vendor/mediapipe/tasks-vision-0.10.14/vision_bundle.mjs": "javascript",
            "/static/vendor/mediapipe/tasks-vision-0.10.14/wasm/vision_wasm_internal.wasm": "application/wasm",
            "/static/vendor/mediapipe/tasks-vision-0.10.14/wasm/vision_wasm_nosimd_internal.wasm": "application/wasm",
            "/static/vendor/mediapipe/models/pose_landmarker_lite.task": "application/octet-stream",
        }
        for path, expected_type in assets.items():
            with self.subTest(path=path):
                with urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(expected_type, response.headers.get("Content-Type", ""))
                    self.assertGreater(int(response.headers.get("Content-Length", "0")), 1000)
                    self.assertGreater(len(response.read()), 1000)

    def test_health_and_user_flow(self):
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(health["mode"], "user")

        status, app_config = self.request("/api/app/config")
        self.assertEqual(status, 200)
        self.assertEqual(app_config["mode"], "user")
        self.assertFalse(app_config["is_admin"])
        self.assertFalse(app_config["show_system_settings"])
        self.assertEqual(app_config["voice_activation_policy"], "one_command_per_wake")
        self.assertEqual(
            app_config["robot_supported_punches"],
            ["jab", "straight", "hook", "uppercut"],
        )

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
                "next_training": {
                    "title": "가드 유지 스트레이트",
                    "duration_sec": 60,
                    "goal": "가드 복귀 안정화",
                },
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
        self.assertEqual(
            json.loads(details["ai_report"]["next_training"])["title"],
            "가드 유지 스트레이트",
        )
        self.assertIn("progress", details)


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
            {
                "pose_detected": True,
                "centered": True,
                "detector_state": "READY",
                "target_locked": True,
                "sync_spread_ms": 12.4,
                "mean_reprojection_error_px": 3.2,
                "mitt_tracker": {
                    "state": "TRACKED",
                    "roi_normalized": [0.68, 0.68, 0.93, 0.90],
                },
                "preview_layout": {
                    "canvas_width": 1440,
                    "canvas_height": 464,
                    "front_tile_xywh": [480, 0, 480, 360],
                },
            },
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
                "violations": [
                    {
                        "joint": "strike_wrist",
                        "code": "straight_path_not_linear",
                        "error_ratio": 1.1,
                    }
                ],
                "impact_point": {
                    "robot_base_mm": {"x": 412.0, "y": -85.0, "z": 920.0}
                },
                "quality": {"impact_sync_spread_ms": 12.4},
            },
        )
        self.assertEqual(status, 201)
        self.assertGreater(punch["event_id"], 0)

        status, vision_status = self.request("/api/vision/status")
        self.assertEqual(status, 200)
        self.assertTrue(vision_status["connected"])
        self.assertEqual(vision_status["live_status"]["detector_state"], "READY")
        self.assertTrue(vision_status["live_status"]["target_locked"])
        self.assertEqual(
            vision_status["live_status"]["mitt_tracker"]["roi_normalized"],
            [0.68, 0.68, 0.93, 0.90],
        )
        self.assertEqual(
            vision_status["live_status"]["preview_layout"]["front_tile_xywh"],
            [480, 0, 480, 360],
        )

        status, events = self.request("/api/vision/events?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"][-1]["type"], "punch")
        punch_payload = events["events"][-1]["payload"]
        self.assertEqual(
            punch_payload["violations"][0]["code"],
            "straight_path_not_linear",
        )
        self.assertEqual(
            punch_payload["impact_point"]["robot_base_mm"]["x"],
            412.0,
        )

        preview_bytes = b"\xff\xd8three-camera-preview\xff\xd9"
        status, preview = self.request(
            "/api/vision/preview",
            "POST",
            {
                "format": "jpeg",
                "frame_id": "front_realsense_color_optical_frame",
                "data_base64": base64.b64encode(preview_bytes).decode("ascii"),
            },
        )
        self.assertEqual(status, 201)
        self.assertGreater(preview["version"], 0)
        with urlopen(
            f"http://127.0.0.1:{self.port}/api/vision/preview.jpg",
            timeout=3,
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("image/jpeg", response.headers.get("Content-Type", ""))
            self.assertEqual(response.read(), preview_bytes)

        front_bytes = b"\xff\xd8front-realsense-shared-frame\xff\xd9"
        status, front = self.request(
            "/api/vision/front",
            "POST",
            {
                "format": "jpeg",
                "frame_id": "front_realsense_color_optical_frame",
                "data_base64": base64.b64encode(front_bytes).decode("ascii"),
            },
        )
        self.assertEqual(status, 201)
        self.assertGreater(front["version"], 0)
        with urlopen(
            f"http://127.0.0.1:{self.port}/api/vision/front.jpg",
            timeout=3,
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("image/jpeg", response.headers.get("Content-Type", ""))
            self.assertEqual(response.read(), front_bytes)
        _, vision_status = self.request("/api/vision/status")
        self.assertTrue(vision_status["front_available"])
        self.assertEqual(vision_status["front_version"], front["version"])

    def test_openai_vision_coach_updates_session_feedback(self):
        _, user = self.request(
            "/api/users",
            "POST",
            {"name": "이미지코치", "height_cm": 175, "dominant_hand": "right"},
        )
        _, session = self.request(
            "/api/sessions",
            "POST",
            {
                "user_id": user["id"],
                "training_type": "straight",
                "hand": "right",
                "duration_sec": 30,
                "punch_count": 3,
                "success_rate": 100,
                "feedback": "기존 로컬 문구",
            },
        )
        baseline = self.server.vision_hub.status()["evidence_version"]
        self.server.vision_hub.set_image("evidence", b"first-triptych", "image/jpeg")
        self.server.vision_hub.set_image("evidence", b"last-triptych", "image/jpeg")
        mocked_result = {
            "coach_message": "팔 신전은 안정적이며, 다음 타격에서는 반대손 가드를 턱 가까이 유지하세요.",
            "observed_strength": "팔 신전이 안정적임",
            "improvement": "반대손 가드를 턱 가까이 유지",
            "next_focus": "가드 유지 스트레이트",
            "visual_confidence": "medium",
            "model": "gpt-5.6-luna",
            "image_count": 2,
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-not-real"}):
            with patch("app.analyze_boxing_images", return_value=mocked_result) as analyze:
                status, result = self.request(
                    "/api/ai/vision-coach",
                    "POST",
                    {
                        "session_id": session["id"],
                        "after_evidence_version": baseline,
                        "fallback_feedback": "기존 로컬 문구",
                        "metrics": {"score": 92},
                    },
                )
        self.assertEqual(status, 200)
        self.assertTrue(result["used_ai"])
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(len(analyze.call_args.args[0]), 2)

        _, details = self.request(f"/api/sessions/{session['id']}/details")
        self.assertEqual(details["session"]["feedback"], mocked_result["coach_message"])
        self.assertEqual(details["ai_report"]["coach_message"], mocked_result["coach_message"])

    def test_vision_coach_falls_back_when_api_key_is_missing(self):
        _, user = self.request(
            "/api/users",
            "POST",
            {"name": "키없음", "height_cm": 170, "dominant_hand": "left"},
        )
        _, session = self.request(
            "/api/sessions",
            "POST",
            {
                "user_id": user["id"],
                "training_type": "straight",
                "hand": "left",
                "duration_sec": 10,
                "punch_count": 1,
                "success_rate": 100,
                "feedback": "로컬 피드백",
            },
        )
        previous = os.environ.pop("OPENAI_API_KEY", None)
        try:
            status, result = self.request(
                "/api/ai/vision-coach",
                "POST",
                {
                    "session_id": session["id"],
                    "after_evidence_version": 0,
                    "fallback_feedback": "로컬 피드백",
                    "metrics": {},
                },
            )
        finally:
            if previous is not None:
                os.environ["OPENAI_API_KEY"] = previous
        self.assertEqual(status, 200)
        self.assertFalse(result["used_ai"])
        self.assertEqual(result["reason"], "api_key_missing")
        self.assertEqual(result["coach_message"], "로컬 피드백")

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


    def test_training_robot_command_payload_is_not_double_nested(self):
        training = {
            "client_session_id": "session-contract",
            "user_id": 9,
            "dominant_hand": "right",
            "height_cm": 173,
            "left_punch_reach_cm": 65.0,
            "right_punch_reach_cm": 67.0,
            "training_type": "straight",
            "hand": "right",
            "duration_sec": 60,
        }
        status, queued = self.request(
            "/api/robot/command",
            "POST",
            {"command": "training_start", "payload": training},
        )
        self.assertEqual(status, 200)
        status, commands = self.request("/api/robot/commands?after=0")
        self.assertEqual(status, 200)
        event_payload = commands["events"][-1]["payload"]
        self.assertEqual(event_payload["command"], "training_start")
        self.assertEqual(event_payload["payload"], training)
        self.assertNotIn("command", event_payload["payload"])

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


class VisionCoachClientTest(unittest.TestCase):
    def test_representative_images_are_evenly_spaced(self):
        images = [(index, f"image-{index}".encode(), "image/jpeg") for index in range(1, 11)]
        selected = select_representative_images(images, 3)
        self.assertEqual([item[0] for item in selected], [1, 5, 10])

    def test_responses_request_contains_images_and_structured_output(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                result = {
                    "coach_message": "팔은 충분히 뻗었으며, 다음에는 반대손 가드를 턱 옆에 유지하세요.",
                    "observed_strength": "충분한 팔 신전",
                    "improvement": "반대손 가드 유지",
                    "next_focus": "가드 유지",
                    "visual_confidence": "high",
                }
                return json.dumps({
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result, ensure_ascii=False)}],
                    }]
                }).encode("utf-8")

        with patch("vision_coach.urlopen", return_value=FakeResponse()) as mocked:
            result = analyze_boxing_images(
                [(1, b"jpeg-one", "image/jpeg"), (2, b"jpeg-two", "image/jpeg")],
                {"punch_count": 2, "hand": "right"},
                api_key="sk-test-not-real",
            )
        self.assertIn("반대손 가드", result["coach_message"])
        request_payload = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
        self.assertFalse(request_payload["store"])
        self.assertEqual(request_payload["text"]["format"]["type"], "json_schema")
        image_inputs = [
            item for item in request_payload["input"][1]["content"]
            if item["type"] == "input_image"
        ]
        self.assertEqual(len(image_inputs), 2)
        self.assertTrue(image_inputs[0]["image_url"].startswith("data:image/jpeg;base64,"))


class CurrentVisionIntegrationContractTest(unittest.TestCase):
    def test_latest_3d_feedback_codes_have_korean_messages(self):
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        expected_codes = {
            "straight_forward_path_off",
            "straight_path_not_linear",
            "hook_lateral_path_off",
            "hook_curve_off",
            "uppercut_upward_path_off",
        }
        for code in expected_codes:
            with self.subTest(code=code):
                self.assertIn(f"{code}:", script)

    def test_evidence_uploads_are_never_dropped_while_another_upload_is_busy(self):
        bridge = (UI_ROOT / "vision_bridge.py").read_text(encoding="utf-8")
        evidence_handler = bridge.split("def on_evidence", 1)[1].split("def on_preview", 1)[0]
        self.assertIn("drop_if_busy=False", evidence_handler)

    def test_training_ui_identifies_the_current_ekf_base_telemetry(self):
        template = (UI_ROOT / "templates/index.html").read_text(encoding="utf-8")
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("6-STATE EKF · ROBOT BASE", template)
        self.assertIn("leftFistPosition", template)
        self.assertIn("rightFistVelocity", template)
        self.assertIn("webGuardFill", template)
        self.assertLess(template.index("webGuardFill"), template.index('id="liveCommand"'))
        self.assertIn('id="visionTargetZone"', template)
        self.assertIn("live.mitt_tracker?.roi_normalized", script)
        self.assertIn('setVisionPunchBadge("active", "ACTIVE")', script)
        self.assertIn('setVisionPunchBadge("cooldown", "COOLDOWN")', script)
        self.assertNotIn("YOLO 3D COACH", template)

    def test_reach_measurement_uses_the_shared_front_realsense_frame(self):
        template = (UI_ROOT / "templates/index.html").read_text(encoding="utf-8")
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn('id="measurementVisionPreview"', template)
        self.assertIn("startSharedFrontPoseLoop", script)
        self.assertIn("/api/vision/front.jpg", script)
        self.assertIn("state.vision.frontAvailable", script)

    def test_training_feed_is_mode_specific_admin_triptych_user_front(self):
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        preferred = script.split("function preferredVisionFeed()", 1)[1].split(
            "function refreshVisionPreview()", 1
        )[0]
        self.assertIn('state.appMode === "admin"', preferred)
        admin_block = preferred.split('state.appMode === "admin"', 1)[1].split('// USER MODE:', 1)[0]
        user_block = preferred.split('// USER MODE:', 1)[1]
        self.assertLess(admin_block.index("state.vision.previewAvailable"), admin_block.index("state.vision.frontAvailable"))
        self.assertIn('path: "/api/vision/preview.jpg"', admin_block)
        self.assertIn('path: "/api/vision/front.jpg"', user_block)
        self.assertNotIn('path: "/api/vision/preview.jpg"', user_block)
        self.assertIn('preview.dataset.visionSource === "front"', script)

    def test_live_target_is_larger_and_impact_pulse_is_stronger(self):
        css = (UI_ROOT / "static/css/app.css").read_text(encoding="utf-8")
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("width: 114px; height: 114px", css)
        self.assertIn("border: 3px solid rgba(105,205,239,.96)", css)
        self.assertIn('transform: "scale(1.52)"', script)
        self.assertIn("duration: success ? 620 : 260", script)
        self.assertIn('filter: "brightness(1.9)"', script)
        self.assertIn("function playImpactSound()", script)
        self.assertIn("context.createOscillator()", script)
        self.assertIn("if (success) playImpactSound();", script)

    def test_vision_bridge_subscribes_to_the_stable_ui_topics(self):
        bridge = (UI_ROOT / "vision_bridge.py").read_text(encoding="utf-8")
        for topic in (
            "/sandbag/vision/status",
            "/sandbag/fist_state",
            "/sandbag/impact_event",
            "/sandbag/impact_feedback_image/compressed",
            "/sandbag/vision/preview/compressed",
            "/sandbag/vision/front/compressed",
        ):
            with self.subTest(topic=topic):
                self.assertIn(f'"{topic}"', bridge)
        for forbidden in ("import cv2", "ultralytics", "mediapipe"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, bridge.lower())

    def test_phase1_runtime_modes_are_single_codebase_switches(self):
        launcher = (INTEGRATED_ROOT / "run_integrated.sh").read_text(encoding="utf-8")
        html = (UI_ROOT / "templates/index.html").read_text(encoding="utf-8")
        script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("--user-mode", launcher)
        self.assertIn("--admin-mode", launcher)
        self.assertIn('export KO_APP_MODE="$app_mode"', launcher)
        self.assertIn('class="sport-nav-item admin-only" data-screen="settings"', html)
        self.assertIn('id="userVisionStatusTitle"', html)
        self.assertIn('id="contextVoiceHelp"', html)
        self.assertIn("function computeVisionFrameHealth", script)
        self.assertIn("function updateContextVoiceHelp", script)
        self.assertIn('state.appMode !== "admin"', script)

    def test_integrated_runner_launches_only_the_current_vision_and_bridge(self):
        launcher = (INTEGRATED_ROOT / "run_integrated.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('start_child "$project_dir/run_vision.sh" "${vision_args[@]}"', launcher)
        self.assertIn('start_child env KO_UI_BASE_URL="$ui_url" "$project_dir/run_ui_bridge.sh"', launcher)
        self.assertIn('start_child "$project_dir/ui/run_ui.sh"', launcher)
        self.assertIn('kill -TERM -- "-$pgid"', launcher)
        self.assertIn('kill -KILL -- "-$pgid"', launcher)
        self.assertIn("trap 'exit 129' HUP", launcher)
        self.assertNotIn("run_ros_3d_mvp.sh", launcher)
        self.assertNotIn("optional/vision_processing", launcher)

    def test_integrated_runner_restores_the_zip_weaving_stack_without_zip_vision(self):
        launcher = (INTEGRATED_ROOT / "run_integrated.sh").read_text(encoding="utf-8")
        robot_node = (INTEGRATED_ROOT / "robot_control/robot_weaving_node.py").read_text(
            encoding="utf-8"
        )
        robot_bridge = (INTEGRATED_ROOT / "robot_control/ui_robot_bridge.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("robot_control/run_robot_node.sh", launcher)
        self.assertIn("robot_control/run_bridge.sh", launcher)
        self.assertIn("/dsr01/motion/move_stop", launcher)
        self.assertIn("AUTO_START_WEAVING_ON_STARTUP", robot_node)
        self.assertIn("WEAVE_READY_J_DEG", robot_node)
        self.assertIn('"/robot_boxing/weave_command"', robot_bridge)
        self.assertNotIn("optional/vision_processing", launcher)

    def test_integrated_runtime_uses_the_local_ui_contract(self):
        launcher = (INTEGRATED_ROOT / "run_integrated.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export HOST="127.0.0.1"', launcher)
        self.assertIn('export PORT="5000"', launcher)
        self.assertIn('export KO_UI_BASE_URL="http://$HOST:$PORT"', launcher)
        self.assertIn("/api/health", launcher)
        self.assertIn("KO_UI_BASE_URL", launcher)
        self.assertIn("현재 sandbag_vision/node.py", launcher)

    def test_current_node_publishes_the_web_preview_topic(self):
        node = (INTEGRATED_ROOT / "sandbag_vision/node.py").read_text(encoding="utf-8")
        runtime = (INTEGRATED_ROOT / "config/runtime.yaml").read_text(encoding="utf-8")
        self.assertIn("self.preview_publisher", node)
        self.assertIn("preview_image_topic", node)
        self.assertIn("preview_image_topic: /sandbag/vision/preview/compressed", runtime)
        self.assertIn("preview_publish_hz: 8.0", runtime)
        self.assertIn("self.front_preview_publisher", node)
        self.assertIn("def _publish_front_preview", node)
        self.assertIn("front_preview_image_topic: /sandbag/vision/front/compressed", runtime)
        self.assertIn("front_preview_publish_hz: 10.0", runtime)


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

    def test_followup_wakeword_variants(self):
        from voice_processing.wakeword_service import WakeWordService

        self.assertTrue(WakeWordService._is_followup_wake("케이오"))
        self.assertTrue(WakeWordService._is_followup_wake("케이 오!"))
        self.assertTrue(WakeWordService._is_followup_wake("K.O."))
        self.assertFalse(WakeWordService._is_followup_wake("웨이크 업 케이오"))
        self.assertFalse(WakeWordService._is_followup_wake("오른손 스트레이트"))

    def test_display_name_changes_only_after_initial_full_wake(self):
        from voice_processing.wakeword_service import WakeWordService

        service = WakeWordService(Path("/tmp/not-started-wakeword.tflite"), lambda *_: None)
        self.assertEqual(service.status()["display_name"], "웨이크 업 케이오")
        self.assertFalse(service.status()["initial_wake_completed"])

        service._mark_initial_wake_completed()

        self.assertEqual(service.status()["display_name"], "케이오")
        self.assertTrue(service.status()["initial_wake_completed"])

    def test_followup_wake_transcription_does_not_emit_a_command(self):
        from unittest.mock import patch
        from voice_processing.wakeword_service import WakeWordService

        events = []
        service = WakeWordService(
            Path("/tmp/not-started-wakeword.tflite"),
            lambda event_type, payload: events.append((event_type, payload)),
        )
        service._mark_initial_wake_completed()
        with patch(
            "voice_processing.wakeword_service.transcribe_audio_bytes",
            return_value={"text": "케이오", "model": "test"},
        ):
            transcript = service._transcribe_followup_wake(b"audio")

        self.assertEqual(transcript, "케이오")
        self.assertNotIn("transcript", [event_type for event_type, _ in events])

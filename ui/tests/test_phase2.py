from __future__ import annotations

import base64
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from app import ForceHub, KoServer
from reporting import build_progress, event_score, infer_tracked_metric, select_best_worst


class ReportingUnitTest(unittest.TestCase):
    def test_progress_direction_is_deterministic(self):
        progress = build_progress(
            {"accuracy_percent": 70, "average_reaction_sec": 0.60},
            {"accuracy_percent": 82, "average_reaction_sec": 0.51},
            "accuracy_percent",
        )
        self.assertTrue(progress["has_previous"])
        self.assertEqual(progress["tracked_result"]["status"], "improved")
        reaction = next(x for x in progress["comparisons"] if x["metric_key"] == "average_reaction_sec")
        self.assertEqual(reaction["status"], "improved")

    def test_best_worst_uses_event_score(self):
        events = [
            {"punch_index": 1, "event_score": 62},
            {"punch_index": 2, "event_score": 91},
            {"punch_index": 3, "event_score": 45},
        ]
        best, worst = select_best_worst(events)
        self.assertEqual(best["punch_index"], 2)
        self.assertEqual(worst["punch_index"], 3)

    def test_force_accuracy_can_contribute_without_using_unfinished_power(self):
        event = {"total_score": 80, "passed": True, "violations": []}
        score = event_score(event, {"valid_hit": True, "accuracy_score": 40, "power_score": 999})
        self.assertAlmostEqual(score, 66.0)

    def test_legacy_guard_feedback_maps_only_to_available_metric(self):
        key = infer_tracked_metric(
            "왼손 가드를 얼굴 높이에 유지하세요.",
            {"guard_error_rate": 30, "accuracy_percent": 70},
            {"guard_error_rate": 10, "accuracy_percent": 80},
        )
        self.assertEqual(key, "guard_error_rate")


class ForceHubUnitTest(unittest.TestCase):
    def test_duplicate_session_hit_is_idempotent(self):
        hub = ForceHub()
        payload = {
            "client_session_id": "session-a",
            "hit_id": 7,
            "stamp_ns": 123456789,
            "valid_hit": True,
        }
        first = hub.publish(payload)
        second = hub.publish(dict(payload))
        self.assertEqual(first["version"], 1)
        self.assertEqual(second["version"], 1)
        self.assertEqual(hub.status()["version"], 1)
        self.assertEqual(len(hub.after(0)), 1)


class Phase2ApiTest(unittest.TestCase):
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

    def request(self, path: str, method: str = "GET", payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
            if "application/json" in content_type:
                return response.status, json.loads(raw.decode())
            return response.status, raw

    def create_session(self, user_id: int, accuracy: float, reaction: float, feedback: str = "") -> int:
        _, session = self.request("/api/sessions", "POST", {
            "user_id": user_id, "training_type": "straight", "hand": "right", "duration_sec": 30,
            "punch_count": 2, "success_rate": accuracy, "avg_reaction_ms": reaction * 1000,
            "posture_score": 80, "feedback": feedback,
        })
        self.request("/api/vision/results", "POST", {
            "session_id": session["id"], "total_punches": 2, "successful_punches": 2,
            "accuracy_percent": accuracy, "average_reaction_sec": reaction,
            "guard_drop_count": 1 if accuracy < 80 else 0, "torso_balance_score": 80,
            "representative_images": [],
        })
        return session["id"]

    def test_sessionize_best_worst_force_and_progress(self):
        _, user = self.request("/api/users", "POST", {"name": "phase2", "height_cm": 175, "dominant_hand": "right"})
        previous_id = self.create_session(user["id"], 70, 0.60, "왼손 가드를 더 안정적으로 유지하세요.")
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO feedback_goals(user_id,source_session_id,metric_key,metric_label,baseline_value,advice,created_at) VALUES(?,?,?,?,?,?,?)",
                (user["id"], previous_id, "accuracy_percent", "타격 정확도", 70, "정확도를 유지하세요.", "2026-08-07T00:00:00Z"),
            )
            db.commit()

        current_id = self.create_session(user["id"], 85, 0.50)
        _, force0 = self.request("/api/force/status")
        force_baseline = int(force0["version"])
        self.request("/api/force/hit", "POST", {
            "stamp_ns": 1_000_000_000, "hit_id": 1, "valid_hit": True, "accuracy_score": 90,
            "peak_force_n": 42, "center_error_mm": 8, "power_score": 0, "safety_stop": False,
        })
        self.request("/api/force/hit", "POST", {
            "stamp_ns": 2_000_000_000, "hit_id": 2, "valid_hit": True, "accuracy_score": 40,
            "peak_force_n": 35, "center_error_mm": 20, "power_score": 0, "safety_stop": False,
        })
        image = base64.b64encode(b"\xff\xd8" + b"x" * 2048 + b"\xff\xd9").decode()
        for impact_id, stamp in ((11, 1_000_000_000), (12, 2_000_000_000)):
            self.request("/api/vision/evidence", "POST", {
                "format": "jpeg", "impact_id": impact_id, "impact_stamp_ns": stamp,
                "data_base64": image,
            })

        _, sessionized = self.request("/api/punch-events/sessionize", "POST", {
            "session_id": current_id, "after_evidence_version": 0, "after_force_version": force_baseline,
            "punch_events": [
                {"punch_id": 11, "punch_side": "right", "punch_type": "impact", "total_score": 92, "passed": True, "violations": [], "raw_event": {"impact_stamp_ns": 1_000_000_000}},
                {"punch_id": 12, "punch_side": "right", "punch_type": "impact", "total_score": 55, "passed": True, "violations": [{"code": "guard_dropped"}], "raw_event": {"impact_stamp_ns": 2_000_000_000}},
            ],
        })
        self.assertEqual(sessionized["count"], 2)
        self.assertEqual(sessionized["best"]["punch_index"], 1)
        self.assertEqual(sessionized["worst"]["punch_index"], 2)
        self.assertEqual(sessionized["force_count"], 2)

        status, raw = self.request(sessionized["best"]["evidence_url"])
        self.assertEqual(status, 200)
        self.assertGreater(len(raw), 1000)

        _, coach = self.request("/api/ai/vision-coach", "POST", {
            "session_id": current_id, "after_evidence_version": 0, "expected_image_count": 2,
            "fallback_feedback": "로컬 피드백",
            "metrics": {"score": 85},
        })
        # No API key in test; progress must still be computed locally.
        self.assertFalse(coach["used_ai"])
        self.assertTrue(coach["progress"]["has_previous"])
        self.assertEqual(coach["progress"]["tracked_result"]["status"], "improved")
        self.assertEqual(coach["best"]["punch_index"], 1)
        self.assertEqual(coach["worst"]["punch_index"], 2)

        _, details = self.request(f"/api/sessions/{current_id}/details")
        self.assertEqual(len(details["punch_events"]), 2)
        self.assertEqual(details["force_summary"]["hit_count"], 2)
        self.assertEqual(details["best_punch"]["punch_index"], 1)
        self.assertEqual(details["check_point"]["punch_index"], 2)

    def test_force_hits_are_isolated_by_client_session_id(self):
        _, user = self.request("/api/users", "POST", {"name": "force-isolation", "height_cm": 173, "dominant_hand": "right"})
        _, session_a = self.request("/api/sessions", "POST", {
            "user_id": user["id"], "training_type": "straight", "hand": "right",
            "duration_sec": 30, "client_session_id": "session-a",
        })
        _, session_b = self.request("/api/sessions", "POST", {
            "user_id": user["id"], "training_type": "straight", "hand": "right",
            "duration_sec": 30, "client_session_id": "session-b",
        })
        _, before = self.request("/api/force/status")
        after_version = int(before["version"])
        self.request("/api/force/hit", "POST", {
            "client_session_id": "session-a", "stamp_ns": 101, "hit_id": 1, "valid_hit": True,
        })
        self.request("/api/force/hit", "POST", {
            "client_session_id": "session-b", "stamp_ns": 202, "hit_id": 1, "valid_hit": True,
        })
        # A legacy/invalid hit without a UUID must not leak into a UUID-bound session.
        self.request("/api/force/hit", "POST", {
            "stamp_ns": 303, "hit_id": 99, "valid_hit": True,
        })
        _, result_a = self.request("/api/punch-events/sessionize", "POST", {
            "session_id": session_a["id"], "after_evidence_version": 0,
            "after_force_version": after_version, "punch_events": [],
        })
        _, result_b = self.request("/api/punch-events/sessionize", "POST", {
            "session_id": session_b["id"], "after_evidence_version": 0,
            "after_force_version": after_version, "punch_events": [],
        })
        self.assertEqual(result_a["force_count"], 1)
        self.assertEqual(result_b["force_count"], 1)
        with sqlite3.connect(self.db_path) as db:
            rows_a = db.execute("SELECT raw_json FROM force_hit_results WHERE session_id=?", (session_a["id"],)).fetchall()
            rows_b = db.execute("SELECT raw_json FROM force_hit_results WHERE session_id=?", (session_b["id"],)).fetchall()
        self.assertEqual(len(rows_a), 1)
        self.assertEqual(len(rows_b), 1)
        self.assertEqual(json.loads(rows_a[0][0])["client_session_id"], "session-a")
        self.assertEqual(json.loads(rows_b[0][0])["client_session_id"], "session-b")


if __name__ == "__main__":
    unittest.main()

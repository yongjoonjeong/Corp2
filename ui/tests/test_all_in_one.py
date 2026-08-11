from __future__ import annotations

import unittest
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UI_ROOT.parent


class AllInOneContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (UI_ROOT / "static/js/app.js").read_text(encoding="utf-8")
        cls.template = (UI_ROOT / "templates/index.html").read_text(encoding="utf-8")
        cls.launcher = (PROJECT_ROOT / "run_integrated.sh").read_text(encoding="utf-8")
        cls.setup = (PROJECT_ROOT / "setup.sh").read_text(encoding="utf-8")
        cls.robot_bridge = (PROJECT_ROOT / "robot_control/ui_robot_bridge.py").read_text(encoding="utf-8")

    def test_reach_measurement_is_automatic_and_announces_transitions(self):
        self.assertIn("state.measurement.samples.length >= 30", self.script)
        self.assertIn("양팔 리치 측정이 완료되었습니다. 오른팔 측정을 시작합니다.", self.script)
        self.assertIn("오른팔 리치 측정이 완료되었습니다. 왼팔 측정을 시작합니다.", self.script)
        self.assertIn("왼팔 리치 측정이 완료되었습니다. 전체 리치 측정이 완료되었습니다.", self.script)
        self.assertIn('id="captureMeasureButton"', self.template)
        self.assertIn("hidden admin-only", self.template)

    def test_five_combinations_are_available_without_fake_robot_coordinates(self):
        for combo_id, name in {
            1: "원투", 2: "잽잽 스트레이트", 3: "원투 훅", 4: "원투 원투", 5: "원투 어퍼",
        }.items():
            self.assertIn(f'{combo_id}: {{ name: "{name}"', self.script)
            self.assertIn(f'value="combination_{combo_id}"', self.template)
        self.assertIn('command == "training_start"', self.robot_bridge)
        self.assertIn('command == "training_go"', self.robot_bridge)
        self.assertNotIn("COMBINATION_TARGET_POSE", self.robot_bridge)

    def test_unverified_robot_punches_are_blocked_before_training_handoff(self):
        validation = self.script.split("function validateRobotTrainingProfile()", 1)[1].split(
            "function showToast", 1
        )[0]
        self.assertIn("robot_supported_punches", validation)
        self.assertIn("실물 경로 검증 후 사용할 수 있습니다", validation)

    def test_training_preparation_uses_front_mediapipe_then_stops_weaving(self):
        alignment = self.script.split("async function startAlignment()", 1)[1].split(
            "async function runCountdown()", 1
        )[0]
        countdown = self.script.split("async function runCountdown()", 1)[1].split(
            "function resetTrainingState()", 1
        )[0]
        self.assertIn("live.front_pose_detected", alignment)
        self.assertIn("frontTrainingPoseDetected(result)", alignment)
        self.assertNotIn("live.centered", alignment)
        self.assertIn('sendRobotCommand("training_start"', countdown)
        self.assertLess(
            countdown.index('showScreen("training")'),
            countdown.index("await waitForRobotTrainingReady()"),
        )
        self.assertIn("REACH_CALIBRATION_APPROACH", self.script)
        self.assertIn("Tool +Z 방향으로 천천히 접근", self.script)
        self.assertIn("MITT_CALIBRATION_PUNCH_READY", self.script)
        self.assertIn("2차 5회 펀치 보정 진행 중", self.script)
        self.assertIn("정면 MediaPipe 관절 검출 완료", self.robot_bridge)

    def test_report_contains_progress_best_check_strengths_and_next_training(self):
        for element_id in (
            "resultProgressCard", "resultBestImage", "resultWorstImage", "resultStrengths",
            "resultImprovements", "resultForceCard", "resultNextTrainingCard",
        ):
            self.assertIn(f'id="{element_id}"', self.template)
        self.assertIn("renderReportDetails(coaching)", self.script)
        self.assertIn("readCurrentReport", self.script)
        self.assertIn("결과읽어", self.script)

    def test_saved_report_can_be_opened_from_history(self):
        self.assertIn('id="screen-report-detail"', self.template)
        self.assertIn("openSavedReport(session.id)", self.script)
        self.assertIn("renderSavedReport(details)", self.script)
        self.assertIn("readSavedReport", self.script)

    def test_user_admin_and_force_modes_are_one_launcher(self):
        self.assertIn("--user-mode", self.launcher)
        self.assertIn("--admin-mode", self.launcher)
        self.assertIn("--force-monitor", self.launcher)
        self.assertIn("--force-control", self.launcher)
        self.assertIn('run_force_stack.sh" integrated', self.launcher)
        self.assertIn("--build-force", self.setup)

    def test_force_control_workspace_is_bundled_as_source(self):
        self.assertTrue((PROJECT_ROOT / "force_control/boxing_robot_ws/src/boxing_interfaces").is_dir())
        self.assertTrue((PROJECT_ROOT / "force_control/boxing_robot_ws/src/mitt_hit_system").is_dir())
        self.assertTrue((PROJECT_ROOT / "force_control/boxing_robot_ws/src/mitt_hit_bringup").is_dir())

    def test_admin_force_status_can_show_last_real_hit(self):
        self.assertIn("status.last_hit", self.script)
        self.assertIn("peak_force_n", self.script)
        self.assertIn("center_error_mm", self.script)


if __name__ == "__main__":
    unittest.main()

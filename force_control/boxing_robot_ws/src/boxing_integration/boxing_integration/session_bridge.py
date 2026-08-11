from __future__ import annotations

import json
import math
import os
import statistics
import threading
from typing import Any
from urllib.request import Request, urlopen

import rclpy
from boxing_interfaces.msg import HitResult, RtMittSample, SystemState, TargetPose
from boxing_interfaces.srv import (
    MoveMittPose,
    PreparePersonPose,
    StartHitTest,
    StopHitTest,
)
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from mitt_hit_system.mitt_pose_planner import (
    MittPosePlanner,
    PersonMeasurement,
    PunchType,
)

from boxing_integration.user_mitt_calibration import (
    VisionTarget,
    apply_tool_xy_correction,
    calculate_vision_target_calibration,
    hand_for_punch_role,
    normal_force_delta_n,
    predict_vision_target_pose,
    reach_calibration_hand,
    tool_z_offset_between,
)


UI_BASE = os.environ.get("KO_UI_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

REACH_STABLE_HOLD_NS = 500_000_000
REACH_FRONT_POSE_GRACE_NS = 750_000_000
REACH_BASELINE_SAMPLE_COUNT = 100
REACH_CONTACT_FORCE_N = 8.0
REACH_RELEASE_FORCE_N = 3.0
REACH_RELEASE_HOLD_NS = 300_000_000
FORCE_CALIBRATION_REQUIRED_HITS = 5
FORCE_CALIBRATION_MAXIMUM_CORRECTION_MM = 50.0


def punch_type_for_request(payload: dict[str, Any]) -> str:
    training_type = str(payload.get("training_type", "straight")).lower()
    hand = str(payload.get("hand", "right")).lower()
    if training_type in {"straight", "jab"}:
        return "STRAIGHT"
    if training_type == "uppercut":
        return "UPPERCUT"
    if training_type == "hook":
        if hand == "left":
            return "LEFT_HOOK"
        if hand == "right":
            return "RIGHT_HOOK"
        raise ValueError("양손 훅은 한 개의 고정 미트 자세로 준비할 수 없습니다.")
    raise ValueError(f"지원하지 않는 훈련 종류입니다: {training_type}")


def arm_length_mm_for_request(payload: dict[str, Any]) -> float:
    hand = str(payload.get("hand", "right")).lower()
    left = payload.get("left_punch_reach_cm")
    right = payload.get("right_punch_reach_cm")
    values = {
        "left": None if left in (None, "") else float(left) * 10.0,
        "right": None if right in (None, "") else float(right) * 10.0,
    }
    if hand in {"left", "right"} and values[hand] is not None:
        return float(values[hand])
    available = [value for value in values.values() if value is not None]
    if hand == "both" and available:
        return min(available)
    raise ValueError("선택한 손의 펀치 리치 측정값이 없습니다.")


def calibration_roles_for_request(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("calibration_roles", [])
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("보정 역할 목록이 올바르지 않습니다.")
    roles = []
    for value in raw:
        role = str(value).strip().lower()
        if role not in {"jab", "straight"}:
            raise ValueError("보정 역할은 jab 또는 straight여야 합니다.")
        if role not in roles:
            roles.append(role)
    return roles


def combination_steps_for_request(payload: dict[str, Any]) -> list[dict[str, str]]:
    if str(payload.get("mode", "single")).lower() != "combination":
        return []
    raw = payload.get("sequence")
    if not isinstance(raw, list) or not raw:
        raise ValueError("콤비네이션 sequence가 없습니다.")
    steps: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"콤비네이션 {index + 1}단계 형식이 올바르지 않습니다.")
        punch = str(item.get("punch", "")).strip().lower()
        hand = str(item.get("hand", "")).strip().lower()
        if punch not in {"jab", "straight", "hook", "uppercut"}:
            raise ValueError(f"지원하지 않는 콤비네이션 펀치입니다: {punch}")
        if hand not in {"left", "right"}:
            raise ValueError(f"콤비네이션 손 정보가 올바르지 않습니다: {hand}")
        steps.append({"punch": punch, "hand": hand})
    return steps


def suppress_rebound_for_request(payload: dict[str, Any]) -> bool:
    """Use direct pose-to-pose transitions only for combination training."""
    return bool(combination_steps_for_request(payload))


def personalized_pose_cache(
    payload: dict[str, Any], calibrations: list[dict[str, Any]]
) -> dict[tuple[str, str], tuple[float, ...]]:
    """Build all requested poses from this user's final hand calibration."""

    steps = combination_steps_for_request(payload)
    if not steps:
        steps = [
            {
                "punch": str(
                    payload.get("training_type") or payload.get("punch_type") or ""
                ).strip().lower(),
                "hand": str(payload.get("hand", "")).strip().lower(),
            }
        ]
    if any(step["punch"] not in {"jab", "straight", "hook", "uppercut"} for step in steps):
        raise ValueError("개인 미트 자세를 생성할 수 없는 펀치가 포함되어 있습니다.")
    if any(step["hand"] not in {"left", "right"} for step in steps):
        raise ValueError("개인 미트 자세에는 왼손 또는 오른손 정보가 필요합니다.")

    try:
        height_mm = float(payload["height_cm"]) * 10.0
        required_hands = {step["hand"] for step in steps}
        reaches = {
            hand: float(payload[f"{hand}_punch_reach_cm"]) * 10.0
            for hand in required_hands
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("개인 미트 자세 생성에 키와 선택한 손의 펀치 리치가 필요합니다.") from error
    if not math.isfinite(height_mm) or height_mm <= 0.0 or any(
        not math.isfinite(value) or value <= 0.0 for value in reaches.values()
    ):
        raise ValueError("개인 미트 자세의 신체 측정값이 올바르지 않습니다.")

    calibrated_by_hand: dict[str, tuple[float, ...]] = {}
    for item in calibrations:
        hand = str(item.get("hand", "")).strip().lower()
        raw_pose = item.get("calibrated_pose")
        if hand not in {"left", "right"} or not isinstance(raw_pose, list):
            continue
        try:
            pose = tuple(float(value) for value in raw_pose)
        except (TypeError, ValueError):
            continue
        if len(pose) == 6 and all(math.isfinite(value) for value in pose):
            calibrated_by_hand[hand] = pose

    cache: dict[tuple[str, str], tuple[float, ...]] = {}
    for step in steps:
        hand = step["hand"]
        punch = step["punch"]
        baseline = calibrated_by_hand.get(hand)
        if baseline is None:
            hand_text = "왼손" if hand == "left" else "오른손"
            raise ValueError(f"{hand_text} 최초 잽/스트레이트 미트 캘리브레이션이 필요합니다.")
        if punch in {"jab", "straight"}:
            cache[(hand, punch)] = baseline
            continue
        reach_mm = reaches[hand]
        planner = MittPosePlanner(
            reference_tcp_pose_mm_deg=baseline,
            reference_person_height_mm=height_mm,
            reference_person_arm_length_mm=reach_mm,
        )
        punch_type = (
            PunchType.LEFT_HOOK
            if punch == "hook" and hand == "left"
            else PunchType.RIGHT_HOOK
            if punch == "hook"
            else PunchType.UPPERCUT
        )
        cache[(hand, punch)] = planner.plan(
            PersonMeasurement(height_mm=height_mm, arm_length_mm=reach_mm),
            punch_type,
        ).tcp_pose_mm_deg
    return cache


def requires_personalized_pose_cache(payload: dict[str, Any]) -> bool:
    if combination_steps_for_request(payload):
        return True
    return str(
        payload.get("training_type") or payload.get("punch_type") or ""
    ).strip().lower() in {"hook", "uppercut"}


def validated_training_request(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    steps = combination_steps_for_request(normalized)
    if steps:
        normalized["training_type"] = steps[0]["punch"]
        normalized["punch_type"] = steps[0]["punch"]
        normalized["hand"] = steps[0]["hand"]
    else:
        training_type = str(
            normalized.get("training_type") or normalized.get("punch_type") or "straight"
        ).strip().lower()
        normalized["training_type"] = training_type
        normalized["punch_type"] = training_type

    punch_type_for_request(normalized)
    arm_length_mm_for_request(normalized)
    calibration_roles_for_request(normalized)
    dominant_hand = str(normalized.get("dominant_hand", "")).lower()
    if dominant_hand not in {"left", "right"}:
        raise ValueError("사용자 주손 정보가 없습니다.")
    height_mm = float(normalized.get("height_cm")) * 10.0
    if not 1000.0 <= height_mm <= 2300.0:
        raise ValueError("사용자 키가 허용 범위를 벗어났습니다.")
    if steps:
        for step in steps:
            test_payload = dict(normalized)
            test_payload["training_type"] = step["punch"]
            test_payload["punch_type"] = step["punch"]
            test_payload["hand"] = step["hand"]
            punch_type_for_request(test_payload)
            arm_length_mm_for_request(test_payload)
    return normalized


class SessionBridge(Node):
    """Own the transition from stopped weaving to one mitt hit session."""

    def __init__(self) -> None:
        super().__init__("boxing_session_bridge")
        self._lock = threading.Lock()
        self._pending: dict[str, Any] | None = None
        self._prepared_pending_start: dict[str, Any] | None = None
        self._last_mitt_state = ""
        self._active_client_session_id = ""
        self._starting = False
        self._stopping = False
        self._shutdown_requested = False
        self._active_payload: dict[str, Any] | None = None
        self._calibration_roles: list[str] = []
        self._calibration_role_index = 0
        self._calibration_samples: dict[str, list[dict[str, Any]]] = {}
        self._calibration_saving = False
        self._latest_fists: dict[str, dict[str, Any]] = {}
        self._last_target_pose: list[float] | None = None
        self._calibration_base_pose: list[float] | None = None
        self._calibration_candidates: list[dict[str, Any]] = []
        self._calibration_vision_impacts: list[dict[str, Any]] = []
        self._calibration_force_hits: list[dict[str, Any]] = []
        self._calibration_candidate_armed = {"left": True, "right": True}
        self._calibration_low_speed_since_ns: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._calibration_move_busy = False
        self._training_tracking_armed = {"left": True, "right": True}
        self._training_tracking_low_speed_since_ns: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._training_tracking_move_busy = False
        self._training_tracking_base_pose: list[float] | None = None
        self._combination_sequence: list[dict[str, str]] = []
        self._combination_index = 0
        self._combination_advance_pending = False
        self._combination_transitioning = False
        self._combination_cancel_command = ""
        self._personalized_pose_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._paused = False
        self._pause_inflight = False
        self._pause_requested = False
        self._deferred_stop_restart_weave: bool | None = None
        self._reach_phase: dict[str, Any] | None = None
        self._force_calibration_pending_hit: dict[str, Any] | None = None
        self._force_calibration_move_busy = False
        self._force_calibration_zeroing = False
        self._force_calibration_zero_not_ready_seen = False
        self._force_calibration_correction_x_mm = 0.0
        self._force_calibration_correction_y_mm = 0.0
        self._force_calibration_correction_limited = False
        self._pre_session_rezero_active = False
        self._pre_session_rezero_not_ready_seen = False

        self._prepare = self.create_client(
            PreparePersonPose, "/mitt/prepare_person_pose"
        )
        self._move_pose = self.create_client(MoveMittPose, "/mitt/move_pose")
        self._start_reach_approach = self.create_client(
            Trigger, "/mitt/start_reach_approach"
        )
        self._stop_motion = self.create_client(Trigger, "/mitt/stop_motion")
        self._start = self.create_client(StartHitTest, "/mitt/start_test")
        self._stop = self.create_client(StopHitTest, "/mitt/stop_test")
        self._motion_guard = self.create_client(SetBool, "/mitt/motion_guard")
        self._motion_rezero = self.create_client(SetBool, "/mitt/motion_rezero")
        self._recalibrate_zero = self.create_client(SetBool, "/mitt/recalibrate_zero")
        self._weave = self.create_publisher(String, "/robot_boxing/weave_command", 10)

        self.create_subscription(
            String, "/robot_boxing/training_request", self._on_training_request, 10
        )
        self.create_subscription(
            String, "/robot_boxing/action_ready", self._on_action_ready, 10
        )
        self.create_subscription(
            String, "/robot_boxing/session_command", self._on_session_command, 10
        )
        self.create_subscription(
            String, "/robot_boxing/weave_state", self._on_weave_state, 10
        )
        self.create_subscription(SystemState, "/mitt/system_state", self._on_state, 10)
        self.create_subscription(HitResult, "/mitt/hit_result", self._on_hit, 10)
        self.create_subscription(TargetPose, "/mitt/target_pose", self._on_target_pose, 10)
        rt_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            RtMittSample, "/mitt/rt_sample", self._on_rt_sample, rt_qos
        )
        self.create_subscription(
            String, "/sandbag/vision/status", self._on_vision_status, rt_qos
        )
        self.create_subscription(String, "/sandbag/fist_state", self._on_fist_state, 10)
        self.create_subscription(
            String, "/sandbag/impact_event", self._on_vision_impact, 10
        )
        self._post_status("INTEGRATION_READY", "미트 통합 브리지 준비")

    def _on_training_request(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("훈련 요청은 JSON 객체여야 합니다.")
            payload = validated_training_request(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._post_status("TRAINING_REQUEST_REJECTED", str(error))
            return
        with self._lock:
            self._pending = dict(payload)
        self._post_status("STOPPING_WEAVE", "훈련 선택 완료 → 위빙 정지 중")

    def _on_action_ready(self, message: String) -> None:
        with self._lock:
            if self._starting:
                return
            payload = dict(self._pending) if self._pending is not None else None
        # action_command carries the same full payload as training_request. This
        # fallback removes the cross-topic DDS ordering race.
        if payload is None:
            try:
                action = json.loads(message.data)
                if not isinstance(action, dict) or action.get("command") != "training_start":
                    return
                action.pop("command", None)
                payload = validated_training_request(action)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._post_status("TRAINING_REQUEST_REJECTED", str(error))
                return
            with self._lock:
                self._pending = dict(payload)
        with self._lock:
            if self._starting:
                return
            self._starting = True
        self._prepare_session(dict(payload))

    def _prepare_session(self, payload: dict[str, Any]) -> None:
        must_use_saved_pose = requires_personalized_pose_cache(payload)
        try:
            user_id = int(payload["user_id"])
        except (KeyError, TypeError, ValueError):
            self._fail_start("사용자 미트 캘리브레이션 조회에 사용자 ID가 필요합니다.")
            return
        calibrations = self._get_json(
            f"/api/users/{user_id}/mitt-calibrations"
        )
        if not isinstance(calibrations, list):
            self._fail_start("사용자 미트 캘리브레이션을 DB에서 불러오지 못했습니다.")
            return
        try:
            cache = personalized_pose_cache(payload, calibrations)
        except (TypeError, ValueError) as error:
            if must_use_saved_pose:
                self._fail_start(str(error))
                return
            with self._lock:
                self._personalized_pose_cache = {}
        else:
            with self._lock:
                self._personalized_pose_cache = dict(cache)
        if not self._prepare.wait_for_service(timeout_sec=2.0):
            self._fail_start("/mitt/prepare_person_pose 서비스가 준비되지 않았습니다.")
            return
        request = PreparePersonPose.Request()
        with self._lock:
            self._last_target_pose = None
        request.person_height_mm = float(payload["height_cm"]) * 10.0
        preparation_payload = dict(payload)
        if calibration_roles_for_request(payload):
            preparation_payload["hand"] = "both"
            preparation_payload["training_type"] = "straight"
        request.arm_length_mm = arm_length_mm_for_request(preparation_payload)
        request.punch_type = punch_type_for_request(preparation_payload)
        self._post_status("MOVING_TO_MITT_READY", "사용자 맞춤 미트 위치 준비 중")
        future = self._prepare.call_async(request)
        future.add_done_callback(lambda completed: self._after_prepare(completed, payload))

    def _after_prepare(self, future: Any, payload: dict[str, Any]) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"미트 위치 준비 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                f"미트 위치 준비 실패: {getattr(response, 'message', '응답 없음')}"
            )
            return
        if self._personalized_pose_for_payload(payload) is not None:
            self._move_to_personalized_initial_pose(payload)
        else:
            # Jab/straight calibration sessions still refresh the physical
            # reach and five-hit centering values stored for this user.
            self._begin_reach_calibration(payload)

    def _personalized_pose_for_payload(
        self, payload: dict[str, Any]
    ) -> tuple[float, ...] | None:
        hand = str(payload.get("hand", "")).strip().lower()
        punch = str(
            payload.get("training_type") or payload.get("punch_type") or ""
        ).strip().lower()
        with self._lock:
            pose = self._personalized_pose_cache.get((hand, punch))
        return tuple(pose) if pose is not None else None

    def _move_to_personalized_initial_pose(self, payload: dict[str, Any]) -> None:
        pose = self._personalized_pose_for_payload(payload)
        if pose is None:
            self._fail_start("DB 기반 사용자 미트 자세가 준비되지 않았습니다.")
            return
        self._post_status(
            "APPLYING_USER_MITT_CALIBRATION",
            "저장된 사용자 기준으로 미트 자세를 적용하고 있습니다.",
        )
        training_payload = dict(payload)
        # A valid saved pose is the completed user calibration. Do not enter
        # the live reach/five-hit calibration state again for jab/straight.
        training_payload["calibration_roles"] = []
        self._move_pose_request(
            pose,
            velocity=20.0,
            acceleration=20.0,
            callback=lambda completed: self._after_personalized_initial_move(
                completed, training_payload, pose
            ),
            failure_prefix="사용자 기준 미트 자세 이동 실패",
        )

    def _after_personalized_initial_move(
        self, future: Any, payload: dict[str, Any], pose: tuple[float, ...]
    ) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"사용자 기준 미트 자세 이동 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                "사용자 기준 미트 자세 이동 실패: "
                f"{getattr(response, 'message', '응답 없음')}"
            )
            return
        with self._lock:
            self._last_target_pose = list(pose)
            self._training_tracking_base_pose = list(pose)
        self._finish_prepare(payload)

    def _begin_reach_calibration(self, payload: dict[str, Any]) -> None:
        try:
            expected_hand = reach_calibration_hand(
                str(payload.get("dominant_hand", ""))
            )
            user_id = int(payload["user_id"])
            with self._lock:
                base_pose = list(self._last_target_pose) if self._last_target_pose else None
            if base_pose is None:
                raise ValueError("사용자 기본 미트 자세를 받지 못했습니다.")
        except (KeyError, TypeError, ValueError) as error:
            self._fail_start(f"앞뒤 리치 보정 준비 실패: {error}")
            return
        with self._lock:
            self._reach_phase = {
                "state": "WAITING_STABLE_FIST",
                "payload": dict(payload),
                "user_id": user_id,
                "hand": expected_hand,
                "base_pose": base_pose,
                "stable_since_ns": None,
                "pose_last_ready_ns": None,
                "fist_stable": False,
                "baseline_samples": [],
                "baseline_normal_force_n": None,
                "contact_sample": None,
                "contact_stop_requested": False,
                "release_since_ns": None,
            }
        hand_text = "왼팔" if expected_hand == "left" else "오른팔"
        self._post_status(
            "REACH_CALIBRATION_POSE",
            f"1차 리치 보정 · {hand_text}을 앞으로 끝까지 뻗고 움직이지 마세요.",
        )

    def _move_pose_request(
        self,
        pose: Any,
        *,
        velocity: float,
        acceleration: float,
        callback: Any,
        failure_prefix: str,
    ) -> None:
        if not self._move_pose.wait_for_service(timeout_sec=2.0):
            self._fail_start(f"{failure_prefix}: 서비스가 준비되지 않았습니다.")
            return
        request = MoveMittPose.Request()
        (
            request.x_mm,
            request.y_mm,
            request.z_mm,
            request.rx_deg,
            request.ry_deg,
            request.rz_deg,
        ) = pose
        request.velocity = float(velocity)
        request.acceleration = float(acceleration)
        future = self._move_pose.call_async(request)
        future.add_done_callback(callback)

    def _apply_saved_calibration(
        self, payload: dict[str, Any], calibration: dict[str, Any]
    ) -> None:
        try:
            role = str(calibration["punch_role"]).lower()
            selected_role = str(payload.get("training_type", "")).lower()
            if role != selected_role or role not in {"jab", "straight"}:
                raise ValueError("선택 훈련과 저장된 미트 보정 종류가 다릅니다.")
            expected_hand = hand_for_punch_role(
                str(payload.get("dominant_hand", "")), role
            )
            if str(calibration.get("hand", "")).lower() != expected_hand:
                raise ValueError("저장된 미트 보정 손 정보가 현재 주손과 다릅니다.")
            correction_x = float(calibration["correction_x_mm"])
            correction_y = float(calibration["correction_y_mm"])
            if not -50.0 <= correction_x <= 50.0 or not -50.0 <= correction_y <= 50.0:
                raise ValueError("저장된 미트 보정량이 ±50mm를 벗어났습니다.")
            with self._lock:
                base_pose = list(self._last_target_pose) if self._last_target_pose else None
            if base_pose is None:
                raise ValueError("사용자 기본 미트 자세를 받지 못했습니다.")
            target_pose = apply_tool_xy_correction(
                base_pose, correction_x, correction_y
            )
        except (KeyError, TypeError, ValueError) as error:
            self._fail_start(f"저장된 미트 보정값 적용 실패: {error}")
            return
        if not self._move_pose.wait_for_service(timeout_sec=2.0):
            self._fail_start("/mitt/move_pose 서비스가 준비되지 않았습니다.")
            return
        request = MoveMittPose.Request()
        (
            request.x_mm,
            request.y_mm,
            request.z_mm,
            request.rx_deg,
            request.ry_deg,
            request.rz_deg,
        ) = target_pose
        request.velocity = 20.0
        request.acceleration = 20.0
        self._post_status(
            "APPLYING_USER_MITT_CALIBRATION",
            f"저장된 {role} 개인 미트 위치 적용 중",
        )
        future = self._move_pose.call_async(request)
        future.add_done_callback(
            lambda completed: self._after_saved_calibration_move(completed, payload)
        )

    def _after_saved_calibration_move(
        self, future: Any, payload: dict[str, Any]
    ) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"저장된 미트 위치 이동 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                f"저장된 미트 위치 이동 실패: {getattr(response, 'message', '응답 없음')}"
            )
            return
        self._finish_prepare(payload)

    def _finish_prepare(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._prepared_pending_start = dict(payload)
            self._pre_session_rezero_active = True
            self._pre_session_rezero_not_ready_seen = False
        self._post_status(
            "MITT_CALIBRATION_ZEROING",
            "1차 이동 완료 · 현재 미트 자세에서 힘 영점을 다시 조정하고 있습니다.",
        )
        if not self._recalibrate_zero.wait_for_service(timeout_sec=2.0):
            self._fail_start("/mitt/recalibrate_zero 서비스가 준비되지 않았습니다.")
            return
        request = SetBool.Request()
        request.data = True
        future = self._recalibrate_zero.call_async(request)
        future.add_done_callback(self._after_pre_session_rezero_request)

    def _after_pre_session_rezero_request(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"현재 미트 자세 힘 영점 재조정 요청 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                "현재 미트 자세 힘 영점 재조정 요청 실패: "
                f"{getattr(response, 'message', '응답 없음')}"
            )

    def _start_prepared_session(self) -> None:
        with self._lock:
            if self._prepared_pending_start is None:
                return
            payload = dict(self._prepared_pending_start)
            self._prepared_pending_start = None
        if not self._start.wait_for_service(timeout_sec=2.0):
            self._fail_start("/mitt/start_test 서비스가 준비되지 않았습니다.")
            return
        self._post_status(
            "MITT_CALIBRATION_ZEROING",
            "영점 조정 중입니다. 잠시 기다려주세요.",
        )
        request = StartHitTest.Request()
        request.target_hit_count = 0
        request.auto_recover = False
        request.suppress_commanded_rebound = suppress_rebound_for_request(payload)
        future = self._start.call_async(request)
        future.add_done_callback(lambda completed: self._after_start(completed, payload))

    def _after_start(self, future: Any, payload: dict[str, Any]) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"타격 세션 시작 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                f"타격 세션 시작 실패: {getattr(response, 'message', '응답 없음')}"
            )
            return
        with self._lock:
            self._active_client_session_id = str(payload.get("client_session_id", ""))
            self._active_payload = dict(payload)
            self._calibration_roles = calibration_roles_for_request(payload)
            self._calibration_role_index = 0
            self._calibration_samples = {role: [] for role in self._calibration_roles}
            self._calibration_saving = False
            self._calibration_base_pose = (
                list(self._last_target_pose) if self._last_target_pose else None
            )
            self._calibration_candidates = []
            self._calibration_vision_impacts = []
            self._calibration_force_hits = []
            self._calibration_candidate_armed = {"left": True, "right": True}
            self._calibration_low_speed_since_ns = {"left": None, "right": None}
            self._calibration_move_busy = False
            self._force_calibration_pending_hit = None
            self._force_calibration_move_busy = False
            self._force_calibration_zeroing = bool(self._calibration_roles)
            self._force_calibration_zero_not_ready_seen = False
            self._force_calibration_correction_x_mm = 0.0
            self._force_calibration_correction_y_mm = 0.0
            self._force_calibration_correction_limited = False
            self._training_tracking_armed = {"left": True, "right": True}
            self._training_tracking_low_speed_since_ns = {"left": None, "right": None}
            self._training_tracking_move_busy = False
            self._training_tracking_base_pose = (
                list(self._last_target_pose) if self._last_target_pose else None
            )
            self._combination_sequence = combination_steps_for_request(payload)
            self._combination_index = 0
            self._combination_advance_pending = False
            self._combination_transitioning = False
            self._combination_cancel_command = ""
            self._pending = None
        if self._calibration_roles:
            self._post_status(
                "MITT_CALIBRATION_ZEROING",
                "영점 조정 중입니다. 잠시 기다려주세요.",
            )
        else:
            self._post_status("TRAINING_READY", "훈련 준비 완료")

    def _fail_start(self, detail: str) -> None:
        with self._lock:
            reach_state = str((self._reach_phase or {}).get("state", ""))
            if reach_state in {
                "MOVING_TO_FIST",
                "STOPPING_AT_CONTACT",
                "STOPPING_POSE_LOST",
                "STOPPING_ABORT",
            }:
                # A distance-unbounded Jog must remain represented in state so
                # a following stop/emergency-stop command can retry Soft Stop.
                preserve_reach_motion = True
            else:
                preserve_reach_motion = False
            force_session_active = self._active_payload is not None
            self._starting = preserve_reach_motion
            self._pending = None
            self._prepared_pending_start = None
            if not preserve_reach_motion:
                self._reach_phase = None
            self._pre_session_rezero_active = False
            self._pre_session_rezero_not_ready_seen = False
        self._post_status("SESSION_START_FAILED", detail)
        # A rejected/unreachable pose happens after weaving has already stopped.
        # Always route the robot out of the abandoned start. Before HitTest or
        # reach Jog exists, no force service needs stopping; otherwise use the
        # normal stop/abort path so compliance and unbounded Jog are released.
        if preserve_reach_motion or force_session_active:
            self._request_stop(restart_weave=True)
        else:
            self._reset_session_state_and_route(restart_weave=True)

    def _on_state(self, message: SystemState) -> None:
        combination_should_advance = False
        start_force_correction = False
        finish_force_zero = False
        with self._lock:
            self._last_mitt_state = message.state
            if self._pre_session_rezero_active:
                if message.state != "READY":
                    self._pre_session_rezero_not_ready_seen = True
                elif self._pre_session_rezero_not_ready_seen:
                    self._pre_session_rezero_active = False
            should_start = (
                message.state == "READY"
                and self._prepared_pending_start is not None
                and self._starting
                and not self._pre_session_rezero_active
            )
            calibration_active = (
                self._calibration_role_index < len(self._calibration_roles)
            )
            ready = bool(message.state == "WAITING_FOR_HIT" and message.accepting_hits)
            if calibration_active and self._force_calibration_zeroing:
                if not ready:
                    self._force_calibration_zero_not_ready_seen = True
                elif (
                    self._force_calibration_zero_not_ready_seen
                    or not self._calibration_samples.get(
                        self._calibration_roles[self._calibration_role_index], []
                    )
                ):
                    self._force_calibration_zeroing = False
                    self._force_calibration_zero_not_ready_seen = False
                    finish_force_zero = True
            if (
                calibration_active
                and ready
                and not self._force_calibration_zeroing
                and self._force_calibration_pending_hit is not None
                and not self._force_calibration_move_busy
            ):
                start_force_correction = True
            if ready and not calibration_active:
                self._starting = False
                combination_should_advance = bool(
                    self._combination_advance_pending
                    and self._combination_sequence
                    and not self._combination_transitioning
                )
                if combination_should_advance:
                    self._combination_advance_pending = False
                    self._combination_transitioning = True

        payload = {
            "mitt_state": message.state,
            # Keep the active reach/force calibration instruction intact.
            # SystemState is a lower-level force status and must not overwrite
            # the user-facing SessionBridge phase message.
            "mitt_message": message.detail,
            "robot_ready": bool(message.robot_ready),
            "compliance_enabled": bool(message.compliance_enabled),
            "accepting_hits": bool(message.accepting_hits),
        }
        if message.state == "WAITING_FOR_HIT" and message.accepting_hits:
            if not calibration_active:
                payload["state"] = "TRAINING_READY"
                payload["message"] = (
                    "다음 콤비네이션 자세 전환 준비"
                    if combination_should_advance
                    else "훈련 준비 완료"
                )
        elif message.state == "ERROR":
            payload["state"] = "MITT_CALIBRATION_FAILED"
            payload["message"] = message.detail
            payload["error_detail"] = message.detail
        self._post_json("/api/robot/status_update", payload)

        if finish_force_zero:
            self._after_force_calibration_zero_complete()
        if start_force_correction:
            self._begin_force_calibration_correction()
        if should_start:
            self._start_prepared_session()
        if combination_should_advance:
            self._begin_combination_transition()

    def _on_hit(self, message: HitResult) -> None:
        with self._lock:
            client_session_id = self._active_client_session_id
        payload = {
            "client_session_id": client_session_id,
            "stamp_ns": int(message.stamp.sec) * 1_000_000_000
            + int(message.stamp.nanosec),
            "hit_id": int(message.hit_id),
            "valid_hit": bool(message.valid_hit),
            "invalid_reason": message.invalid_reason,
            "hit_direction": message.hit_direction,
            "hit_x_mm": float(message.hit_x_mm),
            "hit_y_mm": float(message.hit_y_mm),
            "center_error_mm": float(message.center_error_mm),
            "peak_force_n": float(message.peak_force_n),
            "peak_normal_force_n": float(message.peak_normal_force_n),
            "impulse_ns": float(message.impulse_ns),
            "contact_duration_ms": float(message.contact_duration_ms),
            "accuracy_score": float(message.accuracy_score),
            "power_score": float(message.power_score),
            "total_score": float(message.total_score),
            "force_warning": bool(message.force_warning),
            "safety_stop": bool(message.safety_stop),
        }
        self._post_json("/api/force/hit", payload)
        begin_combination_transition = False
        with self._lock:
            calibration_active = self._calibration_role_index < len(self._calibration_roles)
            if calibration_active:
                if not message.valid_hit:
                    return
                if (
                    self._force_calibration_pending_hit is None
                    and not self._force_calibration_move_busy
                    and not self._force_calibration_zeroing
                ):
                    self._force_calibration_pending_hit = dict(payload)
                return
            if self._combination_sequence and not self._combination_transitioning:
                # In combination mode the analyzer publishes HitResult only
                # after contact force has ended and its state is WAITING_FOR_HIT.
                # Start the next protected pose move immediately.
                self._combination_advance_pending = False
                self._combination_transitioning = True
                begin_combination_transition = True
        if begin_combination_transition:
            self._begin_combination_transition()

    def _begin_combination_transition(self) -> None:
        with self._lock:
            if not self._combination_sequence or not self._combination_transitioning:
                return
            next_index = (self._combination_index + 1) % len(self._combination_sequence)
            step = dict(self._combination_sequence[next_index])
            active = dict(self._active_payload or {})
        next_payload = dict(active)
        next_payload["training_type"] = step["punch"]
        next_payload["punch_type"] = step["punch"]
        next_payload["hand"] = step["hand"]
        self._post_status(
            "COMBINATION_TRANSITION",
            f"다음 미트 자세 준비: {step['hand']} {step['punch']}",
        )
        pose = self._personalized_pose_for_payload(next_payload)
        if pose is None:
            self._fail_combination_transition("다음 사용자 콤비네이션 자세가 캐시에 없습니다.")
            return
        self._request_guarded_move(
            pose,
            velocity=80.0,
            acceleration=400.0,
            angular_velocity=20.0,
            angular_acceleration=40.0,
            rezero_after_move=True,
            done_callback=lambda succeeded, detail: self._after_combination_guarded_move(
                succeeded, detail, next_payload, next_index, pose
            ),
            failure_prefix="콤비네이션 미트 이동 실패",
        )

    def _combination_cancelled(self) -> str:
        with self._lock:
            return self._combination_cancel_command

    def _after_combination_guarded_move(
        self,
        succeeded: bool,
        detail: str,
        payload: dict[str, Any],
        next_index: int,
        pose: tuple[float, ...],
    ) -> None:
        if not succeeded:
            self._fail_combination_transition(detail)
            return
        with self._lock:
            self._last_target_pose = list(pose)
            self._combination_index = next_index
            self._active_payload = dict(payload)
            self._training_tracking_base_pose = (
                list(self._last_target_pose) if self._last_target_pose else None
            )
            self._training_tracking_armed = {"left": True, "right": True}
            self._training_tracking_low_speed_since_ns = {"left": None, "right": None}
            self._training_tracking_move_busy = False
            cancel = self._combination_cancel_command
            pause_requested = self._pause_requested
            self._combination_transitioning = False
        if cancel:
            self._request_stop(restart_weave=cancel == "stop")
            return
        if pause_requested:
            self._request_pause()
            return
        self._post_status(
            "COMBINATION_SETTING_REFERENCE",
            "다음 콤비네이션 미트 자세 도착 · 힘 기준 재설정 중",
        )

    def _fail_combination_transition(self, detail: str) -> None:
        with self._lock:
            self._combination_transitioning = False
            self._combination_advance_pending = False
        self._post_status("COMBINATION_TRANSITION_FAILED", detail)

    def _on_target_pose(self, message: TargetPose) -> None:
        pose = [
            float(message.x_mm),
            float(message.y_mm),
            float(message.z_mm),
            float(message.rx_deg),
            float(message.ry_deg),
            float(message.rz_deg),
        ]
        with self._lock:
            self._last_target_pose = pose

    def _on_rt_sample(self, message: RtMittSample) -> None:
        try:
            stamp_ns = int(message.stamp.sec) * 1_000_000_000 + int(
                message.stamp.nanosec
            )
            wrench = tuple(float(value) for value in message.corrected_wrench)
            tcp_pose = [float(value) for value in message.tcp_pose_mm_deg]
            if len(wrench) != 6 or len(tcp_pose) != 6:
                return
        except (TypeError, ValueError):
            return
        start_approach = False
        stop_at_contact = False
        finish_release = False
        with self._lock:
            phase = self._reach_phase
            if phase is None:
                return
            state = str(phase.get("state"))
            if state == "COLLECTING_FORCE_BASELINE":
                samples = phase.setdefault("baseline_samples", [])
                samples.append(wrench[2])
                if len(samples) >= REACH_BASELINE_SAMPLE_COUNT:
                    phase["baseline_normal_force_n"] = float(
                        statistics.median(samples[-REACH_BASELINE_SAMPLE_COUNT:])
                    )
                    phase["state"] = "APPROACH_READY"
                    start_approach = True
            elif state == "MOVING_TO_FIST":
                baseline = phase.get("baseline_normal_force_n")
                if baseline is not None and phase.get("contact_sample") is None:
                    delta = normal_force_delta_n(wrench, float(baseline))
                    if delta >= REACH_CONTACT_FORCE_N:
                        phase["contact_sample"] = {
                            "stamp_ns": stamp_ns,
                            "tcp_pose": tcp_pose,
                            "normal_force_n": wrench[2],
                            "delta_force_n": delta,
                        }
                        if not bool(phase.get("contact_stop_requested")):
                            phase["contact_stop_requested"] = True
                            phase["state"] = "STOPPING_AT_CONTACT"
                            stop_at_contact = True
            elif state == "WAITING_FORCE_RELEASE":
                baseline = phase.get("baseline_normal_force_n")
                if baseline is None:
                    return
                delta = normal_force_delta_n(wrench, float(baseline))
                if delta <= REACH_RELEASE_FORCE_N:
                    release_since = phase.get("release_since_ns")
                    if release_since is None:
                        phase["release_since_ns"] = stamp_ns
                    elif stamp_ns - int(release_since) >= REACH_RELEASE_HOLD_NS:
                        phase["state"] = "COMPLETE"
                        finish_release = True
                else:
                    phase["release_since_ns"] = None
        if start_approach:
            self._post_status(
                "REACH_CALIBRATION_APPROACH",
                "팔을 움직이지 마세요. 미트가 천천히 주먹으로 이동합니다.",
            )
            self._request_reach_approach()
        if stop_at_contact:
            self._request_reach_contact_stop()
        if finish_release:
            self._complete_reach_release()

    def _request_reach_approach(self) -> None:
        with self._lock:
            phase = self._reach_phase
            if phase is None or phase.get("state") != "APPROACH_READY":
                return
            phase["state"] = "STARTING_APPROACH"
        if not self._start_reach_approach.wait_for_service(timeout_sec=1.0):
            self._fail_start("/mitt/start_reach_approach 서비스가 준비되지 않았습니다.")
            return
        future = self._start_reach_approach.call_async(Trigger.Request())
        future.add_done_callback(self._after_reach_approach_started)

    def _after_reach_approach_started(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception as error:
            with self._lock:
                phase = self._reach_phase
                abort_after_start = bool(
                    phase is not None and phase.get("abort_after_start")
                )
                abort_restart_weave = bool(
                    phase is not None and phase.get("abort_restart_weave")
                )
            if abort_after_start:
                self._reset_session_state_and_route(
                    restart_weave=abort_restart_weave
                )
                return
            self._fail_start(f"앞뒤 리치 보정 연속 접근 시작 실패: {error}")
            return
        if response is None or not response.success:
            with self._lock:
                phase = self._reach_phase
                abort_after_start = bool(
                    phase is not None and phase.get("abort_after_start")
                )
                abort_restart_weave = bool(
                    phase is not None and phase.get("abort_restart_weave")
                )
            if abort_after_start:
                self._reset_session_state_and_route(
                    restart_weave=abort_restart_weave
                )
                return
            self._fail_start(
                "앞뒤 리치 보정 연속 접근 시작 실패: "
                f"{getattr(response, 'message', '응답 없음')}"
            )
            return
        with self._lock:
            phase = self._reach_phase
            if phase is None or phase.get("state") != "STARTING_APPROACH":
                return
            abort_after_start = bool(phase.get("abort_after_start"))
            abort_restart_weave = bool(phase.get("abort_restart_weave"))
            if abort_after_start:
                phase["state"] = "STOPPING_ABORT"
            else:
                phase["state"] = "MOVING_TO_FIST"
        if abort_after_start:
            self._request_reach_abort(restart_weave=abort_restart_weave)

    def _request_reach_contact_stop(self) -> None:
        if not self._stop_motion.wait_for_service(timeout_sec=1.0):
            self._fail_start("접촉 감지 후 /mitt/stop_motion 서비스가 준비되지 않았습니다.")
            return
        future = self._stop_motion.call_async(Trigger.Request())
        future.add_done_callback(self._after_reach_contact_stop)

    def _after_reach_contact_stop(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"접촉 감지 후 연속 접근 Soft Stop 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                "접촉 감지 후 연속 접근 Soft Stop 실패: "
                f"{getattr(response, 'message', '응답 없음')}"
            )
            return
        self._save_reach_contact()

    def _request_reach_pose_lost_stop(self) -> None:
        if not self._stop_motion.wait_for_service(timeout_sec=1.0):
            self._fail_start("정면 관절 유실 후 /mitt/stop_motion 서비스가 준비되지 않았습니다.")
            return
        future = self._stop_motion.call_async(Trigger.Request())
        future.add_done_callback(self._after_reach_pose_lost_stop)

    def _after_reach_pose_lost_stop(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception as error:
            self._fail_start(f"정면 관절 유실 후 연속 접근 Soft Stop 실패: {error}")
            return
        if response is None or not response.success:
            self._fail_start(
                "정면 관절 유실 후 연속 접근 Soft Stop 실패: "
                f"{getattr(response, 'message', '응답 없음')}"
            )
            return
        with self._lock:
            phase = self._reach_phase
            if phase is None or phase.get("state") != "STOPPING_POSE_LOST":
                return
            phase["state"] = "WAITING_STABLE_FIST"
            phase["stable_since_ns"] = None
            phase["baseline_samples"] = []
            phase["baseline_normal_force_n"] = None
            phase["contact_stop_requested"] = False
        self._post_status(
            "REACH_CALIBRATION_POSE",
            "정면 관절 인식이 끊겨 접근을 멈췄습니다. 팔을 다시 뻗어주세요.",
        )

    def _request_reach_abort(self, *, restart_weave: bool) -> None:
        if not self._stop_motion.wait_for_service(timeout_sec=1.0):
            with self._lock:
                self._stopping = False
            self._post_status(
                "SESSION_STOP_FAILED",
                "리치 연속 접근 정지 서비스가 준비되지 않았습니다.",
            )
            return
        future = self._stop_motion.call_async(Trigger.Request())
        future.add_done_callback(
            lambda completed: self._after_reach_abort(
                completed, restart_weave=restart_weave
            )
        )

    def _after_reach_abort(self, future: Any, *, restart_weave: bool) -> None:
        try:
            response = future.result()
        except Exception as error:
            response = None
            detail = str(error)
        else:
            detail = getattr(response, "message", "응답 없음")
        if response is None or not response.success:
            with self._lock:
                self._stopping = False
            self._post_status(
                "SESSION_STOP_FAILED", f"리치 연속 접근 정지 실패: {detail}"
            )
            return
        self._reset_session_state_and_route(restart_weave=restart_weave)

    def _save_reach_contact(self) -> None:
        try:
            with self._lock:
                phase = self._reach_phase
                if phase is None or phase.get("contact_sample") is None:
                    return
                if phase.get("state") == "SAVING_CONTACT":
                    return
                phase["state"] = "SAVING_CONTACT"
                payload = dict(phase["payload"])
                hand = str(phase["hand"])
                base_pose = list(phase["base_pose"])
                baseline = float(phase["baseline_normal_force_n"])
                contact = dict(phase["contact_sample"])
            contact_pose = list(contact["tcp_pose"])
            correction_z = tool_z_offset_between(base_pose, contact_pose)
            correction_z = max(0.0, correction_z)
            user_id = int(payload["user_id"])
        except (KeyError, TypeError, ValueError) as error:
            self._fail_start(f"앞뒤 리치 접촉 저장 실패: {error}")
            return
        document = {
            "hand": hand,
            "correction_z_mm": correction_z,
            "baseline_normal_force_n": baseline,
            "contact_delta_force_n": float(contact["delta_force_n"]),
            "base_pose": base_pose,
            "contact_pose": contact_pose,
        }
        if not self._post_json(
            f"/api/users/{user_id}/reach-calibration", document
        ):
            self._fail_start("앞뒤 리치 접촉 위치 저장에 실패했습니다.")
            return
        with self._lock:
            phase = self._reach_phase
            if phase is None:
                return
            # Keep the actual force-contact TCP captured before the Soft Stop
            # so later calibration starts from the stopped pose.
            self._last_target_pose = list(contact_pose)
            phase["state"] = "WAITING_FORCE_RELEASE"
            phase["release_since_ns"] = None
        self._post_status(
            "REACH_CALIBRATION_CONTACT_SAVED",
            "앞뒤 리치 위치를 저장했습니다. 이제 팔을 내려주세요.",
        )

    def _complete_reach_release(self) -> None:
        with self._lock:
            phase = self._reach_phase
            if phase is None:
                return
            payload = dict(phase["payload"])
            self._reach_phase = None
        self._post_status(
            "REACH_CALIBRATION_COMPLETE",
            "앞뒤 리치 보정 완료 · 타격 위치 보정을 준비합니다.",
        )
        self._finish_prepare(payload)

    def _on_fist_state(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            side = str(payload.get("side", "")).lower()
            if side not in {"left", "right"}:
                return
        except (TypeError, json.JSONDecodeError):
            return
        with self._lock:
            self._latest_fists[side] = payload
        self._consider_training_vision_target(payload)

    def _on_vision_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            stamp_ns = int(payload["stamp_ns"])
            pose_ready = bool(payload.get("front_pose_detected"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

        # The contact calibration advances along a fixed Tool +Z line and is
        # stopped by the force threshold. It does not require noisy 3D fist
        # triangulation; a fresh, complete FRONT 8-joint pose is the intended
        # user-presence/stability gate.
        ready_for_baseline = False
        reset_to_waiting = False
        stop_for_pose_loss = False
        with self._lock:
            phase = self._reach_phase
            if phase is None:
                return
            state = str(phase.get("state"))
            if state in {"SAVING_CONTACT", "WAITING_FORCE_RELEASE", "COMPLETE"}:
                return
            if pose_ready:
                phase["pose_last_ready_ns"] = stamp_ns
            last_ready_ns = phase.get("pose_last_ready_ns")
            pose_recent = (
                last_ready_ns is not None
                and stamp_ns - int(last_ready_ns) <= REACH_FRONT_POSE_GRACE_NS
            )
            if not pose_recent:
                phase["fist_stable"] = False
                phase["stable_since_ns"] = None
                if state in {"COLLECTING_FORCE_BASELINE", "APPROACH_READY"}:
                    phase["state"] = "WAITING_STABLE_FIST"
                    phase["baseline_samples"] = []
                    phase["baseline_normal_force_n"] = None
                    reset_to_waiting = True
                elif state == "MOVING_TO_FIST":
                    phase["state"] = "STOPPING_POSE_LOST"
                    stop_for_pose_loss = True
            else:
                phase["fist_stable"] = True
                if state != "WAITING_STABLE_FIST":
                    return
                stable_since = phase.get("stable_since_ns")
                if stable_since is None:
                    phase["stable_since_ns"] = stamp_ns
                    return
                if stamp_ns - int(stable_since) >= REACH_STABLE_HOLD_NS:
                    phase["state"] = "COLLECTING_FORCE_BASELINE"
                    phase["baseline_samples"] = []
                    ready_for_baseline = True
        if ready_for_baseline:
            self._post_status(
                "REACH_CALIBRATION_BASELINE",
                "정면 8개 관절 확인 완료 · 팔을 유지하세요. 외력 기준을 확인하고 있습니다.",
            )
        elif stop_for_pose_loss:
            self._request_reach_pose_lost_stop()
        elif reset_to_waiting:
            self._post_status(
                "REACH_CALIBRATION_POSE",
                "정면 관절 인식이 끊겼습니다. 팔을 다시 끝까지 뻗고 정지해 주세요.",
            )

    def _on_vision_impact(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            side = str(payload.get("side", "")).lower()
            stamp_ns = int(payload["impact_stamp_ns"])
            if side not in {"left", "right"} or stamp_ns <= 0:
                return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self._lock:
            if self._calibration_role_index >= len(self._calibration_roles):
                return
            # Final KO flow uses force-contact X/Y for the five-hit second
            # calibration. Keep the legacy Vision-impact matcher below for
            # compatibility, but do not feed it during this live flow.
            if self._active_client_session_id:
                return
            self._calibration_vision_impacts.append(
                {"side": side, "stamp_ns": stamp_ns, "payload": payload}
            )
            self._calibration_vision_impacts = self._calibration_vision_impacts[-32:]
        self._attempt_complete_calibration_sample()

    def _consider_vision_target(self, vision: dict[str, Any]) -> None:
        try:
            side = str(vision.get("side", "")).lower()
            stamp_ns = int(vision["stamp_ns"])
            velocity = tuple(float(value) for value in vision["velocity_base_mm_s"])
            speed = math.sqrt(sum(value * value for value in velocity))
        except (KeyError, TypeError, ValueError):
            return
        with self._lock:
            if (
                self._calibration_role_index >= len(self._calibration_roles)
                or self._calibration_saving
            ):
                return
            role = self._calibration_roles[self._calibration_role_index]
            active_payload = dict(self._active_payload or {})
            expected_hand = hand_for_punch_role(
                str(active_payload.get("dominant_hand", "")), role
            )
            if side != expected_hand:
                return
            if speed <= 250.0:
                low_since = self._calibration_low_speed_since_ns.get(side)
                if low_since is None:
                    self._calibration_low_speed_since_ns[side] = stamp_ns
                elif stamp_ns - low_since >= 150_000_000:
                    self._calibration_candidate_armed[side] = True
                return
            self._calibration_low_speed_since_ns[side] = None
            if not self._calibration_candidate_armed.get(side, True):
                return
            if self._calibration_move_busy:
                return
            base_pose = (
                list(self._calibration_base_pose)
                if self._calibration_base_pose is not None
                else None
            )
            current_pose = (
                list(self._last_target_pose) if self._last_target_pose is not None else None
            )
        try:
            if (
                not bool(vision.get("valid"))
                or speed < 500.0
                or float(vision.get("confidence", 0.0)) < 0.10
                or int(vision.get("camera_count", 0)) < 2
                or float(vision.get("position_std_mm", math.inf)) > 150.0
                or base_pose is None
                or current_pose is None
            ):
                return
            predicted = predict_vision_target_pose(
                base_pose,
                current_pose,
                vision["position_base_mm"],
                velocity,
                maximum_offset_mm=50.0,
                minimum_time_to_plane_ms=40.0,
                maximum_time_to_plane_ms=450.0,
                minimum_normal_speed_mm_s=100.0,
            )
        except (KeyError, TypeError, ValueError):
            return
        if predicted is None:
            return
        target_pose, time_to_plane_ms = predicted
        candidate = {
            "side": side,
            "stamp_ns": stamp_ns,
            "predicted_arrival_ns": stamp_ns + int(time_to_plane_ms * 1e6),
            "time_to_plane_ms": time_to_plane_ms,
            "target_pose": list(target_pose),
            "vision": dict(vision),
            "move_succeeded": False,
            "used": False,
        }
        with self._lock:
            if self._calibration_move_busy:
                return
            self._calibration_candidate_armed[side] = False
            self._calibration_move_busy = True
            self._calibration_candidates.append(candidate)
            self._calibration_candidates = self._calibration_candidates[-32:]
        self._move_to_vision_target(candidate)

    def _move_to_vision_target(self, candidate: dict[str, Any]) -> None:
        if not self._move_pose.wait_for_service(timeout_sec=2.0):
            self._finish_vision_target_move(candidate, None, "서비스가 준비되지 않았습니다.")
            return
        request = MoveMittPose.Request()
        (
            request.x_mm,
            request.y_mm,
            request.z_mm,
            request.rx_deg,
            request.ry_deg,
            request.rz_deg,
        ) = candidate["target_pose"]
        request.velocity = 300.0
        request.acceleration = 2000.0
        future = self._move_pose.call_async(request)
        future.add_done_callback(
            lambda completed: self._after_vision_target_move(completed, candidate)
        )

    def _after_vision_target_move(
        self, future: Any, candidate: dict[str, Any]
    ) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                self._finish_vision_target_move(
                    candidate, False, getattr(response, "message", "응답 없음")
                )
                return
        except Exception as error:
            self._finish_vision_target_move(candidate, False, str(error))
            return
        self._finish_vision_target_move(candidate, True, "")

    def _finish_vision_target_move(
        self, candidate: dict[str, Any], succeeded: bool | None, detail: str
    ) -> None:
        with self._lock:
            candidate["move_succeeded"] = bool(succeeded)
            self._calibration_move_busy = False
            if not succeeded:
                self._calibration_candidate_armed[candidate["side"]] = True
        if not succeeded:
            self._post_status(
                "MITT_CALIBRATION",
                f"Vision 목표 이동 실패 · 다시 천천히 타격해 주세요: {detail}",
            )
            return
        self._attempt_complete_calibration_sample()

    def _consider_training_vision_target(self, vision: dict[str, Any]) -> None:
        """Move the mitt once per incoming punch using the tested Vision predictor.

        The same ``predict_vision_target_pose`` path covered by
        ``test_user_mitt_calibration.py`` is reused here for normal training.
        Calibration keeps its existing, separate state machine unchanged.
        """
        try:
            side = str(vision.get("side", "")).lower()
            stamp_ns = int(vision["stamp_ns"])
            velocity = tuple(float(value) for value in vision["velocity_base_mm_s"])
            speed = math.sqrt(sum(value * value for value in velocity))
        except (KeyError, TypeError, ValueError):
            return

        with self._lock:
            calibration_active = self._calibration_role_index < len(self._calibration_roles)
            active_payload = dict(self._active_payload or {})
            selected_hand = str(active_payload.get("hand", "")).lower()
            tracking_allowed = bool(
                self._active_client_session_id
                and active_payload
                and not calibration_active
                and not self._paused
                and not self._stopping
                and not self._combination_transitioning
                and self._last_mitt_state == "WAITING_FOR_HIT"
            )
            if not tracking_allowed:
                return
            if selected_hand not in {"left", "right", "both"}:
                return
            if selected_hand != "both" and side != selected_hand:
                return
            if speed <= 250.0:
                low_since = self._training_tracking_low_speed_since_ns.get(side)
                if low_since is None:
                    self._training_tracking_low_speed_since_ns[side] = stamp_ns
                elif stamp_ns - low_since >= 150_000_000:
                    self._training_tracking_armed[side] = True
                return
            self._training_tracking_low_speed_since_ns[side] = None
            if not self._training_tracking_armed.get(side, True):
                return
            if self._training_tracking_move_busy:
                return
            base_pose = (
                list(self._training_tracking_base_pose)
                if self._training_tracking_base_pose is not None
                else None
            )
            current_pose = list(base_pose) if base_pose is not None else None

        try:
            if (
                not bool(vision.get("valid"))
                or speed < 500.0
                or float(vision.get("confidence", 0.0)) < 0.10
                or int(vision.get("camera_count", 0)) < 2
                or float(vision.get("position_std_mm", math.inf)) > 150.0
                or base_pose is None
                or current_pose is None
            ):
                return
            predicted = predict_vision_target_pose(
                base_pose,
                current_pose,
                vision["position_base_mm"],
                velocity,
                maximum_offset_mm=50.0,
                minimum_time_to_plane_ms=40.0,
                maximum_time_to_plane_ms=450.0,
                minimum_normal_speed_mm_s=100.0,
            )
        except (KeyError, TypeError, ValueError):
            return
        if predicted is None:
            return

        target_pose, _time_to_plane_ms = predicted
        with self._lock:
            if self._training_tracking_move_busy:
                return
            self._training_tracking_armed[side] = False
            self._training_tracking_move_busy = True

        self._request_guarded_move(
            list(target_pose),
            velocity=300.0,
            acceleration=2000.0,
            rezero_after_move=False,
            done_callback=lambda succeeded, detail: self._after_training_vision_target_move(
                succeeded, detail, side
            ),
            failure_prefix="Vision 주먹 추적 미트 이동 실패",
        )

    def _after_training_vision_target_move(
        self, succeeded: bool, detail: str, side: str
    ) -> None:
        with self._lock:
            self._training_tracking_move_busy = False
            if not succeeded:
                self._training_tracking_armed[side] = True
        if not succeeded:
            self.get_logger().warning(f"Vision 주먹 추적 미트 이동 실패: {detail}")

    def _request_guarded_move(
        self,
        pose: list[float] | tuple[float, ...],
        *,
        velocity: float,
        acceleration: float,
        rezero_after_move: bool,
        done_callback: Any,
        failure_prefix: str,
        angular_velocity: float | None = None,
        angular_acceleration: float | None = None,
    ) -> None:
        """Move the mitt during an active hit session without tripping compliance.

        The hit analyzer owns the short intentional-motion guard.  For the
        five-hit force calibration we also request a new wrench zero at the
        settled target.  Normal Vision tracking keeps the current wrench zero.
        """
        services = (
            (self._motion_rezero, "/mitt/motion_rezero"),
            (self._motion_guard, "/mitt/motion_guard"),
            (self._move_pose, "/mitt/move_pose"),
        )
        for client, name in services:
            if not client.wait_for_service(timeout_sec=2.0):
                done_callback(False, f"{failure_prefix}: {name} 서비스가 준비되지 않았습니다.")
                return

        context = {
            "pose": [float(value) for value in pose],
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "angular_velocity": (
                float(angular_velocity) if angular_velocity is not None else None
            ),
            "angular_acceleration": (
                float(angular_acceleration)
                if angular_acceleration is not None
                else None
            ),
            "rezero": bool(rezero_after_move),
            "done_callback": done_callback,
            "failure_prefix": str(failure_prefix),
            "move_success": False,
            "move_detail": "",
        }
        request = SetBool.Request()
        request.data = bool(rezero_after_move)
        future = self._motion_rezero.call_async(request)
        future.add_done_callback(
            lambda completed: self._after_guarded_rezero_mode(completed, context)
        )

    def _after_guarded_rezero_mode(self, future: Any, context: dict[str, Any]) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                detail = getattr(response, "message", "응답 없음")
                context["done_callback"](
                    False, f"{context['failure_prefix']}: 영점 모드 설정 실패: {detail}"
                )
                return
        except Exception as error:
            context["done_callback"](
                False, f"{context['failure_prefix']}: 영점 모드 설정 실패: {error}"
            )
            return

        request = SetBool.Request()
        request.data = True
        guard_future = self._motion_guard.call_async(request)
        guard_future.add_done_callback(
            lambda completed: self._after_guarded_motion_arm(completed, context)
        )

    def _after_guarded_motion_arm(self, future: Any, context: dict[str, Any]) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                detail = getattr(response, "message", "응답 없음")
                self._clear_next_motion_rezero()
                context["done_callback"](
                    False, f"{context['failure_prefix']}: 이동 가드 시작 실패: {detail}"
                )
                return
        except Exception as error:
            self._clear_next_motion_rezero()
            context["done_callback"](
                False, f"{context['failure_prefix']}: 이동 가드 시작 실패: {error}"
            )
            return

        request = MoveMittPose.Request()
        (
            request.x_mm,
            request.y_mm,
            request.z_mm,
            request.rx_deg,
            request.ry_deg,
            request.rz_deg,
        ) = context["pose"]
        request.velocity = context["velocity"]
        request.acceleration = context["acceleration"]
        if context["angular_velocity"] is not None:
            request.angular_velocity = context["angular_velocity"]
        if context["angular_acceleration"] is not None:
            request.angular_acceleration = context["angular_acceleration"]
        move_future = self._move_pose.call_async(request)
        move_future.add_done_callback(
            lambda completed: self._after_guarded_motion_move(completed, context)
        )

    def _clear_next_motion_rezero(self) -> None:
        if not self._motion_rezero.service_is_ready():
            return
        request = SetBool.Request()
        request.data = False
        try:
            self._motion_rezero.call_async(request)
        except Exception:
            pass

    def _after_guarded_motion_move(self, future: Any, context: dict[str, Any]) -> None:
        try:
            response = future.result()
            context["move_success"] = bool(response is not None and response.success)
            if not context["move_success"]:
                context["move_detail"] = getattr(response, "message", "응답 없음")
        except Exception as error:
            context["move_success"] = False
            context["move_detail"] = str(error)

        # Always close the guard, including a failed MoveMittPose request.
        request = SetBool.Request()
        request.data = False
        future = self._motion_guard.call_async(request)
        future.add_done_callback(
            lambda completed: self._after_guarded_motion_disarm(completed, context)
        )

    def _after_guarded_motion_disarm(self, future: Any, context: dict[str, Any]) -> None:
        guard_ok = False
        guard_detail = ""
        try:
            response = future.result()
            guard_ok = bool(response is not None and response.success)
            if not guard_ok:
                guard_detail = getattr(response, "message", "응답 없음")
        except Exception as error:
            guard_detail = str(error)

        if not guard_ok:
            context["done_callback"](
                False,
                f"{context['failure_prefix']}: 이동 가드 종료 실패: {guard_detail}",
            )
            return
        if not context["move_success"]:
            context["done_callback"](
                False,
                f"{context['failure_prefix']}: {context['move_detail'] or '이동 실패'}",
            )
            return
        context["done_callback"](True, "")

    def _begin_force_calibration_correction(self) -> None:
        with self._lock:
            if (
                self._calibration_role_index >= len(self._calibration_roles)
                or self._force_calibration_pending_hit is None
                or self._force_calibration_move_busy
                or self._force_calibration_zeroing
            ):
                return
            role = self._calibration_roles[self._calibration_role_index]
            hit = dict(self._force_calibration_pending_hit)
            self._force_calibration_pending_hit = None
            base_pose = (
                list(self._calibration_base_pose)
                if self._calibration_base_pose is not None
                else None
            )
            before_x = float(self._force_calibration_correction_x_mm)
            before_y = float(self._force_calibration_correction_y_mm)
            self._force_calibration_move_busy = True

        if base_pose is None:
            self._force_calibration_failed("2차 미트 보정 기준 자세가 없습니다.")
            return
        try:
            hit_x = float(hit["hit_x_mm"])
            hit_y = float(hit["hit_y_mm"])
            raw_x = before_x - hit_x
            raw_y = before_y - hit_y
            after_x = max(
                -FORCE_CALIBRATION_MAXIMUM_CORRECTION_MM,
                min(FORCE_CALIBRATION_MAXIMUM_CORRECTION_MM, raw_x),
            )
            after_y = max(
                -FORCE_CALIBRATION_MAXIMUM_CORRECTION_MM,
                min(FORCE_CALIBRATION_MAXIMUM_CORRECTION_MM, raw_y),
            )
            limited = not (
                math.isclose(raw_x, after_x, abs_tol=1e-9)
                and math.isclose(raw_y, after_y, abs_tol=1e-9)
            )
            target_pose = list(apply_tool_xy_correction(base_pose, after_x, after_y))
        except (KeyError, TypeError, ValueError) as error:
            self._force_calibration_failed(f"2차 미트 보정 타격값 오류: {error}")
            return

        context = {
            "role": role,
            "hit": hit,
            "before_x_mm": before_x,
            "before_y_mm": before_y,
            "after_x_mm": after_x,
            "after_y_mm": after_y,
            "limited": limited,
            "target_pose": target_pose,
        }
        self._post_status(
            "MITT_CALIBRATION_ADJUSTING",
            "타격 방향을 반영해 미트 위치를 후보정하고 있습니다.",
        )
        self._request_guarded_move(
            target_pose,
            velocity=80.0,
            acceleration=400.0,
            rezero_after_move=True,
            done_callback=lambda succeeded, detail: self._after_force_calibration_correction_move(
                succeeded, detail, context
            ),
            failure_prefix="2차 미트 위치 보정 이동 실패",
        )

    def _after_force_calibration_correction_move(
        self, succeeded: bool, detail: str, context: dict[str, Any]
    ) -> None:
        if not succeeded:
            self._force_calibration_failed(detail)
            return

        hit = dict(context["hit"])
        sample = {
            "stamp_ns": int(hit.get("stamp_ns", 0)),
            "hit_id": int(hit.get("hit_id", 0)),
            "hit_direction": str(hit.get("hit_direction", "")),
            "hit_x_mm": float(hit.get("hit_x_mm", 0.0)),
            "hit_y_mm": float(hit.get("hit_y_mm", 0.0)),
            "center_error_mm": float(hit.get("center_error_mm", 0.0)),
            "peak_force_n": float(hit.get("peak_force_n", 0.0)),
            "peak_normal_force_n": float(hit.get("peak_normal_force_n", 0.0)),
            "correction_before_x_mm": float(context["before_x_mm"]),
            "correction_before_y_mm": float(context["before_y_mm"]),
            "correction_after_x_mm": float(context["after_x_mm"]),
            "correction_after_y_mm": float(context["after_y_mm"]),
            "target_pose": list(context["target_pose"]),
        }
        with self._lock:
            role = str(context["role"])
            samples = self._calibration_samples.setdefault(role, [])
            samples.append(sample)
            self._force_calibration_correction_x_mm = float(context["after_x_mm"])
            self._force_calibration_correction_y_mm = float(context["after_y_mm"])
            self._force_calibration_correction_limited = bool(
                self._force_calibration_correction_limited or context["limited"]
            )
            self._force_calibration_move_busy = False
            self._force_calibration_zeroing = True
            self._force_calibration_zero_not_ready_seen = False
            self._training_tracking_base_pose = list(context["target_pose"])
        self._post_status(
            "MITT_CALIBRATION_ZEROING",
            "영점 조정 중입니다. 잠시 기다려주세요.",
        )

    def _after_force_calibration_zero_complete(self) -> None:
        with self._lock:
            if self._calibration_role_index >= len(self._calibration_roles):
                return
            role = self._calibration_roles[self._calibration_role_index]
            active_payload = dict(self._active_payload or {})
            samples = list(self._calibration_samples.get(role, []))
            count = len(samples)
        if count >= FORCE_CALIBRATION_REQUIRED_HITS:
            self._save_force_calibration(role, active_payload, samples[:FORCE_CALIBRATION_REQUIRED_HITS])
            return
        self._post_status(
            "MITT_CALIBRATION_PUNCH_READY",
            f"2차 미트 위치 보정 · 다시 펀치하세요. ({count}/{FORCE_CALIBRATION_REQUIRED_HITS})",
        )

    def _save_force_calibration(
        self,
        role: str,
        active_payload: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> None:
        try:
            if len(samples) < FORCE_CALIBRATION_REQUIRED_HITS:
                raise ValueError("2차 미트 보정 타격이 5회 미만입니다.")
            with self._lock:
                base_pose = (
                    list(self._calibration_base_pose)
                    if self._calibration_base_pose is not None
                    else None
                )
                correction_x = float(self._force_calibration_correction_x_mm)
                correction_y = float(self._force_calibration_correction_y_mm)
                correction_limited = bool(self._force_calibration_correction_limited)
            if base_pose is None:
                raise ValueError("2차 미트 보정 기준 자세가 없습니다.")
            calibrated_pose = list(
                apply_tool_xy_correction(base_pose, correction_x, correction_y)
            )
            hit_x_values = [float(sample["hit_x_mm"]) for sample in samples]
            hit_y_values = [float(sample["hit_y_mm"]) for sample in samples]
            raw_center_x = statistics.mean(hit_x_values)
            raw_center_y = statistics.mean(hit_y_values)
            dispersion = math.sqrt(
                statistics.mean(
                    [
                        (x - raw_center_x) ** 2 + (y - raw_center_y) ** 2
                        for x, y in zip(hit_x_values, hit_y_values)
                    ]
                )
            )
            hand = hand_for_punch_role(
                str(active_payload.get("dominant_hand", "")), role
            )
            user_id = int(active_payload["user_id"])
        except (KeyError, TypeError, ValueError) as error:
            self._force_calibration_failed(f"2차 미트 보정 저장 준비 실패: {error}")
            return

        document = {
            "punch_role": role,
            "hand": hand,
            "correction_x_mm": correction_x,
            "correction_y_mm": correction_y,
            "raw_center_x_mm": raw_center_x,
            "raw_center_y_mm": raw_center_y,
            "sample_count": len(samples),
            "accepted_sample_count": len(samples),
            "dispersion_mm": dispersion,
            "correction_limited": correction_limited,
            "base_pose": base_pose,
            "calibrated_pose": calibrated_pose,
            "raw_samples": samples,
            "vision_summary": {
                "source": "force_contact_xy_5_hit_centering",
                "sample_count": len(samples),
                "force_contact_coordinates_used": True,
            },
        }
        if not self._post_json(f"/api/users/{user_id}/mitt-calibrations", document):
            self._force_calibration_failed("사용자 미트 보정값 저장에 실패했습니다.")
            return

        with self._lock:
            self._calibration_role_index += 1
            self._calibration_saving = False
            self._force_calibration_pending_hit = None
            self._force_calibration_move_busy = False
            self._force_calibration_zeroing = False
            self._force_calibration_zero_not_ready_seen = False
            self._calibration_base_pose = calibrated_pose
            self._training_tracking_base_pose = calibrated_pose
            finished = self._calibration_role_index >= len(self._calibration_roles)
            if not finished:
                next_role = self._calibration_roles[self._calibration_role_index]
                self._calibration_samples.setdefault(next_role, [])
                self._force_calibration_correction_x_mm = 0.0
                self._force_calibration_correction_y_mm = 0.0
                self._force_calibration_correction_limited = False
                self._force_calibration_zeroing = True

        if finished:
            self._post_status(
                "MITT_CALIBRATION_COMPLETE",
                "2차 미트 위치 보정 완료 · 실제 훈련을 시작합니다.",
            )
            self._post_status("TRAINING_READY", "훈련 준비 완료")
        else:
            self._post_status(
                "MITT_CALIBRATION_ZEROING",
                "영점 조정 중입니다. 잠시 기다려주세요.",
            )

    def _force_calibration_failed(self, detail: str) -> None:
        with self._lock:
            self._force_calibration_pending_hit = None
            self._force_calibration_move_busy = False
            self._force_calibration_zeroing = False
        self._post_status("MITT_CALIBRATION_FAILED", detail)
        self._request_stop(restart_weave=True)

    def _announce_current_calibration_role(self) -> None:
        with self._lock:
            if self._calibration_role_index >= len(self._calibration_roles):
                return
            role = self._calibration_roles[self._calibration_role_index]
            payload = dict(self._active_payload or {})
            count = len(self._calibration_samples.get(role, []))
        hand = hand_for_punch_role(str(payload.get("dominant_hand")), role)
        role_text = "잽" if role == "jab" else "스트레이트"
        hand_text = "왼손" if hand == "left" else "오른손"
        self._post_status(
            "MITT_CALIBRATION",
            f"{hand_text} {role_text}를 천천히 10회 타격해 주세요. ({count}/10)",
        )

    def _attempt_complete_calibration_sample(self) -> None:
        selected: list[dict[str, Any]] = []
        with self._lock:
            if (
                self._calibration_role_index >= len(self._calibration_roles)
                or self._calibration_saving
            ):
                return
            role = self._calibration_roles[self._calibration_role_index]
            active_payload = dict(self._active_payload or {})
            expected_hand = hand_for_punch_role(
                str(active_payload.get("dominant_hand", "")), role
            )
            best: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
            for candidate in self._calibration_candidates:
                if (
                    candidate.get("used")
                    or not candidate.get("move_succeeded")
                    or candidate.get("side") != expected_hand
                ):
                    continue
                arrival_ns = int(candidate["predicted_arrival_ns"])
                for impact in self._calibration_vision_impacts:
                    if impact.get("side") != expected_hand:
                        continue
                    impact_error = abs(int(impact["stamp_ns"]) - arrival_ns)
                    if impact_error > 500_000_000:
                        continue
                    for force_hit in self._calibration_force_hits:
                        contact_error = abs(
                            int(force_hit["stamp_ns"]) - int(impact["stamp_ns"])
                        )
                        if contact_error > 500_000_000:
                            continue
                        score = impact_error + contact_error
                        if best is None or score < best[0]:
                            best = (score, candidate, impact, force_hit)
            if best is None:
                return
            _, candidate, impact, force_hit = best
            candidate["used"] = True
            self._calibration_vision_impacts.remove(impact)
            self._calibration_force_hits.remove(force_hit)
            sample = {
                "target_pose": list(candidate["target_pose"]),
                "candidate_stamp_ns": int(candidate["stamp_ns"]),
                "predicted_arrival_ns": int(candidate["predicted_arrival_ns"]),
                "time_to_plane_ms": float(candidate["time_to_plane_ms"]),
                "hand": expected_hand,
                "vision_fist": dict(candidate["vision"]),
                "vision_impact": dict(impact["payload"]),
                "force_contact": {
                    "stamp_ns": int(force_hit["stamp_ns"]),
                    "hit_id": int(force_hit["hit_id"]),
                    "peak_force_n": float(force_hit["peak_force_n"]),
                    "peak_normal_force_n": float(force_hit["peak_normal_force_n"]),
                    "valid_hit": True,
                },
            }
            samples = self._calibration_samples.setdefault(role, [])
            samples.append(sample)
            count = len(samples)
            self._calibration_candidate_armed[expected_hand] = True
            if count >= 10:
                self._calibration_saving = True
                selected = list(samples[:10])
        if selected:
            self._save_completed_calibration(
                role, expected_hand, active_payload, selected
            )
        else:
            self._announce_current_calibration_role()

    def _save_completed_calibration(
        self,
        role: str,
        hand: str,
        active_payload: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> None:
        try:
            with self._lock:
                base_pose = (
                    list(self._calibration_base_pose)
                    if self._calibration_base_pose is not None
                    else None
                )
            if base_pose is None:
                raise ValueError("기본 미트 TCP 자세가 기록되지 않았습니다.")
            result = calculate_vision_target_calibration(
                [
                    VisionTarget(tuple(float(value) for value in sample["target_pose"]))
                    for sample in samples
                ],
                base_pose,
                required_sample_count=10,
                maximum_correction_mm=50.0,
            )
            user_id = int(active_payload["user_id"])
        except (KeyError, TypeError, ValueError) as error:
            with self._lock:
                self._calibration_saving = False
            self._post_status("MITT_CALIBRATION_FAILED", str(error))
            return
        calibrated_pose = apply_tool_xy_correction(
            base_pose, result.correction_x_mm, result.correction_y_mm
        )
        document = {
            "punch_role": role,
            "hand": hand,
            "correction_x_mm": result.correction_x_mm,
            "correction_y_mm": result.correction_y_mm,
            "raw_center_x_mm": result.raw_center_x_mm,
            "raw_center_y_mm": result.raw_center_y_mm,
            "sample_count": result.sample_count,
            "accepted_sample_count": result.accepted_sample_count,
            "dispersion_mm": result.dispersion_mm,
            "correction_limited": result.correction_limited,
            "base_pose": base_pose,
            "calibrated_pose": list(calibrated_pose),
            "raw_samples": samples,
            "vision_summary": {
                "source": "vision_predicted_target_confirmed_by_force_contact",
                "sample_count": len(samples),
                "late_policy_used": False,
                "force_contact_coordinates_used": False,
            },
        }
        saved = self._post_json(
            f"/api/users/{user_id}/mitt-calibrations", document
        )
        if not saved:
            with self._lock:
                self._calibration_saving = False
            self._post_status(
                "MITT_CALIBRATION_FAILED", "사용자 미트 보정값 저장에 실패했습니다."
            )
            return
        with self._lock:
            self._calibration_role_index += 1
            self._calibration_saving = False
            self._calibration_candidates = []
            self._calibration_vision_impacts = []
            self._calibration_force_hits = []
            self._calibration_candidate_armed = {"left": True, "right": True}
            self._calibration_low_speed_since_ns = {"left": None, "right": None}
            finished = self._calibration_role_index >= len(self._calibration_roles)
        if finished:
            self._post_status(
                "MITT_CALIBRATION_COMPLETE",
                "잽과 스트레이트 개인 미트 기준값 저장 완료",
            )
            self._request_stop(restart_weave=False)
        else:
            self._announce_current_calibration_role()

    def _on_session_command(self, message: String) -> None:
        command = message.data.strip().lower()
        if command not in {"stop", "emergency_stop", "system_shutdown", "pause", "resume"}:
            return
        if command == "pause":
            self._request_pause()
            return
        if command == "resume":
            self._request_resume()
            return

        self._shutdown_requested = command == "system_shutdown"
        with self._lock:
            has_mitt_session = bool(self._active_client_session_id or self._starting)
            combination_transitioning = self._combination_transitioning
            if combination_transitioning:
                self._combination_cancel_command = command
        if combination_transitioning:
            self._post_status("SESSION_STOP_REQUESTED", "콤비네이션 자세 전환 완료 후 안전하게 세션을 종료합니다.")
            return
        if command == "system_shutdown" and not has_mitt_session:
            self._publish_weave_command("home")
            self._post_status("RETURNING_HOME_FOR_SHUTDOWN", "로봇 HOME 이동 중")
            return
        if command == "emergency_stop":
            self._post_status("EMERGENCY_STOP", "비상정지 요청")
            if not has_mitt_session:
                return
        self._request_stop(restart_weave=command == "stop")

    def _request_pause(self) -> None:
        with self._lock:
            if self._paused:
                self._post_status("TRAINING_PAUSED", "훈련이 이미 일시정지 상태입니다.")
                return
            if self._pause_inflight:
                return
            if not self._active_client_session_id:
                self._post_status("PAUSE_REJECTED", "활성 타격 세션이 없어 일시정지할 수 없습니다.")
                return
            if self._combination_transitioning:
                self._pause_requested = True
                self._post_status("PAUSE_PENDING", "콤비네이션 자세 전환 직후 안전하게 일시정지합니다.")
                return
            self._pause_requested = False
            self._pause_inflight = True
        if not self._stop.wait_for_service(timeout_sec=2.0):
            with self._lock:
                self._pause_inflight = False
            self._post_status("PAUSE_FAILED", "/mitt/stop_test 서비스가 준비되지 않았습니다.")
            return
        self._post_status("PAUSING_TRAINING", "힘 세션을 안전하게 일시정지하는 중")
        future = self._stop.call_async(StopHitTest.Request())
        future.add_done_callback(self._after_pause)

    def _after_pause(self, future: Any) -> None:
        success = False
        detail = ""
        try:
            response = future.result()
            success = bool(response is not None and response.success)
            detail = getattr(response, "message", "응답 없음") if response is not None else "응답 없음"
        except Exception as error:
            detail = str(error)
        with self._lock:
            self._pause_inflight = False
            deferred = self._deferred_stop_restart_weave
            self._deferred_stop_restart_weave = None
            if success:
                self._paused = True
        if deferred is not None:
            if success:
                self._reset_session_state_and_route(restart_weave=deferred)
            else:
                self._request_stop(restart_weave=deferred)
            return
        if not success:
            self._post_status("PAUSE_FAILED", f"타격 세션 일시정지 실패: {detail}")
            return
        self._post_status("TRAINING_PAUSED", "힘/컴플라이언스 세션 정지 완료 · 타격 기록 일시정지")

    def _request_resume(self) -> None:
        with self._lock:
            if self._pause_inflight:
                self._post_status("RESUME_REJECTED", "일시정지 전환이 끝난 뒤 다시 시도해 주세요.")
                return
            if not self._paused or not self._active_client_session_id or self._active_payload is None:
                self._post_status("RESUME_REJECTED", "일시정지된 활성 세션이 없습니다.")
                return
            self._pause_inflight = True
        if not self._start.wait_for_service(timeout_sec=2.0):
            with self._lock:
                self._pause_inflight = False
            self._post_status("RESUME_FAILED", "/mitt/start_test 서비스가 준비되지 않았습니다.")
            return
        request = StartHitTest.Request()
        request.target_hit_count = 0
        request.auto_recover = False
        with self._lock:
            active_payload = dict(self._active_payload or {})
        request.suppress_commanded_rebound = suppress_rebound_for_request(active_payload)
        self._post_status("TRAINING_RESUMING", "힘 기준/컴플라이언스 재활성화 중")
        future = self._start.call_async(request)
        future.add_done_callback(self._after_resume)

    def _after_resume(self, future: Any) -> None:
        success = False
        detail = ""
        try:
            response = future.result()
            success = bool(response is not None and response.success)
            detail = getattr(response, "message", "응답 없음") if response is not None else "응답 없음"
        except Exception as error:
            detail = str(error)
        with self._lock:
            self._pause_inflight = False
            if success:
                self._paused = False
        if not success:
            self._post_status("RESUME_FAILED", f"타격 세션 재개 실패: {detail}")
            return
        # HitAnalyzer publishes WAITING_FOR_HIT after compliance stabilization.
        # UI keeps its timer paused until that state arrives.
        self._post_status("TRAINING_RESUMING", str(detail))

    def _on_weave_state(self, message: String) -> None:
        if self._shutdown_requested and message.data == "IDLE_HOME":
            self._post_json(
                "/api/robot/status_update",
                {
                    "state": "SYSTEM_SHUTDOWN_READY",
                    "message": "로봇 HOME 도착 · 시스템 종료 준비 완료",
                    "shutdown_ready": True,
                },
            )

    def _request_stop(self, *, restart_weave: bool) -> None:
        with self._lock:
            if self._stopping:
                return
            if self._pause_inflight:
                # The in-flight StopHitTest used by pause already performs the
                # required compliance release. Finish routing as soon as it returns.
                self._deferred_stop_restart_weave = bool(restart_weave)
                return
            self._stopping = True
            paused = self._paused
            reach_phase = self._reach_phase
            reach_state = str((reach_phase or {}).get("state", ""))
            if reach_state == "STARTING_APPROACH":
                reach_phase["abort_after_start"] = True
                reach_phase["abort_restart_weave"] = bool(restart_weave)
                return
            reach_motion_active = reach_state in {
                "MOVING_TO_FIST",
                "STOPPING_AT_CONTACT",
                "STOPPING_POSE_LOST",
                "STOPPING_ABORT",
            }
            if reach_motion_active:
                reach_phase["state"] = "STOPPING_ABORT"
            reach_phase_present = reach_phase is not None
        if reach_motion_active:
            self._request_reach_abort(restart_weave=restart_weave)
            return
        if reach_phase_present:
            self._reset_session_state_and_route(restart_weave=restart_weave)
            return
        if paused:
            # Pause already called StopHitTest successfully; do not treat the
            # expected "no hit test active" response as a shutdown failure.
            self._after_stop(None, restart_weave)
            return
        if not self._stop.wait_for_service(timeout_sec=2.0):
            self._after_stop(None, restart_weave)
            return
        future = self._stop.call_async(StopHitTest.Request())
        future.add_done_callback(
            lambda completed: self._after_stop(completed, restart_weave)
        )

    def _after_stop(self, future: Any, restart_weave: bool) -> None:
        if future is not None:
            try:
                response = future.result()
                if response is not None and not response.success:
                    self._post_status("SESSION_STOP_FAILED", response.message)
                    with self._lock:
                        self._stopping = False
                    return
            except Exception as error:
                self._post_status("SESSION_STOP_FAILED", str(error))
                with self._lock:
                    self._stopping = False
                return
        self._reset_session_state_and_route(restart_weave=restart_weave)

    def _reset_session_state_and_route(self, *, restart_weave: bool) -> None:
        with self._lock:
            self._active_client_session_id = ""
            self._active_payload = None
            self._calibration_roles = []
            self._calibration_role_index = 0
            self._calibration_samples = {}
            self._calibration_saving = False
            self._calibration_base_pose = None
            self._calibration_candidates = []
            self._calibration_vision_impacts = []
            self._calibration_force_hits = []
            self._calibration_candidate_armed = {"left": True, "right": True}
            self._calibration_low_speed_since_ns = {"left": None, "right": None}
            self._calibration_move_busy = False
            self._training_tracking_armed = {"left": True, "right": True}
            self._training_tracking_low_speed_since_ns = {"left": None, "right": None}
            self._training_tracking_move_busy = False
            self._training_tracking_base_pose = None
            self._combination_sequence = []
            self._combination_index = 0
            self._combination_advance_pending = False
            self._combination_transitioning = False
            self._combination_cancel_command = ""
            self._personalized_pose_cache = {}
            self._paused = False
            self._pause_inflight = False
            self._pause_requested = False
            self._deferred_stop_restart_weave = None
            self._reach_phase = None
            self._force_calibration_pending_hit = None
            self._force_calibration_move_busy = False
            self._force_calibration_zeroing = False
            self._force_calibration_zero_not_ready_seen = False
            self._force_calibration_correction_x_mm = 0.0
            self._force_calibration_correction_y_mm = 0.0
            self._force_calibration_correction_limited = False
            self._starting = False
            self._pending = None
            self._prepared_pending_start = None
            self._stopping = False
        if restart_weave:
            self._publish_weave_command("start")
            self._post_status("WEAVE_RESTART_REQUESTED", "결과 분석 중 · 위빙 재시작")
        elif self._shutdown_requested:
            self._publish_weave_command("home")
            self._post_status("RETURNING_HOME_FOR_SHUTDOWN", "로봇 HOME 이동 중")

    def _publish_weave_command(self, command: str) -> bool:
        if not rclpy.ok():
            return False
        output = String()
        output.data = command
        try:
            self._weave.publish(output)
            return True
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warning(
                    f"weave_command publish 실패: {type(exc).__name__}: {exc}"
                )
            return False

    def _post_status(self, state: str, message: str) -> None:
        payload = {"state": state, "message": message, "error_detail": ""}
        if state.endswith("FAILED") or state.endswith("REJECTED"):
            payload["error_detail"] = message
        self._post_json("/api/robot/status_update", payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{UI_BASE}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=1.0):
                return True
        except Exception:
            return False

    def _get_json(self, path: str) -> Any:
        request = Request(
            f"{UI_BASE}{path}",
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=1.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SessionBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

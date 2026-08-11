import threading
from types import SimpleNamespace

import pytest

from boxing_integration.session_bridge import (
    SessionBridge,
    arm_length_mm_for_request,
    calibration_roles_for_request,
    personalized_pose_cache,
    punch_type_for_request,
    requires_personalized_pose_cache,
    suppress_rebound_for_request,
)
from mitt_hit_system.wrench_frame_adapter import rotation_from_zyz_degrees


def _bridge_for_start_failure(*, force_active=False, reach_state=""):
    bridge = object.__new__(SessionBridge)
    bridge._lock = threading.Lock()
    bridge._reach_phase = {"state": reach_state} if reach_state else None
    bridge._active_payload = {} if force_active else None
    bridge._starting = True
    bridge._pending = {"training_type": "hook"}
    bridge._prepared_pending_start = None
    bridge._pre_session_rezero_active = False
    bridge._pre_session_rezero_not_ready_seen = False
    return bridge


def test_pre_session_start_failure_routes_directly_back_to_weaving():
    bridge = _bridge_for_start_failure()
    statuses = []
    routes = []
    stops = []
    bridge._post_status = lambda state, detail: statuses.append((state, detail))
    bridge._reset_session_state_and_route = lambda *, restart_weave: routes.append(restart_weave)
    bridge._request_stop = lambda *, restart_weave: stops.append(restart_weave)

    bridge._fail_start("unverified hook pose")

    assert statuses == [("SESSION_START_FAILED", "unverified hook pose")]
    assert routes == [True]
    assert stops == []


@pytest.mark.parametrize("force_active,reach_state", [(True, ""), (False, "MOVING_TO_FIST")])
def test_active_start_failure_uses_normal_stop_or_reach_abort_path(force_active, reach_state):
    bridge = _bridge_for_start_failure(force_active=force_active, reach_state=reach_state)
    routes = []
    stops = []
    bridge._post_status = lambda state, detail: None
    bridge._reset_session_state_and_route = lambda *, restart_weave: routes.append(restart_weave)
    bridge._request_stop = lambda *, restart_weave: stops.append(restart_weave)

    bridge._fail_start("active failure")

    assert routes == []
    assert stops == [True]


def test_straight_and_jab_share_the_front_facing_pose():
    assert punch_type_for_request({"training_type": "straight"}) == "STRAIGHT"
    assert punch_type_for_request({"training_type": "jab"}) == "STRAIGHT"


def test_hook_pose_uses_selected_hand():
    assert punch_type_for_request({"training_type": "hook", "hand": "left"}) == "LEFT_HOOK"
    assert punch_type_for_request({"training_type": "hook", "hand": "right"}) == "RIGHT_HOOK"
    with pytest.raises(ValueError, match="양손 훅"):
        punch_type_for_request({"training_type": "hook", "hand": "both"})


def test_only_combination_requests_suppress_commanded_rebound():
    assert not suppress_rebound_for_request({"mode": "single"})
    assert suppress_rebound_for_request(
        {
            "mode": "combination",
            "sequence": [
                {"punch": "jab", "hand": "left"},
                {"punch": "straight", "hand": "right"},
            ],
        }
    )


def test_hook_and_uppercut_require_personalized_pose_cache():
    assert not requires_personalized_pose_cache(
        {"mode": "single", "training_type": "straight"}
    )
    assert requires_personalized_pose_cache(
        {"mode": "single", "training_type": "hook"}
    )
    assert requires_personalized_pose_cache(
        {
            "mode": "combination",
            "sequence": [{"punch": "jab", "hand": "left"}],
        }
    )


def test_personalized_combo_poses_use_each_hands_db_calibration():
    left_pose = [10.0, 20.0, 300.0, 175.0, 90.0, 83.0]
    right_pose = [40.0, 50.0, 330.0, 175.0, 90.0, 83.0]
    payload = {
        "mode": "combination",
        "height_cm": 180.0,
        "left_punch_reach_cm": 60.0,
        "right_punch_reach_cm": 70.0,
        "sequence": [
            {"punch": "jab", "hand": "left"},
            {"punch": "hook", "hand": "left"},
            {"punch": "straight", "hand": "right"},
            {"punch": "uppercut", "hand": "right"},
        ],
    }
    calibrations = [
        {"hand": "left", "calibrated_pose": left_pose},
        {"hand": "right", "calibrated_pose": right_pose},
    ]

    cache = personalized_pose_cache(payload, calibrations)

    assert cache[("left", "jab")] == pytest.approx(left_pose)
    assert cache[("right", "straight")] == pytest.approx(right_pose)
    assert cache[("left", "hook")] != pytest.approx(left_pose)
    uppercut = cache[("right", "uppercut")]
    rotation = rotation_from_zyz_degrees(uppercut[3:])
    assert tuple(rotation[row][2] for row in range(3)) == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-7
    )
    assert uppercut[:2] == pytest.approx(right_pose[:2])
    assert uppercut[2] < right_pose[2]


def test_personalized_pose_cache_rejects_missing_hand_calibration():
    with pytest.raises(ValueError, match="오른손 최초"):
        personalized_pose_cache(
            {
                "mode": "single",
                "training_type": "hook",
                "hand": "right",
                "height_cm": 173.0,
                "right_punch_reach_cm": 66.0,
            },
            [{"hand": "left", "calibrated_pose": [0.0] * 6}],
        )


def test_saved_straight_pose_is_reused_without_live_calibration_roles():
    straight_pose = (10.0, 20.0, 300.0, 175.0, 90.0, 83.0)
    bridge = object.__new__(SessionBridge)
    bridge._lock = threading.Lock()
    bridge._personalized_pose_cache = {("right", "straight"): straight_pose}
    bridge._post_status = lambda *args: None
    captured = []
    bridge._after_personalized_initial_move = (
        lambda future, payload, pose: captured.append((payload, pose))
    )
    bridge._move_pose_request = lambda pose, **kwargs: kwargs["callback"](
        SimpleNamespace()
    )
    payload = {
        "mode": "single",
        "training_type": "straight",
        "hand": "right",
        "calibration_roles": ["straight"],
    }

    SessionBridge._move_to_personalized_initial_pose(bridge, payload)

    assert captured[0][0]["calibration_roles"] == []
    assert captured[0][1] == pytest.approx(straight_pose)


def test_combination_hit_starts_next_pose_on_contact_release_result():
    bridge = object.__new__(SessionBridge)
    bridge._lock = threading.Lock()
    bridge._active_client_session_id = "combo-session"
    bridge._calibration_roles = []
    bridge._calibration_role_index = 0
    bridge._force_calibration_pending_hit = None
    bridge._force_calibration_move_busy = False
    bridge._force_calibration_zeroing = False
    bridge._combination_sequence = [
        {"punch": "jab", "hand": "left"},
        {"punch": "straight", "hand": "right"},
    ]
    bridge._combination_advance_pending = False
    bridge._combination_transitioning = False
    transitions = []
    bridge._post_json = lambda *args: None
    bridge._begin_combination_transition = lambda: transitions.append("next")
    message = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=2),
        hit_id=1,
        valid_hit=True,
        invalid_reason="",
        hit_direction="CENTER",
        hit_x_mm=0.0,
        hit_y_mm=0.0,
        center_error_mm=0.0,
        peak_force_n=30.0,
        peak_normal_force_n=30.0,
        impulse_ns=1.0,
        contact_duration_ms=20.0,
        accuracy_score=100.0,
        power_score=50.0,
        total_score=75.0,
        force_warning=False,
        safety_stop=False,
    )

    SessionBridge._on_hit(bridge, message)

    assert not bridge._combination_advance_pending
    assert bridge._combination_transitioning
    assert transitions == ["next"]


def test_combination_transition_keeps_hit_session_and_uses_guarded_pose_move():
    bridge = object.__new__(SessionBridge)
    bridge._lock = threading.Lock()
    bridge._combination_sequence = [
        {"punch": "straight", "hand": "right"},
        {"punch": "hook", "hand": "left"},
    ]
    bridge._combination_index = 0
    bridge._combination_transitioning = True
    bridge._active_payload = {"mode": "combination", "training_type": "straight"}
    bridge._combination_cancel_command = ""
    bridge._pause_requested = False
    bridge._last_target_pose = None
    hook_pose = (380.0, -720.0, 395.0, 125.0, -90.0, 90.0)
    bridge._personalized_pose_for_payload = lambda payload: hook_pose
    statuses = []
    guarded_moves = []
    bridge._post_status = lambda state, detail: statuses.append((state, detail))
    bridge._request_guarded_move = lambda pose, **kwargs: guarded_moves.append(
        (pose, kwargs)
    )
    bridge._request_stop = lambda **kwargs: pytest.fail("session must stay active")
    bridge._request_pause = lambda: pytest.fail("session must stay active")

    SessionBridge._begin_combination_transition(bridge)

    assert len(guarded_moves) == 1
    pose, options = guarded_moves[0]
    assert pose == hook_pose
    assert options["rezero_after_move"]
    assert options["velocity"] == 80.0
    assert options["angular_velocity"] == 20.0
    assert options["angular_acceleration"] == 40.0
    options["done_callback"](True, "")
    assert bridge._combination_index == 1
    assert bridge._active_payload["training_type"] == "hook"
    assert bridge._active_payload["hand"] == "left"
    assert not bridge._combination_transitioning
    assert statuses[-1][0] == "COMBINATION_SETTING_REFERENCE"


def test_arm_length_uses_selected_side_and_conservative_both_value():
    profile = {"left_punch_reach_cm": 68, "right_punch_reach_cm": 72}
    assert arm_length_mm_for_request({**profile, "hand": "left"}) == 680
    assert arm_length_mm_for_request({**profile, "hand": "right"}) == 720
    assert arm_length_mm_for_request({**profile, "hand": "both"}) == 680


def test_calibration_roles_are_validated_and_deduplicated():
    assert calibration_roles_for_request(
        {"calibration_roles": ["jab", "straight", "jab"]}
    ) == ["jab", "straight"]
    with pytest.raises(ValueError, match="jab 또는 straight"):
        calibration_roles_for_request({"calibration_roles": ["hook"]})

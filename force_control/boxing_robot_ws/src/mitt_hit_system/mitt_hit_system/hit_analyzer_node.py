"""ROS 2 hit analyzer with an explicit, opt-in test lifecycle."""

from collections import deque
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any

from boxing_interfaces.msg import (
    HitResult,
    RtMittSample,
    SystemState as SystemStateMessage,
)
from boxing_interfaces.srv import StartHitTest, StopHitTest
from dsr_msgs2.srv import (
    ChangeCollisionSensitivity,
    GetRobotSpeedMode,
    MoveLine,
    MoveStop,
    ReadDataRt,
    ReleaseComplianceCtrl,
    TaskComplianceCtrl,
)
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterUninitializedException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import SetBool

from mitt_hit_system.hit_analyzer import (
    HitAnalyzerConfig,
    HitAnalyzerProcessor,
    HitAnalyzerResult,
)
from mitt_hit_system.hit_record_logger import HitRecordLogger
from mitt_hit_system.hit_test_manager import HitTestManager
from mitt_hit_system.compliance_controller import (
    ComplianceConfig,
    ComplianceController,
)
from mitt_hit_system.collision_sensitivity_controller import (
    CollisionSensitivityConfig,
    CollisionSensitivityController,
)
from mitt_hit_system.common.enums import SystemState
from mitt_hit_system.hit_score_calculator import HitScoreCalculator, ScoringConfig
from mitt_hit_system.impact_buffer import BufferedWrenchSample
from mitt_hit_system.return_to_reference import (
    ReturnObservation,
    ReturnStatus,
    ReturnToReferenceConfig,
    ReturnToReferenceMonitor,
)
from mitt_hit_system.rebound_motion import (
    ReboundMotionConfig,
    ReboundMotionController,
    ReboundPhase,
)


class HitAnalyzerNode(Node):

    def __init__(self, **node_kwargs: Any) -> None:
        self._intentional_motion_lock = threading.Lock()
        self._intentional_motion_active = False
        self._intentional_motion_deadline = 0.0
        self._intentional_motion_timeout_sec = 2.0
        self._intentional_reference_pending = False
        self._next_motion_rezero = False
        self._active_motion_rezero = False
        self._settled_motion_rezero = False
        self._post_move_zero_active = False
        self._session_rebound_enabled = True
        super().__init__("hit_analyzer", **node_kwargs)
        self._declare_parameters()
        processor_config = self._make_config()
        self._processor = HitAnalyzerProcessor(processor_config)
        self._score_calculator = HitScoreCalculator(
            ScoringConfig(
                perfect_radius_mm=processor_config.perfect_radius_mm,
                valid_radius_mm=float(
                    self.get_parameter("scoring_valid_radius_mm").value
                ),
            )
        )
        self._record_logger = HitRecordLogger(
            self._resolve_path(str(self.get_parameter("record_dir").value)),
            save_raw_hit_data=bool(
                self.get_parameter("save_contact_samples").value
            ),
        )
        client_callback_group = ReentrantCallbackGroup()
        lifecycle_callback_group = MutuallyExclusiveCallbackGroup()
        compliance = ComplianceController(
            self._make_compliance_config(),
            self.create_client(
                TaskComplianceCtrl,
                str(self.get_parameter("compliance_service").value),
                callback_group=client_callback_group,
            ),
            self.create_client(
                ReleaseComplianceCtrl,
                str(self.get_parameter("release_compliance_service").value),
                callback_group=client_callback_group,
            ),
            self.create_client(
                ReadDataRt,
                str(self.get_parameter("rt_service").value),
                callback_group=client_callback_group,
            ),
            self.create_client(
                GetRobotSpeedMode,
                str(self.get_parameter("speed_mode_service").value),
                callback_group=client_callback_group,
            ),
            activate_request_factory=TaskComplianceCtrl.Request,
            release_request_factory=ReleaseComplianceCtrl.Request,
            rt_request_factory=ReadDataRt.Request,
            speed_mode_request_factory=GetRobotSpeedMode.Request,
        )
        self._rebound_motion = ReboundMotionController(
            self._make_rebound_motion_config(),
            self.create_client(
                MoveLine,
                str(self.get_parameter("rebound_move_line_service").value),
                callback_group=client_callback_group,
            ),
            self.create_client(
                MoveStop,
                str(self.get_parameter("rebound_move_stop_service").value),
                callback_group=client_callback_group,
            ),
            move_line_request_factory=MoveLine.Request,
            move_stop_request_factory=MoveStop.Request,
        )
        self._collision_sensitivity = CollisionSensitivityController(
            CollisionSensitivityConfig(
                enabled=bool(
                    self.get_parameter("collision_sensitivity_override_enabled").value
                ),
                training_percent=int(
                    self.get_parameter("training_collision_sensitivity_percent").value
                ),
                restore_percent=int(
                    self.get_parameter("normal_collision_sensitivity_percent").value
                ),
                service_timeout_sec=float(
                    self.get_parameter("collision_sensitivity_service_timeout_sec").value
                ),
            ),
            self.create_client(
                ChangeCollisionSensitivity,
                str(self.get_parameter("collision_sensitivity_service").value),
                callback_group=client_callback_group,
            ),
            request_factory=ChangeCollisionSensitivity.Request,
        )
        self._manager = HitTestManager(
            compliance,
            self._start_session,
            allow_continuous_session=bool(
                self.get_parameter("allow_continuous_session").value
            ),
        )
        self._validate_compliance_stabilization_config()
        self._return_monitor = self._make_return_monitor()
        self._return_samples: deque[ReturnObservation] = deque(maxlen=10_000)
        self._return_hit_id: int | None = None
        self._robot_ready = False
        self._last_rt_sample_monotonic_ns = time.monotonic_ns()
        self._last_state_snapshot: tuple[object, ...] | None = None
        self._shutdown_session_complete = False
        self._compliance_stabilization_started_monotonic_ns: int | None = None
        self._compliance_reference_pose_mm_deg: tuple[float, ...] | None = None
        self._session_maximum_tcp_displacement_mm = 0.0
        self._session_maximum_tcp_angular_displacement_deg = 0.0
        self._session_maximum_total_force_n = 0.0
        self._session_maximum_total_torque_nm = 0.0
        self._expected_frame_id = str(
            self.get_parameter("expected_frame_id").value
        )
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(HitResult, output_topic, 10)
        self._state_publisher = self.create_publisher(
            SystemStateMessage, "/mitt/system_state", 10
        )
        rt_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            RtMittSample,
            input_topic,
            self._on_rt_sample,
            rt_qos,
            callback_group=lifecycle_callback_group,
        )
        self._start_service = self.create_service(
            StartHitTest,
            "/mitt/start_test",
            self._on_start,
            callback_group=lifecycle_callback_group,
        )
        self._stop_service = self.create_service(
            StopHitTest,
            "/mitt/stop_test",
            self._on_stop,
            callback_group=lifecycle_callback_group,
        )
        self._motion_guard_service = self.create_service(
            SetBool,
            "/mitt/motion_guard",
            self._on_motion_guard,
            callback_group=lifecycle_callback_group,
        )
        self._motion_rezero_service = self.create_service(
            SetBool,
            "/mitt/motion_rezero",
            self._on_motion_rezero,
            callback_group=lifecycle_callback_group,
        )
        self._recalibrate_zero_service = self.create_service(
            SetBool,
            "/mitt/recalibrate_zero",
            self._on_recalibrate_zero,
            callback_group=lifecycle_callback_group,
        )
        self.create_timer(
            0.1,
            self._on_status_timer,
            callback_group=lifecycle_callback_group,
        )
        self.create_timer(
            0.02,
            self._on_rt_watchdog_timer,
            callback_group=lifecycle_callback_group,
        )
        self._last_log_ns = 0
        self._ready_logged = False
        compliance_startup_detail = (
            "Compliance is configured but remains inactive until Start."
            if self._manager.compliance.config.enabled
            else "Compliance is disabled by configuration."
        )
        self.get_logger().info(
            "Hit analyzer started. Keep the mitt untouched for zeroing; hits are "
            "ignored until /mitt/start_test succeeds. " + compliance_startup_detail
        )

    def _make_return_monitor(self) -> ReturnToReferenceMonitor | None:
        if not self._manager.compliance.config.enabled:
            return None
        return ReturnToReferenceMonitor(
            ReturnToReferenceConfig(
                position_tolerance_mm=float(
                    self.get_parameter("return_position_tolerance_mm").value
                ),
                velocity_tolerance_mm_s=float(
                    self.get_parameter("return_velocity_tolerance_mm_s").value
                ),
                force_tolerance_n=float(
                    self.get_parameter("return_force_tolerance_n").value
                ),
                settle_time_ms=float(
                    self.get_parameter("return_settle_time_ms").value
                ),
                timeout_ms=float(self.get_parameter("return_timeout_ms").value),
            )
        )

    def _make_rebound_motion_config(self) -> ReboundMotionConfig:
        return ReboundMotionConfig(
            enabled=bool(self.get_parameter("commanded_rebound_enabled").value),
            minimum_distance_mm=float(
                self.get_parameter("rebound_minimum_distance_mm").value
            ),
            maximum_distance_mm=float(
                self.get_parameter("rebound_maximum_distance_mm").value
            ),
            force_for_maximum_distance_n=float(
                self.get_parameter("rebound_force_for_maximum_distance_n").value
            ),
            retreat_velocity_mm_s=float(
                self.get_parameter("rebound_retreat_velocity_mm_s").value
            ),
            retreat_acceleration_mm_s2=float(
                self.get_parameter("rebound_retreat_acceleration_mm_s2").value
            ),
            return_velocity_mm_s=float(
                self.get_parameter("rebound_return_velocity_mm_s").value
            ),
            return_acceleration_mm_s2=float(
                self.get_parameter("rebound_return_acceleration_mm_s2").value
            ),
            target_tolerance_mm=float(
                self.get_parameter("rebound_target_tolerance_mm").value
            ),
            motion_timeout_sec=float(
                self.get_parameter("rebound_motion_timeout_sec").value
            ),
            service_timeout_sec=float(
                self.get_parameter("rebound_service_timeout_sec").value
            ),
            stop_state_moving_grace_sec=float(
                self.get_parameter("rebound_stop_state_moving_grace_sec").value
            ),
            tool_z_direction_sign=int(
                self.get_parameter("rebound_tool_z_direction_sign").value
            ),
            rehit_enabled=bool(
                self.get_parameter("rebound_rehit_enabled").value
            ),
            rehit_arm_return_progress_mm=float(
                self.get_parameter("rebound_rehit_arm_return_progress_mm").value
            ),
        )

    def _validate_compliance_stabilization_config(self) -> None:
        if not self._manager.compliance.config.enabled:
            return
        maximum_total_force = float(
            self.get_parameter("maximum_total_force_n").value
        )
        maximum_total_torque = float(
            self.get_parameter("maximum_total_torque_nm").value
        )
        maximum_activation_total_force = float(
            self.get_parameter("maximum_activation_total_force_n").value
        )
        maximum_activation_total_torque = float(
            self.get_parameter("maximum_activation_total_torque_nm").value
        )
        translation_velocity = float(
            self.get_parameter(
                "compliance_stabilization_velocity_tolerance_mm_s"
            ).value
        )
        angular_velocity = float(
            self.get_parameter(
                "compliance_stabilization_angular_velocity_tolerance_deg_s"
            ).value
        )
        timeout_ms = float(
            self.get_parameter("compliance_stabilization_timeout_ms").value
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                maximum_total_force,
                maximum_total_torque,
                maximum_activation_total_force,
                maximum_activation_total_torque,
                translation_velocity,
                angular_velocity,
                timeout_ms,
            )
        ):
            raise ValueError(
                "compliance absolute wrench limits, stabilization velocity "
                "tolerances, and timeout must be finite and positive"
            )
        calibration_duration_ms = float(
            self.get_parameter("calibration_duration_ms").value
        )
        if timeout_ms <= calibration_duration_ms:
            raise ValueError(
                "compliance stabilization timeout must exceed wrench "
                "calibration duration"
            )

    def _make_compliance_config(self) -> ComplianceConfig:
        try:
            stiffness_value = self.get_parameter("compliance_stiffness").value
        except ParameterUninitializedException:
            stiffness_value = ()
        return ComplianceConfig(
            enabled=bool(self.get_parameter("compliance_enabled").value),
            stiffness_verified=bool(
                self.get_parameter("compliance_stiffness_verified").value
            ),
            stiffness=tuple(
                float(value)
                for value in stiffness_value
            ),
            reference_id=int(self.get_parameter("compliance_reference_id").value),
            activation_time_sec=float(
                self.get_parameter("compliance_activation_time_sec").value
            ),
            service_timeout_sec=float(
                self.get_parameter("compliance_service_timeout_sec").value
            ),
            maximum_tcp_displacement_mm=float(
                self.get_parameter("maximum_tcp_displacement_mm").value
            ),
            maximum_tcp_angular_displacement_deg=float(
                self.get_parameter(
                    "maximum_tcp_angular_displacement_deg"
                ).value
            ),
            maximum_activation_tcp_displacement_mm=float(
                self.get_parameter(
                    "maximum_activation_tcp_displacement_mm"
                ).value
            ),
            maximum_activation_tcp_angular_displacement_deg=float(
                self.get_parameter(
                    "maximum_activation_tcp_angular_displacement_deg"
                ).value
            ),
            release_max_attempts=int(
                self.get_parameter("release_max_attempts").value
            ),
            release_retry_interval_sec=float(
                self.get_parameter("release_retry_interval_sec").value
            ),
            require_normal_speed_mode=bool(
                self.get_parameter("require_normal_speed_mode").value
            ),
        )

    def _start_session(self) -> str:
        self._processor.start_test()
        self._return_samples.clear()
        self._return_hit_id = None
        self._session_maximum_tcp_displacement_mm = 0.0
        self._session_maximum_tcp_angular_displacement_deg = 0.0
        self._session_maximum_total_force_n = 0.0
        self._session_maximum_total_torque_nm = 0.0
        session_id = self._record_logger.start_session(
            mode="REAL_RT_ANALYSIS", pose_name="PUNCHING_MITT"
        )
        self.get_logger().info(f"Started hit record session {session_id}")
        return session_id

    def _make_config(self) -> HitAnalyzerConfig:
        names = (
            "calibration_duration_ms",
            "minimum_calibration_samples",
            "maximum_zero_force_stddev_n",
            "maximum_zero_moment_stddev_nm",
            "normal_sign",
            "sign_x",
            "sign_y",
            "start_force_n",
            "end_force_n",
            "stable_force_n",
            "minimum_hit_duration_ms",
            "maximum_hit_duration_ms",
            "end_stable_time_ms",
            "debounce_ms",
            "minimum_position_force_n",
            "minimum_position_samples",
            "perfect_radius_mm",
            "direction_deadband_mm",
            "mitt_width_mm",
            "mitt_height_mm",
            "preview_force_warning_n",
        )
        values = {name: self.get_parameter(name).value for name in names}
        return HitAnalyzerConfig(**values)

    def _declare_parameters(self) -> None:
        defaults = {
            "input_topic": "/mitt/rt_sample",
            "output_topic": "/mitt/hit_result",
            "expected_frame_id": "mitt_tool_corrected",
            "calibration_duration_ms": 3000.0,
            "minimum_calibration_samples": 50,
            "maximum_zero_force_stddev_n": 0.5,
            "maximum_zero_moment_stddev_nm": 0.05,
            "normal_sign": -1,
            "sign_x": 1.0,
            "sign_y": 1.0,
            "start_force_n": 10.0,
            "end_force_n": 5.0,
            "stable_force_n": 3.0,
            "minimum_hit_duration_ms": 10.0,
            "maximum_hit_duration_ms": 300.0,
            "end_stable_time_ms": 20.0,
            "debounce_ms": 150.0,
            "minimum_position_force_n": 8.0,
            "minimum_position_samples": 2,
            "perfect_radius_mm": 20.0,
            "direction_deadband_mm": 10.0,
            "mitt_width_mm": 190.0,
            "mitt_height_mm": 150.0,
            "preview_force_warning_n": 30.0,
            "scoring_valid_radius_mm": 100.0,
            "record_dir": "data/hit_records",
            "save_contact_samples": True,
            # When enabled, StartHitTest target_hit_count=0 runs until Stop.
            "allow_continuous_session": False,
            "compliance_enabled": False,
            "compliance_stiffness_verified": False,
            "compliance_reference_id": 1,
            "compliance_activation_time_sec": 0.5,
            "compliance_service_timeout_sec": 0.2,
            # Zero means unset. Compliance activation is rejected until an
            # independently approved displacement limit is supplied.
            "maximum_tcp_displacement_mm": 0.0,
            "maximum_tcp_angular_displacement_deg": 0.0,
            # Separate fail-closed transition bounds. Punch-mode limits are
            # applied only after the settled TCP reference is recaptured.
            "maximum_activation_tcp_displacement_mm": 0.0,
            "maximum_activation_tcp_angular_displacement_deg": 0.0,
            # Independent absolute watchdogs. These include static residual
            # wrench and therefore cannot be replaced by delta hit thresholds.
            "maximum_total_force_n": 0.0,
            "maximum_total_torque_nm": 0.0,
            "maximum_activation_total_force_n": 0.0,
            "maximum_activation_total_torque_nm": 0.0,
            "release_max_attempts": 3,
            "release_retry_interval_sec": 0.05,
            # Fail closed until post-activation settling has been measured.
            "compliance_stabilization_velocity_tolerance_mm_s": 0.0,
            "compliance_stabilization_angular_velocity_tolerance_deg_s": 0.0,
            "compliance_stabilization_timeout_ms": 0.0,
            # Return-to-reference values are intentionally unset. A compliance
            # hit session cannot start until independently verified values are supplied.
            "return_position_tolerance_mm": 0.0,
            "return_velocity_tolerance_mm_s": 0.0,
            "return_force_tolerance_n": 0.0,
            "return_settle_time_ms": 100.0,
            "return_timeout_ms": 2000.0,
            "rt_sample_timeout_ms": 100.0,
            "require_normal_speed_mode": False,
            # Explicit MoveLine rebound is opt-in. Zero-valued motion settings
            # are fail-closed until a dedicated robot profile supplies them.
            "commanded_rebound_enabled": False,
            "rebound_minimum_distance_mm": 0.0,
            "rebound_maximum_distance_mm": 0.0,
            "rebound_force_for_maximum_distance_n": 0.0,
            "rebound_retreat_velocity_mm_s": 0.0,
            "rebound_retreat_acceleration_mm_s2": 0.0,
            "rebound_return_velocity_mm_s": 0.0,
            "rebound_return_acceleration_mm_s2": 0.0,
            "rebound_target_tolerance_mm": 0.0,
            "rebound_motion_timeout_sec": 0.0,
            "rebound_service_timeout_sec": 0.2,
            "rebound_stop_state_moving_grace_sec": 0.5,
            "intentional_motion_timeout_sec": 12.0,
            "rebound_tool_z_direction_sign": -1,
            "rebound_rehit_enabled": True,
            "rebound_rehit_arm_return_progress_mm": 0.0,
            "compliance_service": "/dsr01/force/task_compliance_ctrl",
            "release_compliance_service": "/dsr01/force/release_compliance_ctrl",
            "rt_service": "/dsr01/realtime/read_data_rt",
            "speed_mode_service": "/dsr01/system/get_robot_speed_mode",
            "rebound_move_line_service": "/dsr01/motion/move_line",
            "rebound_move_stop_service": "/dsr01/motion/move_stop",
            "collision_sensitivity_override_enabled": False,
            "training_collision_sensitivity_percent": 40,
            "normal_collision_sensitivity_percent": 40,
            "collision_sensitivity_service_timeout_sec": 0.2,
            "collision_sensitivity_service": (
                "/dsr01/system/change_collision_sensitivity"
            ),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter(
            "compliance_stiffness", Parameter.Type.DOUBLE_ARRAY
        )

    def _on_motion_rezero(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        with self._intentional_motion_lock:
            if self._intentional_motion_active:
                response.success = False
                response.message = "cannot change rezero mode during intentional motion"
                return response
            self._next_motion_rezero = bool(request.data)
        response.success = True
        response.message = (
            "next intentional motion will recalibrate wrench zero"
            if request.data
            else "next intentional motion will retain wrench zero"
        )
        return response

    def _on_recalibrate_zero(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if not request.data:
            response.success = False
            response.message = "recalibrate_zero requires data=true"
            return response
        with self._intentional_motion_lock:
            if self._intentional_motion_active or self._post_move_zero_active:
                response.success = False
                response.message = "cannot recalibrate wrench zero during motion"
                return response
            result = self._manager.begin_calibration()
            if not result.success:
                response.success = False
                response.message = result.message
                return response
            self._processor.begin_zero_recalibration()
            self._ready_logged = False
            self._robot_ready = False
        response.success = True
        response.message = result.message
        self._publish_state()
        return response

    def _on_motion_guard(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        with self._intentional_motion_lock:
            if request.data:
                if self._post_move_zero_active:
                    response.success = False
                    response.message = "post-move wrench zero calibration is active"
                    return response
                if not self._manager.session_active:
                    response.success = False
                    response.message = "hit session is not active"
                    return response
                if self._manager.state is not SystemState.WAITING_FOR_HIT:
                    response.success = False
                    response.message = (
                        "force system is not WAITING_FOR_HIT: "
                        + self._manager.state.value
                    )
                    return response
                if not self._manager.accepting_hits:
                    response.success = False
                    response.message = "force system is not accepting hits"
                    return response
                if self._rebound_motion.moving:
                    response.success = False
                    response.message = "force rebound motion is already active"
                    return response
                timeout_sec = float(
                    self.get_parameter("intentional_motion_timeout_sec").value
                )
                if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
                    response.success = False
                    response.message = "intentional motion timeout must be positive"
                    return response
                self._intentional_motion_timeout_sec = timeout_sec
                self._intentional_motion_active = True
                self._intentional_motion_deadline = time.monotonic() + timeout_sec
                self._intentional_reference_pending = False
                self._active_motion_rezero = self._next_motion_rezero
                self._next_motion_rezero = False
                self._manager.accepting_hits = False
                response.success = True
                response.message = (
                    f"intentional mitt motion armed for {timeout_sec:.1f} s"
                )
                self.get_logger().info("INTENTIONAL MITT MOTION: ARMED")
                self._publish_state()
                return response

            self._intentional_motion_active = False
            self._intentional_motion_deadline = 0.0
            self._settled_motion_rezero = self._active_motion_rezero
            self._active_motion_rezero = False
            self._intentional_reference_pending = bool(
                self._manager.session_active
                and self._manager.compliance.active
                and self._manager.state is not SystemState.ERROR
            )
            if (
                not self._intentional_reference_pending
                and self._manager.session_active
                and self._manager.state is not SystemState.ERROR
            ):
                self._manager.accepting_hits = True
                self._publish_state()
            self._reset_rt_watchdog_after_motion_service()
            self._manager.compliance._moving_allowed_until = time.monotonic() + 0.5  # noqa: SLF001
            response.success = True
            response.message = "intentional mitt motion disarmed"
            self.get_logger().info(
                "INTENTIONAL MITT MOTION: DISARMED; waiting for settled reference"
            )
            return response

    @staticmethod
    def _valid_intentional_motion_pose(
        message: RtMittSample,
    ) -> tuple[float, ...] | None:
        try:
            pose = tuple(float(value) for value in message.tcp_pose_mm_deg)
        except (TypeError, ValueError):
            return None
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            return None
        return pose

    def _capture_intentional_motion_reference(self, pose: tuple[float, ...]) -> bool:
        try:
            self._manager.compliance.recapture_reference_tcp_position(pose)
        except (RuntimeError, ValueError) as error:
            self.get_logger().error(
                f"Intentional motion reference capture failed: {error}"
            )
            return False
        self._compliance_reference_pose_mm_deg = pose
        return True

    def _begin_post_move_zero_recalibration(self) -> None:
        self._manager.accepting_hits = False
        self._processor.begin_zero_recalibration()
        self._post_move_zero_active = True
        self._publish_state()
        self.get_logger().info(
            "POST-MOVE WRENCH ZERO: keep mitt untouched while baseline is collected"
        )

    def _on_rt_sample(self, message: RtMittSample) -> None:
        pose = self._valid_intentional_motion_pose(message)
        now = time.monotonic()
        with self._intentional_motion_lock:
            if self._intentional_reference_pending and (
                not self._manager.session_active
                or not self._manager.compliance.active
                or self._manager.state is SystemState.ERROR
            ):
                self._intentional_reference_pending = False

            if self._intentional_motion_active and now >= self._intentional_motion_deadline:
                self._intentional_motion_active = False
                self._settled_motion_rezero = self._active_motion_rezero
                self._active_motion_rezero = False
                self._intentional_reference_pending = True
                self.get_logger().error(
                    "INTENTIONAL MITT MOTION: "
                    f"{self._intentional_motion_timeout_sec:.1f} s timeout; guard closed"
                )

            if self._intentional_motion_active and pose is not None:
                self._manager.compliance._moving_allowed_until = now + 0.25  # noqa: SLF001
                self._capture_intentional_motion_reference(pose)
            elif (
                self._intentional_reference_pending
                and int(message.robot_state) == 1
                and not self._rebound_motion.moving
                and pose is not None
            ):
                if self._capture_intentional_motion_reference(pose):
                    self._intentional_reference_pending = False
                    self.get_logger().info(
                        "INTENTIONAL MITT MOTION REFERENCE: settled TCP captured"
                    )
                    if self._settled_motion_rezero:
                        self._begin_post_move_zero_recalibration()
                    else:
                        self._manager.accepting_hits = True
                        self._publish_state()
                    self._settled_motion_rezero = False

        self._on_rt_sample_core(message)

        with self._intentional_motion_lock:
            if (
                self._post_move_zero_active
                and self._processor.calibrated
                and self._manager.session_active
                and self._manager.state is SystemState.WAITING_FOR_HIT
            ):
                self._post_move_zero_active = False
                self._manager.accepting_hits = True
                self._publish_state()
                self.get_logger().info(
                    "POST-MOVE WRENCH ZERO COMPLETE: next hit is enabled"
                )
            elif self._post_move_zero_active and (
                not self._manager.session_active
                or self._manager.state is SystemState.ERROR
            ):
                self._post_move_zero_active = False

    def _on_rt_sample_core(self, message: RtMittSample) -> None:
        if message.frame_id != self._expected_frame_id:
            self._log_throttled(
                "warning",
                f"Ignoring frame '{message.frame_id}'; expected "
                f"'{self._expected_frame_id}'",
            )
            return
        values = tuple(float(value) for value in message.corrected_wrench)
        pose = tuple(float(value) for value in message.tcp_pose_mm_deg)
        velocity = tuple(float(value) for value in message.tcp_velocity_mm_deg_s)
        if any(len(item) != 6 for item in (values, pose, velocity)) or not all(
            math.isfinite(value)
            for item in (values, pose, velocity)
            for value in item
        ):
            self._log_throttled("warning", "Ignoring invalid RT sample")
            return
        timestamp_ns = (
            int(message.stamp.sec) * 1_000_000_000
            + int(message.stamp.nanosec)
        )
        self._last_rt_sample_monotonic_ns = time.monotonic_ns()

        # STOPPING is an intentional, bounded teardown in which the controller
        # may serialize RT reads while release_compliance_ctrl completes. The
        # release result itself determines whether shutdown succeeded.
        if self._manager.state is SystemState.STOPPING:
            return

        displacement_mm = 0.0
        angular_displacement_deg = 0.0
        if self._manager.session_active and self._manager.compliance.config.enabled:
            health = self._manager.compliance.observe_rt_state(
                int(message.robot_state),
                pose,
                allow_moving=(
                    True if self._rebound_motion.allows_state_moving else None
                ),
            )
            self._robot_ready = (
                health.healthy
                and self._manager.state
                is not SystemState.STABILIZING_COMPLIANCE
            )
            displacement_mm = health.tcp_displacement_mm
            angular_displacement_deg = health.tcp_angular_displacement_deg
            self._session_maximum_tcp_displacement_mm = max(
                self._session_maximum_tcp_displacement_mm, displacement_mm
            )
            self._session_maximum_tcp_angular_displacement_deg = max(
                self._session_maximum_tcp_angular_displacement_deg,
                angular_displacement_deg,
            )
            total_force, total_torque = self._absolute_wrench_magnitudes(values)
            self._session_maximum_total_force_n = max(
                self._session_maximum_total_force_n, total_force
            )
            self._session_maximum_total_torque_nm = max(
                self._session_maximum_total_torque_nm, total_torque
            )
            if not health.healthy:
                if self._manager.state is SystemState.RETURNING_TO_REFERENCE:
                    self._append_return_observation(
                        timestamp_ns,
                        values,
                        pose,
                        velocity,
                        displacement_mm,
                        int(message.robot_state),
                    )
                self._fault_session(
                    health.detail,
                    displacement_mm,
                    angular_displacement_deg,
                )
                return
            wrench_fault = self._absolute_wrench_limit_fault(
                values,
                activation=(
                    self._manager.state
                    is SystemState.STABILIZING_COMPLIANCE
                ),
            )
            if wrench_fault is not None:
                self._fault_session(
                    wrench_fault,
                    displacement_mm,
                    angular_displacement_deg,
                )
                return

        # A faulted session is terminal. In particular, a failed compliance
        # stabilization must not restart zero collection and print misleading
        # CALIBRATING messages while the public state remains ERROR.
        if self._manager.state is SystemState.ERROR:
            return

        if self._manager.state is SystemState.RETURNING_TO_REFERENCE:
            if self._should_interrupt_return_for_contact(values, displacement_mm):
                if not self._interrupt_return_for_contact(displacement_mm):
                    return
                # Fall through so this same RT sample starts the next hit.
            else:
                self._process_return_sample(
                    timestamp_ns,
                    values,
                    pose,
                    velocity,
                    displacement_mm,
                    int(message.robot_state),
                )
                return

        if self._manager.state is SystemState.STABILIZING_COMPLIANCE:
            self._process_compliance_stabilization(
                timestamp_ns,
                values,
                pose,
                velocity,
                int(message.robot_state),
            )
            return

        was_calibrated = self._processor.calibrated
        if was_calibrated and not self._manager.accepting_hits:
            return
        if (
            was_calibrated
            and self._manager.state is SystemState.WAITING_FOR_HIT
        ):
            normal_force = self._processor.compressive_normal_force(values)
            start_force = float(self.get_parameter("start_force_n").value)
            if normal_force >= start_force:
                if not self._start_rebound_on_contact(
                    normal_force, displacement_mm
                ):
                    return
        result = self._processor.process(timestamp_ns, values)
        if not self._processor.calibrated:
            self._log_throttled(
                "info",
                "CALIBRATING: keep mitt untouched "
                f"({self._processor.calibration_sample_count} samples)",
            )
            return
        if not was_calibrated and not self._ready_logged:
            ready_instruction = (
                "call /mitt/start_test, then keep the mitt untouched until "
                "COMPLIANCE READY and WAITING_FOR_HIT"
                if self._manager.compliance.config.enabled
                else "tap gently by hand; do not punch (preview sampling only)"
            )
            self.get_logger().info(
                f"READY: zero calibration complete; {ready_instruction}"
            )
            self._ready_logged = True
            self._manager.calibration_ready()
            self._robot_ready = not self._manager.compliance.config.enabled
        if result is not None and self._manager.begin_analysis():
            self._publish_state()
            requires_return = self._requires_return_after_hit()
            complete_before_publish = bool(
                not requires_return and not self._session_rebound_enabled
            )
            stopped = None
            if complete_before_publish:
                # A combination HitResult is the contact-release event. Put
                # the force system back in WAITING_FOR_HIT before publishing
                # it so the next guarded pose move can start immediately.
                stopped = self._manager.complete_hit(require_return=False)
                self._publish_state()
            published = self._publish(message, result)
            self._record(result, published)
            self._log_result(result, published.accuracy_score)
            if requires_return:
                stopped = self._manager.complete_hit(require_return=True)
                if self._return_monitor is None:
                    self._fault_session(
                        "return-to-reference monitor is not configured",
                        displacement_mm,
                    )
                    return
                if self._rebound_motion.config.enabled:
                    # Normally started on the first threshold-crossing sample.
                    # Keep a completion-time fallback for synthetic/unit paths
                    # or a contact admitted before commanded rebound was ready.
                    if not self._start_rebound_on_contact(
                        result.peak_normal_force_n, displacement_mm
                    ):
                        return
                self._return_samples.clear()
                self._return_hit_id = int(result.hit_id)
                self._return_monitor.start(timestamp_ns)
            elif not complete_before_publish:
                stopped = self._manager.complete_hit(require_return=False)
            if stopped is not None:
                self.get_logger().info(
                    "Automatic Stop: target hit count reached; "
                    f"{self._manager.last_release_message}"
                )
                self._save_compliance_session_summary(
                    "TEST_COMPLETE", self._manager.detail
                )
                self._record_logger.end_session()
                self._restore_collision_sensitivity("automatic stop")
            self._publish_state()

    def _requires_return_after_hit(self) -> bool:
        return bool(
            self._manager.compliance.config.enabled
            and self._session_rebound_enabled
        )

    def _start_rebound_on_contact(
        self, normal_force_n: float, displacement_mm: float
    ) -> bool:
        if (
            not getattr(self, "_session_rebound_enabled", True)
            or not self._rebound_motion.config.enabled
            or self._rebound_motion.moving
        ):
            return True
        if self._compliance_reference_pose_mm_deg is None:
            self._fault_session(
                "commanded rebound reference pose is missing", displacement_mm
            )
            return False
        success, detail = self._rebound_motion.start_retreat(
            normal_force_n,
            self._compliance_reference_pose_mm_deg,
        )
        if not success:
            self._fault_session(detail, displacement_mm)
            return False
        self._reset_rt_watchdog_after_motion_service()
        self.get_logger().info(
            "IMMEDIATE REBOUND: contact threshold crossed; " + detail
        )
        return True

    def _should_interrupt_return_for_contact(
        self, wrench: tuple[float, ...], displacement_mm: float
    ) -> bool:
        config = self._rebound_motion.config
        if not (config.enabled and config.rehit_enabled and self._rebound_motion.moving):
            return False
        # The public state changes to RETURNING_TO_REFERENCE as soon as the
        # contact record closes, while the commanded motion may still be in
        # its RETREATING phase. Impact rebound then briefly crosses the contact
        # threshold again. Never soft-stop the active retreat for that pulse;
        # admit a new hit only during the actual commanded return leg.
        if self._rebound_motion.phase is not ReboundPhase.RETURNING:
            return False
        arm_displacement_mm = max(
            self._rebound_motion.commanded_distance_mm
            - config.rehit_arm_return_progress_mm,
            0.0,
        )
        if displacement_mm > arm_displacement_mm:
            return False
        if (
            self._manager.target_hit_count > 0
            and self._manager.hit_count >= self._manager.target_hit_count
        ):
            return False
        normal_force = self._processor.compressive_normal_force(wrench)
        return normal_force >= float(self.get_parameter("start_force_n").value)

    def _interrupt_return_for_contact(self, displacement_mm: float) -> bool:
        stopped, stop_detail = self._rebound_motion.soft_stop()
        if not stopped:
            self._fault_session(stop_detail, displacement_mm)
            return False
        self._record_return("INTERRUPTED", "new contact during commanded return")
        transition = self._manager.interrupt_return_for_hit()
        if not transition.success:
            self._fault_session(transition.message, displacement_mm)
            return False
        self._return_samples.clear()
        self._return_hit_id = None
        self.get_logger().info(
            "RETURN INTERRUPTED: new contact admitted; " + stop_detail
        )
        self._publish_state()
        return True

    def _process_return_sample(
        self,
        timestamp_ns: int,
        wrench: tuple[float, ...],
        pose: tuple[float, ...],
        velocity: tuple[float, ...],
        displacement_mm: float,
        robot_state: int,
    ) -> None:
        if self._return_monitor is None:
            self._fault_session(
                "return-to-reference monitor is not configured", displacement_mm
            )
            return
        observation = self._append_return_observation(
            timestamp_ns,
            wrench,
            pose,
            velocity,
            displacement_mm,
            robot_state,
        )
        if self._rebound_motion.config.enabled:
            if self._rebound_motion.timed_out():
                self._fault_session("commanded rebound motion timeout", displacement_mm)
                return
            if self._rebound_motion.phase is ReboundPhase.RETREATING:
                if not self._rebound_motion.retreat_target_reached(pose):
                    return
                if not self._start_commanded_return(displacement_mm):
                    return
                return
        status = self._return_monitor.process(
            timestamp_ns,
            displacement_mm=displacement_mm,
            translation_speed_mm_s=observation.translation_speed_mm_s,
            normal_force_n=observation.normal_force_n,
        )
        if status is ReturnStatus.WAITING:
            return
        if status is ReturnStatus.TIMEOUT:
            self._fault_session(
                "return-to-reference timeout",
                displacement_mm,
            )
            return
        stopped = self._manager.complete_return()
        self._rebound_motion.finish()
        self._record_return("SETTLED", "return conditions remained stable")
        if stopped is not None:
            self.get_logger().info(
                "Automatic Stop after reference return: "
                f"{self._manager.last_release_message}"
            )
            self._save_compliance_session_summary(
                "TEST_COMPLETE", self._manager.detail
            )
            self._record_logger.end_session()
            self._restore_collision_sensitivity("automatic stop")
        self._publish_state()

    def _start_commanded_return(self, displacement_mm: float) -> bool:
        success, detail = self._rebound_motion.start_return()
        if not success:
            self._fault_session(detail, displacement_mm)
            return False
        self._reset_rt_watchdog_after_motion_service()
        self.get_logger().info(
            "REBOUND RETURN: MoveLine started; fixed compliance stiffness retained"
        )
        return True

    def _absolute_wrench_limit_fault(
        self, wrench: tuple[float, ...], *, activation: bool = False
    ) -> str | None:
        total_force, total_torque = HitAnalyzerNode._absolute_wrench_magnitudes(
            wrench
        )
        force_parameter = (
            "maximum_activation_total_force_n"
            if activation
            else "maximum_total_force_n"
        )
        torque_parameter = (
            "maximum_activation_total_torque_nm"
            if activation
            else "maximum_total_torque_nm"
        )
        maximum_force = float(self.get_parameter(force_parameter).value)
        maximum_torque = float(self.get_parameter(torque_parameter).value)
        phase = "activation " if activation else ""
        if total_force > maximum_force:
            return (
                f"absolute {phase}total force exceeded limit: "
                f"{total_force:.3f} > {maximum_force:.3f} N"
            )
        if total_torque > maximum_torque:
            return (
                f"absolute {phase}total torque exceeded limit: "
                f"{total_torque:.3f} > {maximum_torque:.3f} Nm"
            )
        return None

    @staticmethod
    def _absolute_wrench_magnitudes(
        wrench: tuple[float, ...]
    ) -> tuple[float, float]:
        return (
            math.sqrt(sum(value * value for value in wrench[:3])),
            math.sqrt(sum(value * value for value in wrench[3:])),
        )

    def _process_compliance_stabilization(
        self,
        timestamp_ns: int,
        wrench: tuple[float, ...],
        pose: tuple[float, ...],
        velocity: tuple[float, ...],
        robot_state: int,
    ) -> None:
        started_ns = self._compliance_stabilization_started_monotonic_ns
        if started_ns is None:
            self._fault_session(
                "compliance stabilization timer was not initialized", 0.0
            )
            return
        timeout_ns = int(
            float(
                self.get_parameter("compliance_stabilization_timeout_ms").value
            )
            * 1e6
        )
        if time.monotonic_ns() - started_ns > timeout_ns:
            self._fault_session("compliance stabilization timeout", 0.0)
            return

        translation_speed = math.sqrt(sum(value * value for value in velocity[:3]))
        angular_speed = math.sqrt(sum(value * value for value in velocity[3:]))
        translation_limit = float(
            self.get_parameter(
                "compliance_stabilization_velocity_tolerance_mm_s"
            ).value
        )
        angular_limit = float(
            self.get_parameter(
                "compliance_stabilization_angular_velocity_tolerance_deg_s"
            ).value
        )
        stable = (
            robot_state == self._manager.compliance.STATE_STANDBY
            and translation_speed <= translation_limit
            and angular_speed <= angular_limit
        )
        if not stable:
            self._processor.begin_zero_recalibration()
            self._log_throttled(
                "info",
                "STABILIZING_COMPLIANCE: waiting for stationary no-contact "
                f"state (v={translation_speed:.3f} mm/s, "
                f"w={angular_speed:.3f} deg/s, state={robot_state})",
            )
            return

        self._processor.process(timestamp_ns, wrench)
        calibration = self._processor.zero_calibration
        if calibration is None:
            self._log_throttled(
                "info",
                "STABILIZING_COMPLIANCE: collecting post-activation wrench "
                f"baseline ({self._processor.calibration_sample_count} samples)",
            )
            return

        activation_maximum_displacement_mm = float(
            self._manager.compliance.maximum_observed_displacement_mm
        )
        activation_maximum_angular_displacement_deg = float(
            self._manager.compliance.maximum_observed_angular_displacement_deg
        )
        try:
            reference = self._manager.compliance.recapture_reference_tcp_position(
                pose
            )
        except (RuntimeError, ValueError) as error:
            self._fault_session(
                f"failed to capture settled compliance reference: {error}", 0.0
            )
            return
        self._compliance_reference_pose_mm_deg = tuple(pose)
        result = self._manager.compliance_stabilized()
        if not result.success:
            self._fault_session(result.message, 0.0)
            return
        self._compliance_stabilization_started_monotonic_ns = None
        self._robot_ready = True
        standard_deviation = tuple(float(value) for value in calibration.stddev)
        three_sigma_range = tuple(
            (
                float(offset - 3.0 * deviation),
                float(offset + 3.0 * deviation),
            )
            for offset, deviation in zip(calibration.offset, standard_deviation)
        )
        saved, save_error = self._record_logger.log_compliance_baseline(
            {
                "created_at": calibration.created_at,
                "sample_count": int(calibration.sample_count),
                "activation_maximum_tcp_displacement_mm": (
                    activation_maximum_displacement_mm
                ),
                "activation_maximum_tcp_angular_displacement_deg": (
                    activation_maximum_angular_displacement_deg
                ),
                "activation_maximum_total_force_n": (
                    self._session_maximum_total_force_n
                ),
                "activation_maximum_total_torque_nm": (
                    self._session_maximum_total_torque_nm
                ),
                "tcp_reference_mm": list(reference),
                "tcp_reference_mm_deg": list(pose),
                "wrench_offset": list(calibration.offset),
                "wrench_stddev": list(standard_deviation),
                "wrench_three_sigma_range": [
                    list(bounds) for bounds in three_sigma_range
                ],
            }
        )
        if not saved:
            self.get_logger().error(
                f"Failed to save compliance baseline: {save_error}"
            )
        self.get_logger().info(
            "COMPLIANCE READY: post-activation TCP and wrench baselines captured; "
            f"reference=({reference[0]:.3f}, {reference[1]:.3f}, "
            f"{reference[2]:.3f}) mm; activation_max_displacement="
            f"{activation_maximum_displacement_mm:.3f} mm; "
            f"activation_max_angular_displacement="
            f"{activation_maximum_angular_displacement_deg:.3f} deg; "
            f"wrench_offset="
            f"{tuple(round(value, 4) for value in calibration.offset)}; "
            "wrench_stddev="
            f"{tuple(round(value, 4) for value in standard_deviation)}"
        )
        self._publish_state()

    def _append_return_observation(
        self,
        timestamp_ns: int,
        wrench: tuple[float, ...],
        pose: tuple[float, ...],
        velocity: tuple[float, ...],
        displacement_mm: float,
        robot_state: int,
    ) -> ReturnObservation:
        position = tuple(float(value) for value in pose[:3])
        linear_velocity = tuple(float(value) for value in velocity[:3])
        observation = ReturnObservation(
            timestamp_ns=timestamp_ns,
            tcp_position_mm=position,  # type: ignore[arg-type]
            tcp_velocity_mm_s=linear_velocity,  # type: ignore[arg-type]
            displacement_mm=float(displacement_mm),
            translation_speed_mm_s=math.sqrt(
                sum(value * value for value in linear_velocity)
            ),
            normal_force_n=self._processor.compressive_normal_force(wrench),
            robot_state=int(robot_state),
        )
        self._return_samples.append(observation)
        return observation

    def _record_return(self, outcome: str, detail: str) -> None:
        if self._return_hit_id is None:
            return
        samples = tuple(self._return_samples)
        duration_ms = (
            (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e6
            if len(samples) >= 2
            else 0.0
        )
        success, error = self._record_logger.log_return(
            self._return_hit_id,
            {
                "outcome": str(outcome),
                "detail": str(detail),
                "duration_ms": duration_ms,
                "sample_count": len(samples),
                "max_displacement_mm": max(
                    (sample.displacement_mm for sample in samples), default=0.0
                ),
                "max_translation_speed_mm_s": max(
                    (sample.translation_speed_mm_s for sample in samples),
                    default=0.0,
                ),
                "max_normal_force_n": max(
                    (sample.normal_force_n for sample in samples), default=0.0
                ),
            },
            samples,
        )
        if not success:
            self.get_logger().error(f"Failed to save return record: {error}")
        self._return_samples.clear()
        self._return_hit_id = None

    def _fault_session(
        self,
        detail: str,
        displacement_mm: float,
        angular_displacement_deg: float = 0.0,
    ) -> None:
        was_returning = self._manager.state is SystemState.RETURNING_TO_REFERENCE
        if self._rebound_motion.moving:
            stopped, stop_detail = self._rebound_motion.soft_stop()
            if not stopped:
                detail = f"{detail}; {stop_detail}"
        self._compliance_stabilization_started_monotonic_ns = None
        self._robot_ready = False
        self._manager.fault(detail)
        self._restore_collision_sensitivity("safety fault")
        if was_returning:
            outcome = "TIMEOUT" if "timeout" in detail.lower() else "FAULT"
            self._record_return(outcome, detail)
        self._save_compliance_session_summary("FAULT", self._manager.detail)
        self.get_logger().error(
            f"Safety fault: {self._manager.detail}; "
            f"displacement={displacement_mm:.3f} mm; "
            f"angular_displacement={angular_displacement_deg:.3f} deg; "
            f"{self._manager.last_release_message}"
        )
        self._record_logger.end_session()
        self._publish_state()

    def _on_start(
        self, request: StartHitTest.Request, response: StartHitTest.Response
    ) -> StartHitTest.Response:
        if not self._rebound_motion.services_ready():
            response.success = False
            response.message = "required rebound motion service is unavailable"
            self.get_logger().warning(
                f"StartHitTest: success=False: {response.message}"
            )
            return response
        if not self._collision_sensitivity.service_ready():
            response.success = False
            response.message = "collision sensitivity service is unavailable"
            self.get_logger().warning(
                f"StartHitTest: success=False: {response.message}"
            )
            return response
        sensitivity_success, sensitivity_detail = (
            self._collision_sensitivity.apply_training()
        )
        if not sensitivity_success:
            response.success = False
            response.message = sensitivity_detail
            self.get_logger().warning(
                f"StartHitTest: success=False: {response.message}"
            )
            return response
        self.get_logger().info(sensitivity_detail)
        self._rebound_motion.finish()
        self._compliance_reference_pose_mm_deg = None
        result = self._manager.start(int(request.target_hit_count))
        if not result.success:
            restored, restore_detail = self._collision_sensitivity.restore()
            if not restored:
                result = type(result)(
                    False, f"{result.message}; {restore_detail}"
                )
        if result.success:
            self._session_rebound_enabled = not bool(
                request.suppress_commanded_rebound
            )
            # Compliance activation can serialize RT service responses. Start
            # the stream deadline from the completed controller service call,
            # not from the last sample received before activation.
            self._reset_rt_watchdog_after_motion_service()
        if (
            result.success
            and self._manager.state is SystemState.STABILIZING_COMPLIANCE
        ):
            self._processor.begin_zero_recalibration()
            self._compliance_stabilization_started_monotonic_ns = (
                time.monotonic_ns()
            )
            self._robot_ready = False
        else:
            self._compliance_stabilization_started_monotonic_ns = None
            self._robot_ready = result.success
        log = self.get_logger().info if result.success else self.get_logger().warning
        log(f"StartHitTest: success={result.success}: {result.message}")
        response.success = result.success
        response.message = result.message
        self._publish_state()
        return response

    def _on_stop(
        self, request: StopHitTest.Request, response: StopHitTest.Response
    ) -> StopHitTest.Response:
        del request
        was_returning = self._manager.state is SystemState.RETURNING_TO_REFERENCE
        motion_stop_success, motion_stop_detail = self._rebound_motion.soft_stop()
        result = self._manager.stop()
        restored, restore_detail = self._collision_sensitivity.restore()
        if not restored:
            result = type(result)(False, f"{result.message}; {restore_detail}")
        if not motion_stop_success:
            result = type(result)(False, f"{result.message}; {motion_stop_detail}")
        self._compliance_stabilization_started_monotonic_ns = None
        if was_returning:
            self._record_return("STOPPED", result.message)
        self._save_compliance_session_summary(
            "STOPPED" if result.success else "FAULT",
            result.message,
        )
        if result.success:
            self._record_logger.end_session()
        log = self.get_logger().info if result.success else self.get_logger().warning
        log(
            f"StopHitTest: success={result.success}: {result.message}; "
            f"{self._manager.last_release_message}"
        )
        response.success = result.success
        response.message = result.message
        self._publish_state()
        return response

    def _on_status_timer(self) -> None:
        self._publish_state()

    def _on_rt_watchdog_timer(self) -> None:
        with self._intentional_motion_lock:
            if (
                self._intentional_motion_active
                and time.monotonic() < self._intentional_motion_deadline
            ):
                return
        if not (
            self._manager.session_active
            and self._manager.compliance.config.enabled
            and self._manager.state is not SystemState.STOPPING
        ):
            return
        timeout_ns = int(
            float(self.get_parameter("rt_sample_timeout_ms").value) * 1e6
        )
        elapsed_ns = time.monotonic_ns() - self._last_rt_sample_monotonic_ns
        if elapsed_ns > timeout_ns:
            self._fault_session("RT sample stream timeout", 0.0)

    def _reset_rt_watchdog_after_motion_service(self) -> None:
        """Restart the RT deadline after a bounded controller motion service."""
        self._last_rt_sample_monotonic_ns = time.monotonic_ns()

    def _publish_state(self) -> None:
        message = SystemStateMessage()
        message.stamp = self.get_clock().now().to_msg()
        message.state = self._manager.state.value
        message.detail = self._manager.detail
        message.robot_ready = self._processor.calibrated and self._robot_ready
        message.compliance_enabled = bool(self._manager.compliance.active)
        message.accepting_hits = self._manager.accepting_hits
        self._state_publisher.publish(message)
        snapshot = (
            message.state,
            message.detail,
            message.robot_ready,
            message.compliance_enabled,
            message.accepting_hits,
        )
        if snapshot != self._last_state_snapshot:
            self.get_logger().info(
                "STATE "
                f"{message.state}: {message.detail}; "
                f"robot_ready={message.robot_ready}, "
                f"compliance_enabled={message.compliance_enabled}, "
                f"accepting_hits={message.accepting_hits}"
            )
            self._last_state_snapshot = snapshot

    def shutdown_session(self) -> None:
        """Release an active session while the ROS executor is still running."""
        if self._shutdown_session_complete:
            return
        was_active = self._manager.accepting_hits or bool(
            self._manager.compliance.active
        )
        was_returning = self._manager.state is SystemState.RETURNING_TO_REFERENCE
        self._compliance_stabilization_started_monotonic_ns = None
        if self._rebound_motion.moving:
            stopped, stop_detail = self._rebound_motion.soft_stop()
            if not stopped:
                self.get_logger().error(stop_detail)
        self._manager.shutdown()
        self._restore_collision_sensitivity("shutdown")
        if was_returning:
            self._record_return("SHUTDOWN", "node shutdown during return")
        self._save_compliance_session_summary(
            "SHUTDOWN", "node shutdown during active compliance session"
        )
        if was_active:
            self.get_logger().info(
                f"Shutdown release: {self._manager.last_release_message}"
            )
        self._record_logger.end_session()
        self._shutdown_session_complete = True

    def _restore_collision_sensitivity(self, reason: str) -> bool:
        success, detail = self._collision_sensitivity.restore()
        log = self.get_logger().info if success else self.get_logger().error
        if detail != "collision sensitivity restore not required":
            log(f"{reason}: {detail}")
        return success

    def _save_compliance_session_summary(
        self, outcome: str, detail: str
    ) -> None:
        if not self._record_logger.session_id:
            return
        if not self._manager.compliance.config.enabled:
            return
        saved, save_error = self._record_logger.log_compliance_summary(
            {
                "outcome": str(outcome),
                "detail": str(detail),
                "hit_count": int(self._manager.hit_count),
                "maximum_tcp_displacement_mm": (
                    self._session_maximum_tcp_displacement_mm
                ),
                "maximum_tcp_angular_displacement_deg": (
                    self._session_maximum_tcp_angular_displacement_deg
                ),
                "maximum_post_reference_tcp_displacement_mm": float(
                    self._manager.compliance.maximum_observed_displacement_mm
                ),
                "maximum_post_reference_tcp_angular_displacement_deg": float(
                    self._manager.compliance.maximum_observed_angular_displacement_deg
                ),
                "maximum_total_force_n": self._session_maximum_total_force_n,
                "maximum_total_torque_nm": self._session_maximum_total_torque_nm,
            }
        )
        if not saved:
            self.get_logger().error(
                f"Failed to save compliance session summary: {save_error}"
            )

    def destroy_node(self) -> bool:
        self.shutdown_session()
        return super().destroy_node()

    def _publish(
        self, source: RtMittSample, result: HitAnalyzerResult
    ) -> HitResult:
        message = HitResult()
        message.stamp = source.stamp
        message.hit_id = result.hit_id
        message.valid_hit = result.valid
        message.invalid_reason = result.reason
        message.hit_direction = result.direction.value
        message.hit_x_mm = result.x_mm
        message.hit_y_mm = result.y_mm
        message.center_error_mm = result.center_error_mm
        message.peak_force_n = result.peak_force_n
        message.peak_normal_force_n = result.peak_normal_force_n
        message.impulse_ns = result.impulse_ns
        message.contact_duration_ms = result.contact_duration_ms
        message.accuracy_score = (
            self._score_calculator.calculate(result.center_error_mm)
            if result.valid
            else 0.0
        )
        message.power_score = 0.0
        message.total_score = message.accuracy_score
        message.force_warning = result.force_warning
        message.safety_stop = False
        self._publisher.publish(message)
        return message

    def _record(self, result: HitAnalyzerResult, message: HitResult) -> None:
        samples = tuple(
            BufferedWrenchSample(
                timestamp_ns=timestamp_ns,
                raw_wrench=wrench,
                filtered_wrench=wrench,
            )
            for timestamp_ns, wrench in result.contact_samples
        )
        success, error = self._record_logger.log_hit(
            {
                "timestamp_ns": int(message.stamp.sec) * 1_000_000_000
                + int(message.stamp.nanosec),
                "hit_id": int(message.hit_id),
                "valid_hit": bool(message.valid_hit),
                "invalid_reason": str(message.invalid_reason),
                "hit_direction": str(message.hit_direction),
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
                "sample_count": int(result.sample_count),
            },
            samples,
        )
        if not success:
            self.get_logger().error(f"Failed to save hit record: {error}")

    def _log_result(
        self, result: HitAnalyzerResult, accuracy_score: float
    ) -> None:
        if result.valid:
            self.get_logger().info(
                f"EVENT #{result.hit_id} {result.direction.value}: "
                f"x={result.x_mm:+.1f} mm, y={result.y_mm:+.1f} mm, "
                f"peak={result.peak_normal_force_n:.1f} N, "
                f"duration={result.contact_duration_ms:.1f} ms, "
                f"impulse={result.impulse_ns:.3f} N*s, "
                f"accuracy={accuracy_score:.1f}/10"
            )
        else:
            self.get_logger().warning(
                f"EVENT #{result.hit_id} INVALID ({result.reason}): "
                f"peak={result.peak_normal_force_n:.1f} N, "
                f"duration={result.contact_duration_ms:.1f} ms"
            )

    def _log_throttled(self, level: str, text: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_log_ns < 1_000_000_000:
            return
        getattr(self.get_logger(), level)(text)
        self._last_log_ns = now_ns

    @staticmethod
    def _resolve_path(path: str) -> Path:
        resolved = Path(path).expanduser()
        return resolved if resolved.is_absolute() else Path.cwd() / resolved


def main(args: list[str] | None = None) -> None:
    # Keep the ROS context and executor alive long enough to receive the
    # release_compliance_ctrl response during Ctrl+C or launch SIGTERM.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HitAnalyzerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def interrupt_main(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_main)
    try:
        while thread.is_alive():
            thread.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown_session()
            executor.remove_node(node)
            executor.shutdown(timeout_sec=2.0)
            thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # launch may deliver SIGINT again during ROS entity destruction.
            pass
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

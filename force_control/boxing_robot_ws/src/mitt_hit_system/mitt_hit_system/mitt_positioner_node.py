"""ROS node for initialising and positioning the boxing mitt."""

import math
import threading
from typing import Any, Sequence

import rclpy
from boxing_interfaces.msg import TargetPose
from boxing_interfaces.srv import MoveMittPose, MoveNamedPose, PreparePersonPose
from dsr_msgs2.srv import Fkin, GetCurrentPosx, Jog, MoveJoint, MoveLine, MoveStop
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from mitt_hit_system.mitt_pose_planner import (
    MittPosePlanner,
    PersonMeasurement,
    PunchType,
)

class MittPositionerNode(Node):
    DR_BASE = 0
    DR_TOOL = 1
    DR_MV_MOD_ABS = 0
    DR_SYNC_SYNC = 0
    DR_SSTOP = 2
    JOG_AXIS_TASK_Z = 8

    def __init__(self) -> None:
        super().__init__("mitt_positioner")
        self._declare_parameters()
        self._motion_lock = threading.Lock()
        callback_group = ReentrantCallbackGroup()
        self._move_joint_client = self.create_client(
            MoveJoint,
            str(self.get_parameter("move_joint_service").value),
            callback_group=callback_group,
        )
        self._move_line_client = self.create_client(
            MoveLine,
            str(self.get_parameter("move_line_service").value),
            callback_group=callback_group,
        )
        self._move_stop_client = self.create_client(
            MoveStop,
            str(self.get_parameter("move_stop_service").value),
            callback_group=callback_group,
        )
        self._jog_client = self.create_client(
            Jog,
            str(self.get_parameter("jog_service").value),
            callback_group=callback_group,
        )
        self._continuous_jog_active = False
        self._fkin_client = self.create_client(
            Fkin,
            str(self.get_parameter("fkin_service").value),
            callback_group=callback_group,
        )
        self._current_posx_client = self.create_client(
            GetCurrentPosx,
            str(self.get_parameter("get_current_posx_service").value),
            callback_group=callback_group,
        )
        self._target_publisher = self.create_publisher(
            TargetPose, "/mitt/target_pose", 10
        )
        self._named_pose_service = self.create_service(
            MoveNamedPose,
            "/mitt/move_named_pose",
            self._on_move_named_pose,
            callback_group=callback_group,
        )
        self._mitt_pose_service = self.create_service(
            MoveMittPose,
            "/mitt/move_pose",
            self._on_move_mitt_pose,
            callback_group=callback_group,
        )
        self._stop_motion_service = self.create_service(
            Trigger,
            "/mitt/stop_motion",
            self._on_stop_motion,
            callback_group=callback_group,
        )
        self._start_reach_approach_service = self.create_service(
            Trigger,
            "/mitt/start_reach_approach",
            self._on_start_reach_approach,
            callback_group=callback_group,
        )
        self._prepare_person_pose_service = self.create_service(
            PreparePersonPose,
            "/mitt/prepare_person_pose",
            self._on_prepare_person_pose,
            callback_group=callback_group,
        )
        self.get_logger().info(
            "mitt positioner ready: SYSTEM_INITIAL, PERSON_READY, "
            "LEFT_HOOK_READY, RIGHT_HOOK_READY, UPPERCUT_READY, or PREPARE"
        )

    def _declare_parameters(self) -> None:
        parameters: dict[str, Any] = {
            "allow_real_motion": False,
            "nonstraight_pose_verified": False,
            "service_timeout_sec": 2.0,
            "initial_joint_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
            "reference_joint_deg": [
                -90.0, 60.0, 30.0, -90.0, -90.0, 0.0
            ],
            "joint_velocity_deg_s": 20.0,
            "joint_acceleration_deg_s2": 20.0,
            "reference_person_height_mm": 1730.0,
            "reference_person_arm_length_mm": 666.0,
            "arm_extension_ratio": 1.0,
            "hook_arm_extension_ratio": 0.70,
            "uppercut_arm_extension_ratio": 0.55,
            "shoulder_height_ratio": 0.818,
            "uppercut_height_ratio": 0.72,
            "hook_face_angle_deg": 55.0,
            "task_velocity_mm_s": 20.0,
            "task_angular_velocity_deg_s": 10.0,
            "task_acceleration_mm_s2": 20.0,
            "task_angular_acceleration_deg_s2": 20.0,
            "reach_jog_speed_percent": 1.0,
            "move_joint_service": "/dsr01/motion/move_joint",
            "move_line_service": "/dsr01/motion/move_line",
            "move_stop_service": "/dsr01/motion/move_stop",
            "jog_service": "/dsr01/motion/jog",
            "fkin_service": "/dsr01/motion/fkin",
            "get_current_posx_service": "/dsr01/aux_control/get_current_posx",
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)

    def _on_move_named_pose(
        self, request: MoveNamedPose.Request, response: MoveNamedPose.Response
    ) -> MoveNamedPose.Response:
        pose_name = request.pose_name.strip().upper()
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "another motion request is active"
            return response
        try:
            if pose_name == "SYSTEM_INITIAL":
                response.success, response.message = self._move_initial()
            elif pose_name in {
                "PERSON_READY",
                "STRAIGHT_READY",
                "LEFT_HOOK_READY",
                "RIGHT_HOOK_READY",
                "UPPERCUT_READY",
            }:
                punch_type = {
                    "PERSON_READY": PunchType.STRAIGHT,
                    "STRAIGHT_READY": PunchType.STRAIGHT,
                    "LEFT_HOOK_READY": PunchType.LEFT_HOOK,
                    "RIGHT_HOOK_READY": PunchType.RIGHT_HOOK,
                    "UPPERCUT_READY": PunchType.UPPERCUT,
                }[pose_name]
                response.success, response.message = self._move_person_ready(
                    punch_type
                )
            elif pose_name == "PREPARE":
                success, detail = self._move_initial()
                if success:
                    success, detail = self._move_person_ready(PunchType.STRAIGHT)
                response.success, response.message = success, detail
            else:
                response.success = False
                response.message = (
                    "unknown pose_name; use SYSTEM_INITIAL, PERSON_READY, "
                    "LEFT_HOOK_READY, RIGHT_HOOK_READY, UPPERCUT_READY, or PREPARE"
                )
        except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
            response.success = False
            response.message = str(error)
        finally:
            self._motion_lock.release()
        return response

    def _on_start_reach_approach(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "another motion request is active"
            return response
        keep_motion_lock = False
        try:
            if not self._real_motion_allowed():
                response.success = True
                response.message = "dry-run continuous Tool +Z approach"
                return response
            jog_request = Jog.Request()
            jog_request.jog_axis = self.JOG_AXIS_TASK_Z
            jog_request.move_reference = self.DR_TOOL
            jog_request.speed = self._positive_parameter("reach_jog_speed_percent")
            result = self._call_service(self._jog_client, jog_request)
            response.success = bool(result is not None and result.success)
            response.message = (
                "continuous Tool +Z approach started"
                if response.success
                else "Jog returned success=false"
            )
            if response.success:
                self._continuous_jog_active = True
                keep_motion_lock = True
        except (RuntimeError, TimeoutError, ValueError) as error:
            response.success = False
            response.message = str(error)
        finally:
            if not keep_motion_lock:
                self._motion_lock.release()
        return response

    def _on_stop_motion(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if not self._real_motion_allowed():
            response.success = True
            response.message = "dry-run motion soft-stop"
            return response
        stop_request = MoveStop.Request()
        stop_request.stop_mode = self.DR_SSTOP
        try:
            result = self._call_service(self._move_stop_client, stop_request)
            response.success = bool(result is not None and result.success)
            response.message = (
                "motion soft-stopped"
                if response.success
                else "MoveStop returned success=false"
            )
            if response.success and self._continuous_jog_active:
                self._continuous_jog_active = False
                self._motion_lock.release()
        except (RuntimeError, TimeoutError) as error:
            response.success = False
            response.message = str(error)
        return response

    def _on_move_mitt_pose(
        self, request: MoveMittPose.Request, response: MoveMittPose.Response
    ) -> MoveMittPose.Response:
        pose = (
            request.x_mm,
            request.y_mm,
            request.z_mm,
            request.rx_deg,
            request.ry_deg,
            request.rz_deg,
        )
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "another motion request is active"
            return response
        try:
            response.success, response.message = self._move_task_pose(
                pose,
                velocity=float(request.velocity),
                acceleration=float(request.acceleration),
                angular_velocity=(
                    float(request.angular_velocity)
                    if float(request.angular_velocity) != 0.0
                    else None
                ),
                angular_acceleration=(
                    float(request.angular_acceleration)
                    if float(request.angular_acceleration) != 0.0
                    else None
                ),
                source="DIRECT_SERVICE",
            )
        except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
            response.success = False
            response.message = str(error)
        finally:
            self._motion_lock.release()
        return response

    def _on_prepare_person_pose(
        self,
        request: PreparePersonPose.Request,
        response: PreparePersonPose.Response,
    ) -> PreparePersonPose.Response:
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "another motion request is active"
            return response
        try:
            punch_type = PunchType(str(request.punch_type).strip().upper())
            person = PersonMeasurement(
                height_mm=float(request.person_height_mm),
                arm_length_mm=float(request.arm_length_mm),
            )
            success, detail = self._move_mitt_reference()
            if not success:
                response.success = False
                response.message = detail
                return response
            # Preserve the controller's *actual* task orientation after the
            # reference MoveJ.  Recomputing the same pose through FKIN can return
            # an equivalent Euler representation and make the following MoveL
            # wind a wrist joint by ~360 degrees.
            reference_pose = self._reference_tcp_from_current_robot()
            plan = self._person_pose_planner(reference_pose).plan(
                person, punch_type
            )
            if not self._nonstraight_motion_verified(punch_type, plan.tcp_pose_mm_deg):
                response.success = False
                response.message = (
                    f"{punch_type.value} target planned but real motion is locked; "
                    "verify the dry-run target and set nonstraight_pose_verified=true"
                )
                return response
            if all(
                abs(actual - reference) <= 1e-6
                for actual, reference in zip(plan.tcp_pose_mm_deg, reference_pose)
            ):
                self._publish_target(plan.tcp_pose_mm_deg, "USER_PROFILE:REFERENCE")
                response.success = True
                response.message = "MITT_READY completed; reference profile"
            else:
                response.success, response.message = self._move_task_pose(
                    plan.tcp_pose_mm_deg,
                    velocity=self._positive_parameter("task_velocity_mm_s"),
                    acceleration=self._positive_parameter("task_acceleration_mm_s2"),
                    source=f"USER_PROFILE:{punch_type.value}",
                )
        except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
            response.success = False
            response.message = str(error)
        finally:
            self._motion_lock.release()
        return response

    def _move_initial(self) -> tuple[bool, str]:
        return self._move_joint_parameter("initial_joint_deg", "SYSTEM_INITIAL")

    def _move_joint_parameter(
        self, parameter_name: str, pose_name: str
    ) -> tuple[bool, str]:
        joints = _finite_values(
            self.get_parameter(parameter_name).value,
            f"{pose_name} joint pose",
        )
        if len(joints) != 6:
            raise ValueError(f"{pose_name} joint pose must contain six values")
        if not self._real_motion_allowed():
            return True, f"dry-run {pose_name}: {list(joints)} deg"

        request = MoveJoint.Request()
        request.pos = list(joints)
        request.vel = self._positive_parameter("joint_velocity_deg_s")
        request.acc = self._positive_parameter("joint_acceleration_deg_s2")
        request.time = 0.0
        request.radius = 0.0
        request.mode = self.DR_MV_MOD_ABS
        request.blend_type = 0
        request.sync_type = self.DR_SYNC_SYNC
        success = self._call_motion(self._move_joint_client, request)
        return success, f"{pose_name} completed" if success else "MoveJoint failed"

    def _move_mitt_reference(self) -> tuple[bool, str]:
        return self._move_joint_parameter("reference_joint_deg", "MITT_READY")

    def _move_person_ready(self, punch_type: PunchType) -> tuple[bool, str]:
        person = PersonMeasurement(
            self._positive_parameter("reference_person_height_mm"),
            self._positive_parameter("reference_person_arm_length_mm"),
        )
        success, detail = self._move_mitt_reference()
        if not success:
            return success, detail
        reference_pose = self._reference_tcp_from_current_robot()
        plan = self._person_pose_planner(reference_pose).plan(person, punch_type)
        if not self._nonstraight_motion_verified(punch_type, plan.tcp_pose_mm_deg):
            return False, (
                f"{punch_type.value} target planned but real motion is locked; "
                "verify the dry-run target and set nonstraight_pose_verified=true"
            )
        if punch_type is PunchType.STRAIGHT:
            self._publish_target(plan.tcp_pose_mm_deg, "REFERENCE_PROFILE")
            return True, detail
        return self._move_task_pose(
            plan.tcp_pose_mm_deg,
            velocity=self._positive_parameter("task_velocity_mm_s"),
            acceleration=self._positive_parameter("task_acceleration_mm_s2"),
            source=f"REFERENCE_PROFILE:{punch_type.value}",
        )

    def _reference_tcp_from_current_robot(self) -> tuple[float, ...]:
        """Return the controller's actual BASE TCP pose after MITT_READY MoveJ.

        The current controller orientation representation is intentionally kept
        for the following personalized MoveL.  This avoids commanding an
        equivalent-but-differently-wrapped Euler orientation that can cause an
        unnecessary wrist revolution.
        """
        request = GetCurrentPosx.Request()
        request.ref = self.DR_BASE
        try:
            response = self._call_service(self._current_posx_client, request)
            if response is None or not bool(response.success):
                raise RuntimeError("get_current_posx failed after MITT_READY")
            task_pos_info = list(response.task_pos_info)
            if not task_pos_info:
                raise RuntimeError("get_current_posx returned no task pose")
            pose = _finite_values(task_pos_info[0].data[:6], "current MITT_READY TCP pose")
            if len(pose) != 6:
                raise RuntimeError("get_current_posx returned an invalid TCP pose")
            return pose
        except Exception as error:
            # Keep compatibility with roboton variants where GetCurrentPosx is
            # temporarily unavailable; FKIN remains a fallback only.
            self.get_logger().warning(
                f"현재 TCP 자세 조회 실패 → FKIN fallback: {error}"
            )
            return self._reference_tcp_from_robot()

    def _reference_tcp_from_robot(self) -> tuple[float, ...]:
        joints = _finite_values(
            self.get_parameter("reference_joint_deg").value,
            "MITT_READY joint pose",
        )
        request = Fkin.Request()
        request.pos = list(joints)
        request.ref = self.DR_BASE
        response = self._call_service(self._fkin_client, request)
        if response is None or not bool(response.success):
            raise RuntimeError("forward kinematics failed for MITT_READY")
        pose = _finite_values(response.conv_posx, "MITT_READY TCP pose")
        if len(pose) != 6:
            raise RuntimeError("forward kinematics returned an invalid TCP pose")
        return pose

    def _person_pose_planner(
        self, reference_pose: Sequence[float]
    ) -> MittPosePlanner:
        return MittPosePlanner(
            arm_extension_ratio=self._positive_parameter("arm_extension_ratio"),
            hook_arm_extension_ratio=self._positive_parameter("hook_arm_extension_ratio"),
            uppercut_arm_extension_ratio=self._positive_parameter("uppercut_arm_extension_ratio"),
            shoulder_height_ratio=self._positive_parameter("shoulder_height_ratio"),
            uppercut_height_ratio=self._positive_parameter("uppercut_height_ratio"),
            hook_face_angle_deg=self._positive_parameter("hook_face_angle_deg"),
            reference_tcp_pose_mm_deg=reference_pose,
            reference_person_height_mm=self._positive_parameter(
                "reference_person_height_mm"
            ),
            reference_person_arm_length_mm=self._positive_parameter(
                "reference_person_arm_length_mm"
            ),
        )

    def _nonstraight_motion_verified(
        self, punch_type: PunchType, target_pose: Sequence[float]
    ) -> bool:
        if punch_type is PunchType.STRAIGHT or not self._real_motion_allowed():
            return True
        if bool(self.get_parameter("nonstraight_pose_verified").value):
            return True
        self._publish_target(target_pose, f"UNVERIFIED_PLAN:{punch_type.value}")
        self.get_logger().warning(
            f"{punch_type.value} target published without motion; "
            "nonstraight_pose_verified is false"
        )
        return False

    def _move_task_pose(
        self,
        pose: Sequence[float],
        *,
        velocity: float,
        acceleration: float,
        angular_velocity: float | None = None,
        angular_acceleration: float | None = None,
        source: str,
    ) -> tuple[bool, str]:
        values = _finite_values(pose, "TCP pose")
        if len(values) != 6:
            raise ValueError("TCP pose must contain six values")
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError("velocity must be finite and positive")
        if not math.isfinite(acceleration) or acceleration <= 0.0:
            raise ValueError("acceleration must be finite and positive")
        if angular_velocity is not None and (
            not math.isfinite(angular_velocity) or angular_velocity <= 0.0
        ):
            raise ValueError("angular velocity must be finite and positive")
        if angular_acceleration is not None and (
            not math.isfinite(angular_acceleration) or angular_acceleration <= 0.0
        ):
            raise ValueError("angular acceleration must be finite and positive")
        self._publish_target(values, source)
        if not self._real_motion_allowed():
            return True, f"dry-run target published: {list(values)}"

        request = MoveLine.Request()
        request.pos = list(values)
        request.vel = [
            velocity,
            angular_velocity
            if angular_velocity is not None
            else self._positive_parameter("task_angular_velocity_deg_s"),
        ]
        request.acc = [
            acceleration,
            angular_acceleration
            if angular_acceleration is not None
            else self._positive_parameter("task_angular_acceleration_deg_s2"),
        ]
        request.time = 0.0
        request.radius = 0.0
        request.ref = self.DR_BASE
        request.mode = self.DR_MV_MOD_ABS
        request.blend_type = 0
        request.sync_type = self.DR_SYNC_SYNC
        success = self._call_motion(self._move_line_client, request)
        return success, "TCP motion completed" if success else "MoveLine failed"

    def _publish_target(self, pose: Sequence[float], source: str) -> None:
        message = TargetPose()
        message.stamp = self.get_clock().now().to_msg()
        message.x_mm, message.y_mm, message.z_mm = pose[:3]
        message.rx_deg, message.ry_deg, message.rz_deg = pose[3:]
        message.source = source
        message.confidence = 1.0
        self._target_publisher.publish(message)

    def _call_motion(self, client: Any, request: Any) -> bool:
        response = self._call_service(client, request)
        return response is not None and bool(response.success)

    def _call_service(self, client: Any, request: Any) -> Any:
        timeout = self._positive_parameter("service_timeout_sec")
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"motion service unavailable: {client.srv_name}")
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        # Doosan의 sync_type=0 서비스는 명령 접수 시점이 아니라 실제 모션이
        # 끝난 뒤 응답한다. service_timeout_sec은 서비스 탐색에만 사용하고,
        # 모션 응답은 정상 완료 또는 노드 종료까지 기다린다. 여기서 2초 후
        # future를 취소하면 로봇은 계속 움직이는데 UI만 실패로 판단하게 된다.
        while rclpy.ok() and not event.wait(0.1):
            pass
        if not event.is_set():
            raise RuntimeError(f"motion interrupted: {client.srv_name}")
        try:
            response = future.result()
        except Exception as error:
            raise RuntimeError(
                f"motion service failed: {client.srv_name}: {error}"
            ) from error
        return response

    def _real_motion_allowed(self) -> bool:
        return bool(self.get_parameter("allow_real_motion").value)

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value


def _finite_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} values must be finite")
    return result


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MittPositionerNode()
    # One worker can be waiting for the synchronous MoveLine response while a
    # second worker issues MoveStop. Keep another worker free to receive the
    # controller service responses that complete both futures.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M0609 로봇 복싱 UI 연동 U자 위빙 노드

동작 흐름
1. UI 실행과 함께 이 노드가 시작되면 HOME
   `[-180, 0, 90, -180, -90, 0]`으로 이동한다.
2. 별도 사용자 입력 없이 위빙 준비 자세
   `[-180, 0, 90, -270, 90, 0]`으로 movej 이동한다.
3. 위빙 준비 자세 도착이 완료되면 해당 TCP를 U자의 오른쪽 상단 기준점으로 사용하고
   U자 위빙을 자동으로 반복 시작한다.
4. UI의 Wake Word가 `/wakeword_detected=True`를 발행하면 현재 위빙을 Soft Stop한다.
5. 위빙 준비 자세 `[-180, 0, 90, -270, 90, 0]`으로 복귀한 뒤 음성 명령을 기다린다.
6. STT 훈련 명령이 `/robot_boxing/action_command`로 들어오면 준비 자세를 확인한 뒤
   원래 명령을 `/robot_boxing/action_ready`로 전달한다.

U자 경로는 X=0에서 -X 방향으로만 전체 폭 170 mm를 사용하며,
여러 왕복을 한 번의 movesx()에 전달해 명령 경계의 완전 정지를 줄인다.

이 노드는 마이크와 openWakeWord를 직접 사용하지 않는다.
Wake Word와 STT는 KO UI가 담당한다.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Literal, Optional

import rclpy
from dsr_msgs2.srv import MoveStop
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

import DR_init


# ============================================================
# 로봇 설정
# ============================================================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# 노드 실행 직후 가장 먼저 이동하는 HOME 자세.
HOME_J_DEG = [-180.0, 0.0, 90.0, -180.0, -90.0, 0.0]

# HOME 도착 후 자동으로 이동하고, Wake Word 정지 후 복귀하는 위빙 준비 자세.
# HOME 대비 J4는 -180° → -270°, J5는 -90° → 90°로 함께 변경된다.
WEAVE_READY_J_DEG = [-180.0, 0.0, 90.0, -270.0, 90.0, 0.0]

MOVE_HOME_BEFORE_READY = False
MOVE_HOME_ON_STARTUP = True

# UI/run.sh에서 이 노드가 실행되면 HOME 이동 후 자동으로 위빙을 시작한다.
# False로 바꾸면 외부 start 명령을 받을 때까지 대기한다.
AUTO_START_WEAVING_ON_STARTUP = True

# 큰 U자 위빙 범위 [mm]
# 기존 half-width 값 85 mm를 유지하되, 위빙 준비 TCP를 U자의 오른쪽 상단으로 본다.
# 따라서 경로는 X=0에서 X=-170 mm까지만 사용하며 +X 영역으로는 이동하지 않는다.
U_HALF_WIDTH_MM = 85.0
U_DEPTH_MM = 68.0

# 조인트 속도 [deg/s], 가속도 [deg/s^2]
HOME_VEL_DEG_S = 15.0
HOME_ACC_DEG_S2 = 25.0
READY_VEL_DEG_S = 12.0
READY_ACC_DEG_S2 = 20.0
RETURN_VEL_DEG_S = 10.0
RETURN_ACC_DEG_S2 = 15.0

# Task-space 속도 [선속도 mm/s, 각속도 deg/s]
WEAVE_VEL = [450.0, 15.0]
WEAVE_ACC = [900.0, 30.0]

# 한 번의 movesx()에 포함할 부드러운 왕복 U 횟수와 점 개수.
# 4왕복 × 20점 = 80점으로, 상단마다 movesx()를 새로 호출하지 않는다.
SMOOTH_ROUND_TRIPS_PER_MOVESX = 4
SMOOTH_POINTS_PER_ROUND_TRIP = 20

# MoveStop: DR_SSTOP(2)
DR_SSTOP_MODE = 2


# ============================================================
# ROS 토픽
# ============================================================

WAKEWORD_TOPIC = "/wakeword_detected"

WEAVE_COMMAND_TOPIC = "/robot_boxing/weave_command"
ACTION_COMMAND_TOPIC = "/robot_boxing/action_command"
ACTION_READY_TOPIC = "/robot_boxing/action_ready"
STATE_TOPIC = "/robot_boxing/weave_state"


StopAction = Literal["ready", "home", "none"]


class WeaveCommandNode(Node):
    """UI에서 전달된 Wake Word와 훈련 전환 명령을 관리한다."""

    def __init__(self) -> None:
        super().__init__("robot_boxing_weave_command")

        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.reposition_event = threading.Event()
        self.shutdown_event = threading.Event()

        self._action_lock = threading.Lock()
        self._stop_action: StopAction = "ready"
        self._pending_robot_action: Optional[str] = None

        self._motion_lock = threading.Lock()
        self._motion_active = False

        self.state_pub = self.create_publisher(String, STATE_TOPIC, 10)
        self.action_ready_pub = self.create_publisher(
            String,
            ACTION_READY_TOPIC,
            10,
        )

        self.stop_client = self.create_client(
            MoveStop,
            f"/{ROBOT_ID}/motion/move_stop",
        )
        if not self.stop_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning(
                f"MoveStop 서비스가 아직 없음: /{ROBOT_ID}/motion/move_stop"
            )

        # KO UI ROS 브리지가 Wake Word 감지 시 True를 발행한다.
        # 수동 ros2 topic pub 테스트도 동일하게 사용할 수 있다.
        self.create_subscription(
            Bool,
            WAKEWORD_TOPIC,
            self._on_wakeword,
            10,
        )
        self.create_subscription(
            String,
            WEAVE_COMMAND_TOPIC,
            self._on_weave_command,
            10,
        )
        self.create_subscription(
            String,
            ACTION_COMMAND_TOPIC,
            self._on_action_command,
            10,
        )

        self.publish_state("INITIALIZING")
        self.get_logger().info(
            "통합 노드 준비 | "
            f"weave={WEAVE_COMMAND_TOPIC}, "
            f"action={ACTION_COMMAND_TOPIC}, "
            f"action_ready={ACTION_READY_TOPIC}"
        )

    # --------------------------------------------------------
    # 상태/동기화
    # --------------------------------------------------------

    def publish_state(self, state: str) -> None:
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.get_logger().info(f"[STATE] {state}")

    def set_motion_active(self, active: bool) -> None:
        with self._motion_lock:
            self._motion_active = active

    def is_motion_active(self) -> bool:
        with self._motion_lock:
            return self._motion_active

    def set_stop_action(self, action: StopAction) -> None:
        with self._action_lock:
            self._stop_action = action

    def consume_stop_action(self) -> StopAction:
        with self._action_lock:
            action = self._stop_action
            self._stop_action = "ready"
            return action

    def set_pending_robot_action(self, command: str) -> None:
        with self._action_lock:
            # 연속 명령이 들어오면 가장 최근 명령을 실행 대상으로 유지
            self._pending_robot_action = command

    def has_pending_robot_action(self) -> bool:
        with self._action_lock:
            return self._pending_robot_action is not None

    def consume_pending_robot_action(self) -> Optional[str]:
        with self._action_lock:
            command = self._pending_robot_action
            self._pending_robot_action = None
            return command

    # --------------------------------------------------------
    # 명령 처리
    # --------------------------------------------------------

    def request_start(self, source: str = "manual") -> None:
        if self.has_pending_robot_action():
            self.get_logger().info(
                "다른 동작 전환이 진행 중이므로 위빙 시작 요청을 무시합니다."
            )
            return

        if self.is_motion_active() or self.start_event.is_set():
            self.get_logger().info("이미 위빙 또는 자세 이동 중입니다.")
            return

        self.stop_event.clear()
        self.reposition_event.clear()
        self.set_stop_action("ready")
        self.start_event.set()
        self.get_logger().info(f"위빙 시작 요청 | source={source}")

    def request_stop(self, action: StopAction = "ready") -> None:
        already_stopping = self.stop_event.is_set()

        self.set_stop_action(action)
        self.stop_event.set()
        self.start_event.clear()

        if self.is_motion_active():
            # 이미 stop 요청을 전송했다면 복귀 모션까지 다시 끊지 않는다.
            if already_stopping:
                return

            if not self.stop_client.service_is_ready():
                self.get_logger().warning(
                    "MoveStop 서비스가 준비되지 않아 현재 movesx 사이클 "
                    "종료 후 정지합니다."
                )
                return

            self.get_logger().info("현재 로봇 모션 Soft Stop 요청")
            request = MoveStop.Request()
            request.stop_mode = DR_SSTOP_MODE

            future = self.stop_client.call_async(request)
            future.add_done_callback(self._on_move_stop_done)
        else:
            # 현재 위빙 중이 아니어도 READY/HOME으로 위치시킨다.
            self.reposition_event.set()

    def request_robot_action(self, command: str) -> None:
        command = command.strip()
        if not command:
            return

        self.get_logger().info(
            f"다른 로봇 동작 요청: {command} | "
            "위빙 정지 후 준비 자세로 복귀합니다."
        )
        self.set_pending_robot_action(command)
        self.publish_state("STOPPING_FOR_ACTION")
        self.request_stop("ready")

    def notify_ready_for_robot_action(self) -> None:
        command = self.consume_pending_robot_action()
        if command is None:
            return

        msg = String()
        msg.data = command
        self.action_ready_pub.publish(msg)
        self.publish_state("ACTION_READY")
        self.get_logger().info(
            f"준비 자세 복귀 완료 → 동작 명령 전달: "
            f"{ACTION_READY_TOPIC} = {command}"
        )

    def _on_move_stop_done(self, future) -> None:
        try:
            response = future.result()
            if response is not None and response.success:
                self.get_logger().info("MoveStop Soft Stop 성공")
            else:
                self.get_logger().warning("MoveStop 응답 success=False")
        except Exception as exc:
            self.get_logger().warning(
                f"MoveStop 호출 실패: {type(exc).__name__}: {exc}"
            )

    def _on_wakeword(self, msg: Bool) -> None:
        if not msg.data:
            return

        # 이 버전에서 Wake Word는 위빙 시작 신호가 아니라
        # 사용자가 음성 명령을 내리기 위한 정지 신호다.
        self.get_logger().info(
            "실제 Wake Word 감지 이벤트 수신 → "
            "위빙 정지 및 위빙 준비 자세 복귀"
        )
        self.publish_state("STOPPING_FOR_VOICE")
        self.request_stop("ready")

    def _on_action_command(self, msg: String) -> None:
        self.request_robot_action(msg.data)

    def _on_weave_command(self, msg: String) -> None:
        command = msg.data.strip()
        normalized = command.lower()
        self.get_logger().info(f"위빙 명령 수신: {command}")

        if normalized in {
            "start",
            "weave",
            "wakeup",
            "운동 준비",
            "위빙 시작",
        }:
            self.request_start(source="topic")
        elif normalized in {
            "stop",
            "ready",
            "위빙 정지",
        }:
            self.request_stop("ready")
        elif normalized in {
            "home",
            "홈",
            "종료",
        }:
            self.request_stop("home")
        else:
            # 운동 시작, 스트레이트, 캘리브레이션 등 다른 명령은
            # 위빙 정지 → 준비 자세 복귀 → action_ready로 넘긴다.
            self.request_robot_action(command)



class WeaveMotionWorker:
    """DSR_ROBOT2 모션을 별도 스레드에서 수행한다."""

    def __init__(
        self,
        command_node: WeaveCommandNode,
        motion_node: Node,
        dsr_module,
    ) -> None:
        self.command_node = command_node
        self.motion_node = motion_node
        self.dsr = dsr_module
        self.thread: Optional[threading.Thread] = None

        self.home_j = self.dsr.posj(*HOME_J_DEG)
        self.weave_ready_j = self.dsr.posj(*WEAVE_READY_J_DEG)

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name="weave_motion_worker",
            daemon=True,
        )
        self.thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _build_relative_path(self, absolute_targets_xz):
        """현재 TCP 기준 XZ 절대 목표열을 movesx 상대 증분 경로로 변환한다."""
        relative_steps_xz = []
        prev_x = 0.0
        prev_z = 0.0
        for target_x, target_z in absolute_targets_xz:
            relative_steps_xz.append(
                (float(target_x) - prev_x, float(target_z) - prev_z)
            )
            prev_x = float(target_x)
            prev_z = float(target_z)

        return [
            self.dsr.posx(dx, 0.0, dz, 0.0, 0.0, 0.0)
            for dx, dz in relative_steps_xz
        ]

    def _build_smooth_weave_path(self):
        """
        여러 번의 왕복 U를 하나의 movesx() 경로로 생성한다.

        기존에는 오른쪽→왼쪽과 왼쪽→오른쪽을 각각 별도 movesx()로
        호출하여 매 상단에서 명령이 끝나고 완전히 감속했다.
        이 함수는 여러 왕복을 하나의 스플라인 경로로 묶어 상단의
        불필요한 정지와 재출발을 줄인다.

        좌표 범위는 기존과 동일하다.
        - X: 준비 위치 기준 0 ~ -170 mm
        - Z: 준비 위치 기준 0 ~ -68 mm
        """
        width = 2.0 * float(U_HALF_WIDTH_MM)
        depth = float(U_DEPTH_MM)
        round_trips = max(1, int(SMOOTH_ROUND_TRIPS_PER_MOVESX))
        points_per_round_trip = max(12, int(SMOOTH_POINTS_PER_ROUND_TRIP))

        absolute_targets_xz = []

        for _ in range(round_trips):
            # theta=0은 현재 시작점이므로 제외하고 다음 점부터 생성한다.
            # theta: 0→π  : 오른쪽 상단 → 왼쪽 상단
            # theta: π→2π: 왼쪽 상단 → 오른쪽 상단
            for index in range(1, points_per_round_trip + 1):
                theta = (
                    2.0
                    * math.pi
                    * index
                    / points_per_round_trip
                )

                # X는 0 → -width → 0으로 부드럽게 왕복한다.
                x = -0.5 * width * (1.0 - math.cos(theta))

                # 각 편도 구간의 중앙에서 U 바닥에 도달한다.
                # 상단에서는 기울기가 0이어서 급격한 꺾임을 줄인다.
                z = -depth * (math.sin(theta) ** 2)

                absolute_targets_xz.append((x, z))

        smooth_path = self._build_relative_path(absolute_targets_xz)

        self.command_node.get_logger().info(
            f"부드러운 연속 U자: X=0.0~-{width:.1f}mm, "
            f"깊이={depth:.1f}mm, "
            f"왕복/호출={round_trips}, "
            f"점/왕복={points_per_round_trip}, "
            f"총점={len(smooth_path)}, "
            f"vel={WEAVE_VEL}, acc={WEAVE_ACC}"
        )
        return smooth_path

    def _move_home_on_startup(self) -> bool:
        """노드 시작 직후 HOME으로 이동하고 성공 여부를 반환한다."""
        if not MOVE_HOME_ON_STARTUP:
            self.command_node.publish_state("IDLE_HOME")
            return True

        self.command_node.set_motion_active(True)
        try:
            self.command_node.publish_state("MOVING_HOME")
            self.command_node.get_logger().info(
                f"초기 HOME movej 전송: {HOME_J_DEG}"
            )
            ret = self.dsr.movej(
                self.home_j,
                vel=HOME_VEL_DEG_S,
                acc=HOME_ACC_DEG_S2,
            )
            self.command_node.get_logger().info(
                f"초기 HOME movej return={ret}"
            )
            if ret == -1:
                raise RuntimeError("초기 HOME movej 실패(return=-1)")
            self.command_node.publish_state("IDLE_HOME")
            return True
        except Exception as exc:
            self.command_node.get_logger().error(
                f"초기 HOME 복귀 오류: {type(exc).__name__}: {exc}"
            )
            self.command_node.publish_state("ERROR")
            return False
        finally:
            self.command_node.set_motion_active(False)

    def _move_to_ready(self) -> bool:
        if MOVE_HOME_BEFORE_READY:
            self.command_node.publish_state("MOVING_HOME")
            self.command_node.get_logger().info(
                f"HOME movej 전송: {HOME_J_DEG}"
            )
            ret = self.dsr.movej(
                self.home_j,
                vel=HOME_VEL_DEG_S,
                acc=HOME_ACC_DEG_S2,
            )
            self.command_node.get_logger().info(
                f"HOME movej return={ret}"
            )
            if ret == -1:
                raise RuntimeError("HOME movej 실패(return=-1)")

            if self.command_node.stop_event.is_set():
                return False

        # 별도 접근 자세 없이 HOME에서 위빙 준비 자세로 바로 이동한다.
        # movej는 동기식이므로 목표 자세 도착 완료 후 다음 위빙 명령으로 진행한다.
        self.command_node.publish_state("MOVING_WEAVE_READY")
        self.command_node.get_logger().info(
            "HOME → 위빙 준비 자세 직접 movej | "
            f"target={WEAVE_READY_J_DEG}"
        )
        ret = self.dsr.movej(
            self.weave_ready_j,
            vel=READY_VEL_DEG_S,
            acc=READY_ACC_DEG_S2,
        )
        self.command_node.get_logger().info(
            f"WEAVE_READY movej return={ret}"
        )
        if ret == -1:
            raise RuntimeError(
                "위빙 준비 자세 movej 실패(return=-1)"
            )
        return not self.command_node.stop_event.is_set()

    def _return_after_stop(self, action: StopAction) -> None:
        if action == "none" or self.command_node.shutdown_event.is_set():
            return

        time.sleep(0.3)

        if action == "home":
            self.command_node.publish_state("RETURNING_HOME")
            ret = self.dsr.movej(
                self.home_j,
                vel=RETURN_VEL_DEG_S,
                acc=RETURN_ACC_DEG_S2,
            )
            self.command_node.get_logger().info(
                f"RETURN HOME movej return={ret}"
            )
            if ret == -1:
                raise RuntimeError(
                    "정지 후 HOME 복귀 실패(return=-1)"
                )
            self.command_node.publish_state("IDLE_HOME")
            return

        self.command_node.publish_state("RETURNING_WEAVE_READY")
        ret = self.dsr.movej(
            self.weave_ready_j,
            vel=RETURN_VEL_DEG_S,
            acc=RETURN_ACC_DEG_S2,
        )
        self.command_node.get_logger().info(
            f"RETURN READY movej return={ret}"
        )
        if ret == -1:
            raise RuntimeError(
                "정지 후 위빙 준비 자세 복귀 실패(return=-1)"
            )

        self.command_node.publish_state("READY")
        self.command_node.notify_ready_for_robot_action()

    def _run_idle_reposition(self, action: StopAction) -> None:
        if action == "none" or self.command_node.shutdown_event.is_set():
            return

        self.command_node.set_motion_active(True)
        try:
            if action == "home":
                self.command_node.publish_state("MOVING_HOME")
                ret = self.dsr.movej(
                    self.home_j,
                    vel=RETURN_VEL_DEG_S,
                    acc=RETURN_ACC_DEG_S2,
                )
                self.command_node.get_logger().info(
                    f"IDLE HOME movej return={ret}"
                )
                if ret == -1:
                    raise RuntimeError(
                        "IDLE HOME movej 실패(return=-1)"
                    )
                self.command_node.publish_state("IDLE_HOME")
            else:
                self.command_node.publish_state(
                    "MOVING_WEAVE_READY"
                )
                ret = self.dsr.movej(
                    self.weave_ready_j,
                    vel=RETURN_VEL_DEG_S,
                    acc=RETURN_ACC_DEG_S2,
                )
                self.command_node.get_logger().info(
                    f"IDLE READY movej return={ret}"
                )
                if ret == -1:
                    raise RuntimeError(
                        "IDLE READY movej 실패(return=-1)"
                    )
                self.command_node.publish_state("READY")
                self.command_node.notify_ready_for_robot_action()
        except Exception as exc:
            self.command_node.get_logger().error(
                f"IDLE 재배치 오류: "
                f"{type(exc).__name__}: {exc}"
            )
            self.command_node.publish_state("ERROR")
        finally:
            self.command_node.set_motion_active(False)
            self.command_node.stop_event.clear()

    def _run_one_session(self) -> None:
        self.command_node.stop_event.clear()
        self.command_node.set_motion_active(True)

        try:
            self.command_node.get_logger().info(
                "모션 시작: TP AUTO/AUTONOMOUS + Servo ON 확인"
            )

            if not self._move_to_ready():
                return

            smooth_weave_path = self._build_smooth_weave_path()

            # 준비 자세가 U자의 오른쪽 상단이므로 별도의 +X 진입 동작 없이 즉시 시작한다.
            self.command_node.publish_state("WEAVING")

            batch = 0
            while (
                rclpy.ok()
                and not self.command_node.shutdown_event.is_set()
                and not self.command_node.stop_event.is_set()
            ):
                batch += 1
                self.command_node.get_logger().info(
                    "부드러운 연속 U자 위빙 "
                    f"batch={batch}, "
                    f"round_trips={SMOOTH_ROUND_TRIPS_PER_MOVESX}"
                )

                ret = self.dsr.movesx(
                    smooth_weave_path,
                    vel=WEAVE_VEL,
                    acc=WEAVE_ACC,
                    ref=self.dsr.DR_BASE,
                    mod=self.dsr.DR_MV_MOD_REL,
                )
                self.command_node.get_logger().info(
                    f"smooth movesx return={ret}"
                )
                if ret == -1:
                    raise RuntimeError(
                        "smooth movesx 실패(return=-1)"
                    )

        except Exception as exc:
            if self.command_node.stop_event.is_set():
                self.command_node.get_logger().info(
                    f"위빙 모션 정지 처리: "
                    f"{type(exc).__name__}"
                )
            else:
                self.command_node.get_logger().error(
                    f"위빙 실행 오류: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.command_node.publish_state("ERROR")
                self.command_node.set_stop_action("none")
        finally:
            action = self.command_node.consume_stop_action()

            try:
                # motion_active를 유지한 채 복귀까지 완료하여
                # 새 위빙이 중간에 겹치지 않게 한다.
                self._return_after_stop(action)
            except Exception as exc:
                self.command_node.get_logger().error(
                    f"정지 후 복귀 오류: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.command_node.publish_state("ERROR")
            finally:
                self.command_node.set_motion_active(False)
                self.command_node.stop_event.clear()

    def _run(self) -> None:
        self.command_node.get_logger().info("모션 작업 스레드 시작")
        startup_ok = self._move_home_on_startup()

        if (
            startup_ok
            and AUTO_START_WEAVING_ON_STARTUP
            and not self.command_node.shutdown_event.is_set()
        ):
            self.command_node.get_logger().info(
                "초기 HOME 이동 완료 → 자동 위빙 시작"
            )
            self.command_node.request_start(source="startup")

        while (
            rclpy.ok()
            and not self.command_node.shutdown_event.is_set()
        ):
            if self.command_node.start_event.is_set():
                self.command_node.start_event.clear()
                if self.command_node.shutdown_event.is_set():
                    break
                self._run_one_session()
                continue

            if self.command_node.reposition_event.is_set():
                self.command_node.reposition_event.clear()
                if self.command_node.shutdown_event.is_set():
                    break
                action = self.command_node.consume_stop_action()
                self._run_idle_reposition(action)
                continue

            time.sleep(0.05)

        self.command_node.get_logger().info("모션 작업 스레드 종료")



def main(args=None) -> None:
    rclpy.init(args=args)

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    motion_node = rclpy.create_node(
        "robot_boxing_weave_motion",
        namespace=ROBOT_ID,
    )
    DR_init.__dsr__node = motion_node

    # DR_init 설정 이후 import해야 함
    import DSR_ROBOT2 as dsr

    command_node = WeaveCommandNode()
    motion_worker = WeaveMotionWorker(command_node, motion_node, dsr)
    motion_worker.start()

    command_executor = SingleThreadedExecutor()
    command_executor.add_node(command_node)

    command_node.get_logger().info(
        "UI 연동 위빙 노드 시작 | 시작 시 자동 위빙 | "
        "Wake Word 수신 시 위빙 정지 후 준비 자세 복귀 | "
        "로컬 마이크 미사용"
    )

    try:
        command_executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        if rclpy.ok():
            command_node.get_logger().info("종료 요청")
    finally:
        command_node.shutdown_event.set()
        command_node.set_stop_action("none")
        command_node.request_stop("none")
        command_node.start_event.set()
        command_node.reposition_event.set()
        motion_worker.join(timeout=3.0)

        command_executor.remove_node(command_node)
        command_executor.shutdown()
        command_node.destroy_node()
        motion_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])

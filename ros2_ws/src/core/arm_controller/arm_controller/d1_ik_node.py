#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from go2_msgs.msg import ArmAngles
from unitree_arm.msg import ArmString

from arm_controller.d1_ik_solver import D1IKSolver


class D1IKNode(Node):
    COMMAND_THRESHOLD_DEG = 0.1
    COMMAND_COOLDOWN_S = 0
    SINGLE_JOINT_INDICES = (1, 2, 3, 4, 5)
    COMMAND_SEQ = 4
    COMMAND_ADDRESS = 1
    COMMAND_FUNCODE_SINGLE_SERVO_ANGLE = 1
    COMMAND_FUNCODE_SINGLE_MODE = 4
    COMMAND_DELAY_MS = 0
    COMMAND_MODE_SERVO_ID = 0
    COMMAND_MODE_VALUE = 0
    STARTUP_MODE_DELAY_S = 0.1
    JOINT0_MODE_DELAY_S = 1.0

    def __init__(self) -> None:
        super().__init__("d1_ik_node")

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.solver = D1IKSolver()
        self.q_out: np.ndarray | None = None
        self.x_vel = 0.0
        self._nominal_seeded = False
        self._last_command_time_by_joint: dict[int, float] = {}
        self._pending_single_mode_timers: list = []

        self.pub_arm_command = self.create_publisher(
            ArmString,
            "/arm_Command",
            qos,
        )
        self.pub_arm_ik_debug = self.create_publisher(
            String,
            "/arm_ik_debug",
            qos,
        )
        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            "/arm_angles",
            self.on_arm_angles,
            qos,
        )
        self._startup_mode_timer = self.create_timer(
            self.STARTUP_MODE_DELAY_S,
            self.publish_startup_single_mode_command,
        )

        self.get_logger().info("d1_ik_node listening on /arm_angles and publishing /arm_ik_debug")

    def on_arm_angles(self, msg: ArmAngles) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        # self.handle_joint0_limits(current_q)
        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]
        if not self._nominal_seeded:
            self.solver.set_nominal_q(q_in)
            self._nominal_seeded = True
            self.get_logger().info(
                "Seeded IK nominal q0 from current /arm_angles: "
                + np.array2string(q_in, precision=2, separator=", ")
            )

        try:
            q_out_solver = self.solver.solve_with_fixed_joint1(q_in)
            q_out = q_out_solver.copy()
            q_out[0] = -q_out[0]
            q_out[3] = -q_out[3]
            self.q_out = q_out
            self.publish_arm_commands(q_in, q_out_solver, q_out)
        except RuntimeError as exc:
            self.get_logger().warning(f"IK solve failed: {exc}")

    def handle_joint0_limits(self, current_q: np.ndarray) -> None:
        joint0_angle = float(current_q[0])

        if -80.0 < joint0_angle < 0.0:
            self.publish_single_joint_command(0, -82.0)
        elif -180.0 < joint0_angle < -100.0:
            self.publish_single_joint_command(0, -98.0)
        else:
            self.x_vel = 0.0

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
    ) -> None:
        now = time.monotonic()
        joints_to_update: list[int] = []

        for joint_index in self.SINGLE_JOINT_INDICES:
            last_command_time = self._last_command_time_by_joint.get(joint_index)
            if (
                last_command_time is not None
                and (now - last_command_time) < self.COMMAND_COOLDOWN_S
            ):
                continue
            if abs(float(q_in[joint_index]) - float(q_out_solver[joint_index])) <= self.COMMAND_THRESHOLD_DEG:
                continue
            joints_to_update.append(joint_index)

        if not joints_to_update:
            return

        self.publish_arm_ik_debug(q_out)

        joints_to_publish = joints_to_update
        if not joints_to_publish:
            return

        for joint_index in joints_to_publish:
            self.publish_single_joint_command(joint_index, float(q_out[joint_index]))

        for joint_index in joints_to_publish:
            self._last_command_time_by_joint[joint_index] = now

    def publish_single_joint_command(self, servo_id: int, angle_deg: float) -> None:
        payload = {
            "seq": self.COMMAND_SEQ,
            "address": self.COMMAND_ADDRESS,
            "funcode": self.COMMAND_FUNCODE_SINGLE_SERVO_ANGLE,
            "data": {
                "id": servo_id,
                "angle": float(angle_deg),
                "delay_ms": self.COMMAND_DELAY_MS,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

        if servo_id == 0:
            self.schedule_single_mode_command(self.JOINT0_MODE_DELAY_S)

    def schedule_single_mode_command(self, delay_s: float) -> None:
        timer_holder = {"timer": None}

        def _publish_when_ready() -> None:
            timer = timer_holder["timer"]
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
                if timer in self._pending_single_mode_timers:
                    self._pending_single_mode_timers.remove(timer)
            self.publish_single_mode_command()

        timer = self.create_timer(delay_s, _publish_when_ready)
        timer_holder["timer"] = timer
        self._pending_single_mode_timers.append(timer)

    def publish_single_mode_command(self) -> None:
        payload = {
            "seq": self.COMMAND_SEQ,
            "address": self.COMMAND_ADDRESS,
            "funcode": self.COMMAND_FUNCODE_SINGLE_MODE,
            "data": {
                "id": self.COMMAND_MODE_SERVO_ID,
                "mode": self.COMMAND_MODE_VALUE,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_startup_single_mode_command(self) -> None:
        if self._startup_mode_timer is not None:
            self._startup_mode_timer.cancel()
            self.destroy_timer(self._startup_mode_timer)
            self._startup_mode_timer = None

        self.publish_single_mode_command()
        self.get_logger().info(
            "Published startup single mode command: "
            + json.dumps(
                {
                    "seq": self.COMMAND_SEQ,
                    "address": self.COMMAND_ADDRESS,
                    "funcode": self.COMMAND_FUNCODE_SINGLE_MODE,
                    "data": {
                        "id": self.COMMAND_MODE_SERVO_ID,
                        "mode": self.COMMAND_MODE_VALUE,
                    },
                },
                separators=(",", ":"),
            )
        )

    def publish_arm_ik_debug(self, q_out: np.ndarray) -> None:
        msg = String()
        nominal_q_str = np.array2string(self.solver.nominal_q, precision=2, separator=", ")
        q_out_str = np.array2string(q_out, precision=1, separator=", ")
        msg.data = f"q0={nominal_q_str}\nq_out={q_out_str}"
        self.pub_arm_ik_debug.publish(msg)


def main() -> None:
    rclpy.init()
    node = D1IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

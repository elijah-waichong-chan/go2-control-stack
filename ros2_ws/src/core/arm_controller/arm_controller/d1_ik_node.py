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
    COMMAND_THRESHOLD_DEG = 1.0
    COMMAND_COOLDOWN_S = 0.5

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
        self._nominal_seeded = False
        self._last_command_time_by_joint: dict[int, float] = {}

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

        self.get_logger().info("d1_ik_node listening on /arm_angles and publishing /arm_ik_debug")

    def on_arm_angles(self, msg: ArmAngles) -> None:
        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
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
            gripper_angle = float(msg.angle_deg[6]) if len(msg.angle_deg) > 6 else 0.0
            self.publish_arm_commands(q_in, q_out_solver, q_out, gripper_angle)
        except RuntimeError as exc:
            self.get_logger().warning(f"IK solve failed: {exc}")

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
        gripper_angle: float,
    ) -> None:
        now = time.monotonic()
        joints_to_update: list[int] = []

        for joint_id in range(len(q_out_solver)):
            last_command_time = self._last_command_time_by_joint.get(joint_id)
            if (
                last_command_time is not None
                and (now - last_command_time) < self.COMMAND_COOLDOWN_S
            ):
                continue
            if abs(float(q_in[joint_id]) - float(q_out_solver[joint_id])) <= self.COMMAND_THRESHOLD_DEG:
                continue
            joints_to_update.append(joint_id)

        if not joints_to_update:
            return
        
        self.publish_arm_ik_debug(q_out)
        
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 2,
            "data": {
                "mode": 1,
                "angle0": float(q_out[0]),
                "angle1": float(q_out[1]),
                "angle2": float(q_out[2]),
                "angle3": float(q_out[3]),
                "angle4": float(q_out[4]),
                "angle5": float(q_out[5]),
                "angle6": float(gripper_angle),
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        # self.pub_arm_command.publish(msg)

        mode_payload = {
            "seq": 4,
            "address": 1,
            "funcode": 4,
            "data": {
                "id": 0,
                "mode": 0,
            },
        }
        mode_msg = ArmString()
        mode_msg.data = json.dumps(mode_payload, separators=(",", ":"))
        # self.pub_arm_command.publish(mode_msg)

        for joint_id in joints_to_update:
            self._last_command_time_by_joint[joint_id] = now

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

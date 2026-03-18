#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from go2_msgs.msg import ArmAngles

from arm_controller.d1_ik_solver import D1IKSolver


class D1IKNode(Node):
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

        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            "/arm_angles",
            self.on_arm_angles,
            qos,
        )

        self.get_logger().info("d1_ik_node listening on /arm_angles")

    def on_arm_angles(self, msg: ArmAngles) -> None:
        q_in = np.asarray(msg.angle_deg[:6], dtype=float)

        try:
            self.q_out = self.solver.solve_with_fixed_joint1(q_in)
        except RuntimeError as exc:
            self.get_logger().warning(f"IK solve failed: {exc}")


def main() -> None:
    rclpy.init()
    node = D1IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

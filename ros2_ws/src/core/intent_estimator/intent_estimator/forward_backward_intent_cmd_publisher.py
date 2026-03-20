#!/usr/bin/env python3
"""Publish locomotion commands from forward/backward intent labels."""

from __future__ import annotations

import time

import rclpy
from go2_msgs.msg import LocomotionCmd
from rclpy.node import Node

import numpy as np
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from go2_msgs.msg import ArmAngles
from unitree_arm.msg import ArmString


class ForwardBackwardIntentCmdPublisher(Node):
    """Drive `/locomotion_cmd`"""

    def __init__(self) -> None:
        super().__init__("forward_backward_intent_cmd_publisher")

        self.publish_hz = float(50.0)
        self.x_vel = float(0.0)
        self.y_vel = float(0.0)
        self.yaw_rate = float(0.0)
        self.z_pos = float(0.27)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            "/arm_angles",
            self.on_arm_angles,
            qos,
        )
        self.pub_cmd = self.create_publisher(
            LocomotionCmd, 
            "/locomotion_cmd", 
            qos
        )
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

    def on_arm_angles(self, msg: ArmAngles) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        if current_q[0] > -80.0 and current_q[0] < 0.0:
            self.x_vel = 0.3
        elif current_q[0] < -100.0 and current_q[0] > -180.0:
            self.x_vel = -0.5
        else:
            self.x_vel = 0.0


    def on_timer(self) -> None:
        msg = LocomotionCmd()
        msg.stamp = self.get_clock().now().to_msg()
        msg.x_vel = self.x_vel
        msg.y_vel = self.y_vel
        msg.z_pos = self.z_pos
        msg.yaw_rate = self.yaw_rate
        self.pub_cmd.publish(msg)


def main() -> None:
    rclpy.init()
    node = ForwardBackwardIntentCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

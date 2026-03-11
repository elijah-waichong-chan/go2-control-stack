#!/usr/bin/env python3
"""Publish locomotion commands from forward/backward intent labels."""

from __future__ import annotations

import time

import rclpy
from go2_msgs.msg import LocomotionCmd
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32

from direction_intent_estimator.intent_cmd_override import TimedIntentXVelocity


class ForwardBackwardIntentCmdPublisher(Node):
    """Drive `/locomotion_cmd` directly from forward/backward intent labels."""

    def __init__(self) -> None:
        super().__init__("forward_backward_intent_cmd_publisher")

        self.declare_parameter(
            "intent_topic", "/direction_intent/forward_backward"
        )
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("intent_hold_s", 1.0)
        self.declare_parameter("backward_x_vel", -0.5)
        self.declare_parameter("forward_x_vel", 0.5)
        self.declare_parameter("y_vel", 0.0)
        self.declare_parameter("yaw_rate", 0.0)
        self.declare_parameter("z_pos", 0.27)
        self.declare_parameter("gait_hz", 3.0)

        self.intent_topic = str(self.get_parameter("intent_topic").value)
        self.locomotion_cmd_topic = str(
            self.get_parameter("locomotion_cmd_topic").value
        )
        self.publish_hz = max(
            1.0, float(self.get_parameter("publish_hz").value)
        )
        self.y_vel = float(self.get_parameter("y_vel").value)
        self.yaw_rate = float(self.get_parameter("yaw_rate").value)
        self.z_pos = float(self.get_parameter("z_pos").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)
        self.x_override = TimedIntentXVelocity(
            hold_s=max(0.0, float(self.get_parameter("intent_hold_s").value)),
            backward_x_vel=float(self.get_parameter("backward_x_vel").value),
            forward_x_vel=float(self.get_parameter("forward_x_vel").value),
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_intent = self.create_subscription(
            Int32, self.intent_topic, self.on_intent, qos
        )
        self.pub_cmd = self.create_publisher(
            LocomotionCmd, self.locomotion_cmd_topic, qos
        )
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

        self.get_logger().info(
            "forward_backward_intent_cmd_publisher ready: "
            f"{self.intent_topic} -> {self.locomotion_cmd_topic}, "
            f"hold={self.x_override.hold_s:.2f}s, "
            f"backward={self.x_override.backward_x_vel:.2f}m/s, "
            f"forward={self.x_override.forward_x_vel:.2f}m/s"
        )

    def on_intent(self, msg: Int32) -> None:
        self.x_override.on_intent(int(msg.data), time.monotonic())

    def on_timer(self) -> None:
        msg = LocomotionCmd()
        msg.stamp = self.get_clock().now().to_msg()
        msg.x_vel = self.x_override.current_x_vel(time.monotonic())
        msg.y_vel = self.y_vel
        msg.z_pos = self.z_pos
        msg.yaw_rate = self.yaw_rate
        msg.gait_hz = self.gait_hz
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

#!/usr/bin/env python3
"""Publish locomotion commands from intent-estimator topics."""

from __future__ import annotations

from collections import deque

import rclpy
from hq_pcot_msgs.msg import LocomotionCmd
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32


class IntentCommandCoordinator(Node):
    """Drive `/locomotion_cmd` from published intent signals."""

    def __init__(self) -> None:
        super().__init__("intent_command_coordinator")

        self.declare_parameter("forward_backward_intent_topic", "/direction_intent/front_back")
        self.declare_parameter("left_right_intent_topic", "/direction_intent/left_right")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("forward_x_vel", 0.6)
        self.declare_parameter("backward_x_vel", -0.7)
        self.declare_parameter("idle_intent_label", 0)
        self.declare_parameter("forward_intent_label", 1)
        self.declare_parameter("backward_intent_label", 2)
        self.declare_parameter("forward_backward_intent_timeout_s", 0.5)
        self.declare_parameter("y_vel", 0.0)
        self.declare_parameter("z_pos", 0.27)
        self.declare_parameter("left_intent_label", 3)
        self.declare_parameter("right_intent_label", 4)
        self.declare_parameter("left_window_size", 5)
        self.declare_parameter("right_window_size", 5)
        self.declare_parameter("zero_window_size", 5)
        self.declare_parameter("zero_hold_s", 0.2)
        self.declare_parameter("left_y_vel", 0.5)
        self.declare_parameter("right_y_vel", -0.5)
        self.declare_parameter("left_right_intent_timeout_s", 0.5)

        self.forward_backward_intent_topic = str(
            self.get_parameter("forward_backward_intent_topic").value
        )
        self.left_right_intent_topic = str(
            self.get_parameter("left_right_intent_topic").value
        )
        self.locomotion_cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.forward_x_vel = float(self.get_parameter("forward_x_vel").value)
        self.backward_x_vel = float(self.get_parameter("backward_x_vel").value)
        self.idle_intent_label = int(self.get_parameter("idle_intent_label").value)
        self.forward_intent_label = int(self.get_parameter("forward_intent_label").value)
        self.backward_intent_label = int(self.get_parameter("backward_intent_label").value)
        self.forward_backward_intent_timeout_s = max(
            0.0, float(self.get_parameter("forward_backward_intent_timeout_s").value)
        )
        self.base_y_vel = float(self.get_parameter("y_vel").value)
        self.z_pos = float(self.get_parameter("z_pos").value)
        self.left_intent_label = int(self.get_parameter("left_intent_label").value)
        self.right_intent_label = int(self.get_parameter("right_intent_label").value)
        self.left_window_size = max(1, int(self.get_parameter("left_window_size").value))
        self.right_window_size = max(1, int(self.get_parameter("right_window_size").value))
        self.zero_window_size = max(1, int(self.get_parameter("zero_window_size").value))
        self.zero_hold_s = max(0.0, float(self.get_parameter("zero_hold_s").value))
        self.left_right_window_size = max(
            self.left_window_size, self.right_window_size, self.zero_window_size
        )
        self.left_y_vel = float(self.get_parameter("left_y_vel").value)
        self.right_y_vel = float(self.get_parameter("right_y_vel").value)
        self.left_right_intent_timeout_s = max(
            0.0, float(self.get_parameter("left_right_intent_timeout_s").value)
        )

        self.latest_forward_backward_intent: int | None = None
        self.last_forward_backward_intent_time_ns: int | None = None
        self.latest_left_right_intent: int | None = None
        self.last_left_right_intent_time_ns: int | None = None
        self.left_right_recent: deque[int] = deque(maxlen=self.left_right_window_size)
        self.active_y_vel = self.base_y_vel
        self.zero_hold_until_ns: int | None = None

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_forward_backward_intent = self.create_subscription(
            Int32,
            self.forward_backward_intent_topic,
            self.on_forward_backward_intent,
            qos,
        )
        self.sub_left_right_intent = self.create_subscription(
            Int32,
            self.left_right_intent_topic,
            self.on_left_right_intent,
            qos,
        )
        self.pub_cmd = self.create_publisher(
            LocomotionCmd,
            self.locomotion_cmd_topic,
            qos,
        )
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)
        self.get_logger().info(
            "intent_command_coordinator ready: "
            f"{self.forward_backward_intent_topic} + {self.left_right_intent_topic} -> {self.locomotion_cmd_topic}, "
            f"publish={self.publish_hz:.1f}Hz, "
            f"front/back intent {self.forward_backward_intent_topic}: "
            f"{self.forward_intent_label}->{self.forward_x_vel:+.2f}m/s, "
            f"{self.backward_intent_label}->{self.backward_x_vel:+.2f}m/s, "
            f"idle={self.idle_intent_label}, timeout={self.forward_backward_intent_timeout_s:.2f}s, "
            f"left/right intent {self.left_right_intent_topic}: "
            f"{self.left_intent_label}->{self.left_y_vel:+.2f}m/s, "
            f"{self.right_intent_label}->{self.right_y_vel:+.2f}m/s, "
            f"left needs {self.left_window_size} matches, right needs {self.right_window_size} matches, "
            f"{self.zero_window_size} zeros -> hold 0.0 for {self.zero_hold_s:.2f}s"
        )

    def on_forward_backward_intent(self, msg: Int32) -> None:
        self.latest_forward_backward_intent = int(msg.data)
        self.last_forward_backward_intent_time_ns = self.get_clock().now().nanoseconds

    def on_left_right_intent(self, msg: Int32) -> None:
        intent = int(msg.data)
        now_ns = self.get_clock().now().nanoseconds
        self.latest_left_right_intent = intent
        self.last_left_right_intent_time_ns = now_ns
        self.left_right_recent.append(intent)
        recent = list(self.left_right_recent)
        if len(recent) >= self.zero_window_size and all(
            value == 0 for value in recent[-self.zero_window_size :]
        ):
            self.zero_hold_until_ns = now_ns + int(self.zero_hold_s * 1e9)

    def compute_y_vel(self) -> float:
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.latest_left_right_intent is None
            or self.last_left_right_intent_time_ns is None
            or (now_ns - self.last_left_right_intent_time_ns) * 1e-9
            > self.left_right_intent_timeout_s
        ):
            self.left_right_recent.clear()
            self.active_y_vel = self.base_y_vel
            self.zero_hold_until_ns = None
            return self.base_y_vel

        recent = list(self.left_right_recent)
        saw_zero_window = len(recent) >= self.zero_window_size and all(
            value == 0 for value in recent[-self.zero_window_size :]
        )

        if self.active_y_vel != self.base_y_vel:
            if saw_zero_window:
                self.active_y_vel = self.base_y_vel
                self.zero_hold_until_ns = now_ns + int(self.zero_hold_s * 1e9)
            else:
                return self.active_y_vel

        if self.zero_hold_until_ns is not None and now_ns < self.zero_hold_until_ns:
            return 0.0

        if len(recent) >= self.left_window_size and all(
            intent == self.left_intent_label for intent in recent[-self.left_window_size :]
        ):
            self.active_y_vel = self.left_y_vel
            return self.active_y_vel
        if len(recent) >= self.right_window_size and all(
            intent == self.right_intent_label for intent in recent[-self.right_window_size :]
        ):
            self.active_y_vel = self.right_y_vel
            return self.active_y_vel
        return self.base_y_vel

    def compute_x_vel(self) -> float:
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.latest_forward_backward_intent is None
            or self.last_forward_backward_intent_time_ns is None
            or (now_ns - self.last_forward_backward_intent_time_ns) * 1e-9
            > self.forward_backward_intent_timeout_s
        ):
            return 0.0

        if self.latest_forward_backward_intent == self.forward_intent_label:
            return self.forward_x_vel
        if self.latest_forward_backward_intent == self.backward_intent_label:
            return self.backward_x_vel
        if self.latest_forward_backward_intent == self.idle_intent_label:
            return 0.0
        return 0.0

    def on_timer(self) -> None:
        msg = LocomotionCmd()
        msg.stamp = self.get_clock().now().to_msg()
        msg.x_vel = self.compute_x_vel()
        msg.y_vel = self.compute_y_vel()
        msg.z_pos = self.z_pos
        msg.yaw_rate = 0.0
        self.pub_cmd.publish(msg)


def main() -> None:
    rclpy.init()
    node = IntentCommandCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

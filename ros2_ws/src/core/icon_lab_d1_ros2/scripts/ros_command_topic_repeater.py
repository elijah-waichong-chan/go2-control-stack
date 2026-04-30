#!/usr/bin/env python3

import copy

import rclpy
from rclpy.node import Node

from icon_lab_d1_ros2.msg import ServoCommand


DEFAULT_INPUT_TOPIC = "/arm/servo_command_input"
DEFAULT_OUTPUT_TOPIC = "/arm/servo_command"
DEFAULT_REPEAT_RATE_HZ = 20.0


class RosCommandTopicRepeater(Node):
    def __init__(self) -> None:
        super().__init__("ros_command_topic_repeater")
        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("repeat_rate_hz", DEFAULT_REPEAT_RATE_HZ)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.repeat_rate_hz = float(self.get_parameter("repeat_rate_hz").value)

        if self.repeat_rate_hz <= 0.0:
            raise ValueError("repeat_rate_hz must be > 0")

        self.last_command: ServoCommand | None = None
        self.subscription = self.create_subscription(
            ServoCommand, self.input_topic, self._handle_command, 10
        )
        self.publisher = self.create_publisher(ServoCommand, self.output_topic, 10)
        self.timer = self.create_timer(1.0 / self.repeat_rate_hz, self._publish_last_command)

        self.get_logger().info(
            f"Repeating ROS '{self.input_topic}' onto '{self.output_topic}' "
            f"at {self.repeat_rate_hz:.1f} Hz"
        )

    def _handle_command(self, msg: ServoCommand) -> None:
        self.last_command = copy.deepcopy(msg)
        self._publish_last_command()

    def _publish_last_command(self) -> None:
        if self.last_command is None:
            return

        repeated_command = copy.deepcopy(self.last_command)
        repeated_command.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(repeated_command)


def main() -> None:
    rclpy.init()
    node = RosCommandTopicRepeater()
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

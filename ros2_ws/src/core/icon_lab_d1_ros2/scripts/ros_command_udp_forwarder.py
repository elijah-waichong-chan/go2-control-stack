#!/usr/bin/env python3

import socket
import struct

import rclpy
from rclpy.node import Node

from icon_lab_d1_ros2.msg import ServoCommand


SERVO_COUNT = 7
DEFAULT_COMMAND_TOPIC = "/arm/servo_command_input"
DEFAULT_BIND_ADDRESS = "127.0.0.1"
DEFAULT_UDP_PORT = 17701
PACKET_FORMAT = "<29f"


class RosCommandUdpForwarder(Node):
    def __init__(self) -> None:
        super().__init__("ros_command_udp_forwarder")
        self.declare_parameter("command_topic", DEFAULT_COMMAND_TOPIC)
        self.declare_parameter("udp_address", DEFAULT_BIND_ADDRESS)
        self.declare_parameter("udp_port", DEFAULT_UDP_PORT)

        self.command_topic = self.get_parameter("command_topic").value
        self.udp_address = self.get_parameter("udp_address").value
        self.udp_port = int(self.get_parameter("udp_port").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.subscription = self.create_subscription(
            ServoCommand, self.command_topic, self._handle_command, 10
        )

        self.get_logger().info(
            f"Forwarding ROS '{self.command_topic}' to UDP {self.udp_address}:{self.udp_port}"
        )

    def _handle_command(self, msg: ServoCommand) -> None:
        if (
            len(msg.mode) != SERVO_COUNT
            or len(msg.angle_deg) != SERVO_COUNT
            or len(msg.velocity_deg_s) != SERVO_COUNT
            or len(msg.damping_power_mw) != SERVO_COUNT
        ):
            self.get_logger().warning(
                "Ignoring ServoCommand with unexpected array lengths"
            )
            return

        packet = struct.pack(
            PACKET_FORMAT,
            *[float(value) for value in msg.mode],
            *[float(value) for value in msg.angle_deg],
            *[float(value) for value in msg.velocity_deg_s],
            *[float(value) for value in msg.damping_power_mw],
            float(msg.interval_ms),
        )
        self.sock.sendto(packet, (self.udp_address, self.udp_port))

    def destroy_node(self) -> bool:
        self.sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RosCommandUdpForwarder()
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

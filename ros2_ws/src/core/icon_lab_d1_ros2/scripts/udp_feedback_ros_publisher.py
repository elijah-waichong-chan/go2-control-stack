#!/usr/bin/env python3

import socket
import struct

import rclpy
from rclpy.node import Node

from icon_lab_d1_ros2.msg import ServoFeedback


SERVO_COUNT = 7
DEFAULT_FEEDBACK_TOPIC = "/arm/servo_feedback"
DEFAULT_BIND_ADDRESS = "127.0.0.1"
DEFAULT_UDP_PORT = 17702
PACKET_FORMAT = "<21f"


class UdpFeedbackRosPublisher(Node):
    def __init__(self) -> None:
        super().__init__("udp_feedback_ros_publisher")
        self.declare_parameter("feedback_topic", DEFAULT_FEEDBACK_TOPIC)
        self.declare_parameter("udp_address", DEFAULT_BIND_ADDRESS)
        self.declare_parameter("udp_port", DEFAULT_UDP_PORT)
        self.declare_parameter("frame_id", "")

        self.feedback_topic = self.get_parameter("feedback_topic").value
        self.udp_address = self.get_parameter("udp_address").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.publisher = self.create_publisher(ServoFeedback, self.feedback_topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((self.udp_address, self.udp_port))

        self.timer = self.create_timer(0.01, self._poll_udp)

        self.get_logger().info(
            f"Publishing ROS '{self.feedback_topic}' from UDP {self.udp_address}:{self.udp_port}"
        )

    def _poll_udp(self) -> None:
        while True:
            try:
                packet, _addr = self.sock.recvfrom(struct.calcsize(PACKET_FORMAT))
            except BlockingIOError:
                return

            if len(packet) != struct.calcsize(PACKET_FORMAT):
                self.get_logger().warning(
                    f"Ignoring feedback UDP packet with unexpected size {len(packet)}"
                )
                continue

            values = struct.unpack(PACKET_FORMAT, packet)
            msg = ServoFeedback()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.angle_deg = [float(values[3 * i]) for i in range(SERVO_COUNT)]
            msg.current_ma = [float(values[3 * i + 1]) for i in range(SERVO_COUNT)]
            msg.velocity_deg_s = [float(values[3 * i + 2]) for i in range(SERVO_COUNT)]
            self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self.sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = UdpFeedbackRosPublisher()
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

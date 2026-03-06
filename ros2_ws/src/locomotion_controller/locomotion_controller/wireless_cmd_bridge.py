#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from go2_msgs.msg import LocomotionCmd, PushEvent
from unitree_go.msg import WirelessController


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class WirelessCmdBridge(Node):
    _A_MASK = 1 << 8
    _Y_MASK = 1 << 11
    _DPAD_UP_MASK = 1 << 12
    _DPAD_RIGHT_MASK = 1 << 13
    _DPAD_DOWN_MASK = 1 << 14
    _DPAD_LEFT_MASK = 1 << 15

    def __init__(self) -> None:
        super().__init__("wireless_cmd_bridge")

        self.declare_parameter("wireless_topic", "/wirelesscontroller")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("push_event_topic", "/data/push_event")
        self.declare_parameter("push_event_hz", 10.0)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("cmd_timeout_s", 0.5)
        self.declare_parameter("deadzone", 0.05)

        # Axis scales for command generation.
        self.declare_parameter("scale_x", 0.6)     # ly -> x_vel
        self.declare_parameter("scale_y", -0.4)    # lx -> y_vel (flipped)
        self.declare_parameter("scale_yaw", -1.2)  # rx -> yaw_rate (flipped)

        # Keep fixed values for policy inputs not controlled by wireless sticks.
        self.declare_parameter("z_pos", 0.27)
        self.declare_parameter("gait_hz", 3.0)

        self.wireless_topic = str(self.get_parameter("wireless_topic").value)
        self.cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.push_event_topic = str(self.get_parameter("push_event_topic").value)
        self.push_event_hz = float(self.get_parameter("push_event_hz").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.cmd_timeout_s = float(self.get_parameter("cmd_timeout_s").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.scale_x = float(self.get_parameter("scale_x").value)
        self.scale_y = float(self.get_parameter("scale_y").value)
        self.scale_yaw = float(self.get_parameter("scale_yaw").value)
        self.z_pos = float(self.get_parameter("z_pos").value)
        self.gait_hz = float(self.get_parameter("gait_hz").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_wireless = self.create_subscription(
            WirelessController, self.wireless_topic, self.on_wireless, qos
        )
        self.pub_cmd = self.create_publisher(LocomotionCmd, self.cmd_topic, qos)
        self.pub_push_event = self.create_publisher(PushEvent, self.push_event_topic, qos)

        self._lx = 0.0
        self._ly = 0.0
        self._rx = 0.0
        self._latest_keys = 0
        self._last_wireless_time = None

        period = 1.0 / max(self.publish_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)
        push_event_period = 1.0 / max(self.push_event_hz, 1.0)
        self.push_event_timer = self.create_timer(push_event_period, self.on_push_event_timer)
        self.get_logger().info(
            f"wireless_cmd_bridge running: {self.wireless_topic} -> {self.cmd_topic}, "
            f"push events -> {self.push_event_topic}"
        )

    def _apply_deadzone(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        return v

    def on_wireless(self, msg: WirelessController) -> None:
        keys = int(msg.keys) & 0xFFFF
        self._latest_keys = keys

        self._lx = self._apply_deadzone(float(msg.lx))
        self._ly = self._apply_deadzone(float(msg.ly))
        self._rx = self._apply_deadzone(float(msg.rx))
        self._last_wireless_time = time.monotonic()

    def _label_from_keys(self, keys: int) -> int:
        if keys & self._Y_MASK:
            return int(PushEvent.UP)
        if keys & self._A_MASK:
            return int(PushEvent.DOWN)
        if keys & self._DPAD_UP_MASK:
            return int(PushEvent.FWD)
        if keys & self._DPAD_DOWN_MASK:
            return int(PushEvent.BACK)
        if keys & self._DPAD_LEFT_MASK:
            return int(PushEvent.LEFT)
        if keys & self._DPAD_RIGHT_MASK:
            return int(PushEvent.RIGHT)
        return int(PushEvent.NO_PUSH)

    def _publish_push_event(self, label: int) -> None:
        msg = PushEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wireless_cmd_bridge"
        msg.label = int(label)
        self.pub_push_event.publish(msg)

    def on_push_event_timer(self) -> None:
        now = time.monotonic()
        stale = self._last_wireless_time is None or (now - self._last_wireless_time) > self.cmd_timeout_s
        if stale:
            self._publish_push_event(PushEvent.NO_PUSH)
            return
        self._publish_push_event(self._label_from_keys(self._latest_keys))

    def on_timer(self) -> None:
        now = time.monotonic()
        stale = self._last_wireless_time is None or (now - self._last_wireless_time) > self.cmd_timeout_s
        if stale:
            # Don't publish stale zeros continuously; let policy timeout handle it.
            return

        msg = LocomotionCmd()
        msg.stamp = self.get_clock().now().to_msg()
        msg.x_vel = _clamp(self._ly * self.scale_x, -abs(self.scale_x), abs(self.scale_x))
        msg.y_vel = _clamp(self._lx * self.scale_y, -abs(self.scale_y), abs(self.scale_y))
        msg.yaw_rate = _clamp(self._rx * self.scale_yaw, -abs(self.scale_yaw), abs(self.scale_yaw))
        msg.z_pos = self.z_pos
        msg.gait_hz = self.gait_hz
        self.pub_cmd.publish(msg)


def main() -> None:
    rclpy.init()
    node = WirelessCmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

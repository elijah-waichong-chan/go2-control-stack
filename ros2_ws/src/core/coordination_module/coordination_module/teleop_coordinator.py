#!/usr/bin/env python3
"""Publish tele-op locomotion commands, push events, and arm z-motion tasks."""

from __future__ import annotations

from collections import deque
import math
import time

import rclpy
from hq_pcot_msgs.msg import ArmTask, LocomotionCmd, LoopStatus, PushEvent
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from unitree_go.msg import WirelessController


def _make_loop_status(
    status_code: int,
    avg_loop_ms: float = -1.0,
    p99_loop_ms: float = -1.0,
    max_loop_ms: float = -1.0,
    budget_ms: float = -1.0,
    deadline_miss_count: int = -1,
    sample_count: int = -1,
) -> LoopStatus:
    msg = LoopStatus()
    msg.status = int(status_code)
    msg.avg_loop_ms = float(avg_loop_ms)
    msg.p99_loop_ms = float(p99_loop_ms)
    msg.max_loop_ms = float(max_loop_ms)
    msg.budget_ms = float(budget_ms)
    msg.deadline_miss_count = int(deadline_miss_count)
    msg.sample_count = int(sample_count)
    return msg


def _percentile99(samples: deque[float]) -> float:
    if not samples:
        return -1.0
    ordered = sorted(samples)
    idx = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return float(ordered[min(idx, len(ordered) - 1)])


class TeleopCoordinator(Node):
    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    A_MASK = 1 << 8
    Y_MASK = 1 << 11
    DPAD_UP_MASK = 1 << 12
    DPAD_RIGHT_MASK = 1 << 13
    DPAD_DOWN_MASK = 1 << 14
    DPAD_LEFT_MASK = 1 << 15

    def __init__(self) -> None:
        super().__init__("teleop_coordinator")

        self.declare_parameter("wireless_topic", "/wirelesscontroller")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("arm_task_topic", "/arm_task")
        self.declare_parameter("push_event_topic", "/data/push_event")
        self.declare_parameter("status_topic", "/status/coordination_module")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("push_event_hz", 10.0)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("cmd_timeout_s", 0.5)
        self.declare_parameter("deadzone", 0.05)
        self.declare_parameter("arm_deadzone", 0.2)
        self.declare_parameter("scale_x", 0.6)
        self.declare_parameter("scale_y", -0.4)
        self.declare_parameter("scale_yaw", -1.2)
        self.declare_parameter("z_pos", 0.27)

        self.wireless_topic = str(self.get_parameter("wireless_topic").value)
        self.locomotion_cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.arm_task_topic = str(self.get_parameter("arm_task_topic").value)
        self.push_event_topic = str(self.get_parameter("push_event_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.push_event_hz = max(1.0, float(self.get_parameter("push_event_hz").value))
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.cmd_timeout_s = max(0.0, float(self.get_parameter("cmd_timeout_s").value))
        self.deadzone = max(0.0, float(self.get_parameter("deadzone").value))
        self.arm_deadzone = max(0.0, float(self.get_parameter("arm_deadzone").value))
        self.scale_x = float(self.get_parameter("scale_x").value)
        self.scale_y = float(self.get_parameter("scale_y").value)
        self.scale_yaw = float(self.get_parameter("scale_yaw").value)
        self.z_pos = float(self.get_parameter("z_pos").value)
        self.loop_budget_ms = 1000.0 / max(self.publish_hz, 1.0)
        self.loop_stats_window = max(100, int(self.publish_hz * 10.0))

        wireless_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_wireless = self.create_subscription(
            WirelessController,
            self.wireless_topic,
            self.on_wireless,
            wireless_qos,
        )
        self.pub_cmd = self.create_publisher(LocomotionCmd, self.locomotion_cmd_topic, qos)
        self.pub_arm_task = self.create_publisher(ArmTask, self.arm_task_topic, qos)
        self.pub_push_event = self.create_publisher(PushEvent, self.push_event_topic, qos)
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)

        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)
        self.push_event_timer = self.create_timer(
            1.0 / self.push_event_hz,
            self.on_push_event_timer,
        )
        self.status_timer = self.create_timer(1.0 / self.status_hz, self.on_status_timer)

        self.last_wireless_time: float | None = None
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.latest_keys = 0
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.loop_times_ms: deque[float] = deque(maxlen=self.loop_stats_window)
        self.deadline_flags: deque[int] = deque(maxlen=self.loop_stats_window)
        self._update_status()

        self.get_logger().info(
            "teleop_coordinator ready: "
            f"{self.wireless_topic} -> {self.locomotion_cmd_topic} + {self.arm_task_topic} + {self.push_event_topic}, "
            f"publish={self.publish_hz:.1f}Hz"
        )

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.deadzone else float(value)

    def _is_stale(self) -> bool:
        if self.last_wireless_time is None:
            return True
        return (time.monotonic() - self.last_wireless_time) > self.cmd_timeout_s

    def _clamp(self, value: float, limit: float) -> float:
        abs_limit = abs(float(limit))
        return max(-abs_limit, min(abs_limit, float(value)))

    def _label_from_keys(self, keys: int) -> int:
        if (keys & self.Y_MASK) != 0:
            return PushEvent.UP
        if (keys & self.A_MASK) != 0:
            return PushEvent.DOWN
        if (keys & self.DPAD_UP_MASK) != 0:
            return PushEvent.FWD
        if (keys & self.DPAD_DOWN_MASK) != 0:
            return PushEvent.BACK
        if (keys & self.DPAD_LEFT_MASK) != 0:
            return PushEvent.LEFT
        if (keys & self.DPAD_RIGHT_MASK) != 0:
            return PushEvent.RIGHT
        return PushEvent.NO_PUSH

    def _compute_z_motion_direction(self) -> int:
        if self._is_stale():
            return 0
        if self.ry > self.arm_deadzone:
            return 1
        if self.ry < -self.arm_deadzone:
            return -1
        return 0

    def _publish_arm_task(self, z_motion_direction: int) -> None:
        msg = ArmTask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "teleop_coordinator"
        msg.task = int(ArmTask.TASK_UP_DOWN_TRACK)
        msg.z_motion_direction = int(max(-1, min(1, z_motion_direction)))
        self.pub_arm_task.publish(msg)

    def _publish_push_event(self, label: int) -> None:
        msg = PushEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "teleop_coordinator"
        msg.label = int(label)
        self.pub_push_event.publish(msg)

    def _set_status(self, status_code: int) -> None:
        status_code = int(status_code)
        if status_code == self.status_code:
            return
        self.status_code = status_code
        self._publish_status()

    def _record_loop_time_ms(self, loop_time_ms: float) -> None:
        self.loop_times_ms.append(float(loop_time_ms))
        self.deadline_flags.append(1 if loop_time_ms > self.loop_budget_ms else 0)

    def _publish_status(self) -> None:
        if not self.loop_times_ms:
            self.pub_status.publish(
                _make_loop_status(
                    self.status_code,
                    budget_ms=self.loop_budget_ms,
                )
            )
            return

        max_loop_ms = max(self.loop_times_ms)
        avg_loop_ms = sum(self.loop_times_ms) / len(self.loop_times_ms)
        p99_loop_ms = _percentile99(self.loop_times_ms)
        deadline_miss_count = sum(self.deadline_flags)
        sample_count = len(self.loop_times_ms)
        self.pub_status.publish(
            _make_loop_status(
                self.status_code,
                avg_loop_ms=avg_loop_ms,
                p99_loop_ms=p99_loop_ms,
                max_loop_ms=max_loop_ms,
                budget_ms=self.loop_budget_ms,
                deadline_miss_count=deadline_miss_count,
                sample_count=sample_count,
            )
        )

    def _update_status(self) -> None:
        if self._is_stale():
            if self.waiting_on != self.wireless_topic:
                self.get_logger().info(
                    f"teleop_coordinator waiting for {self.wireless_topic}..."
                )
                self.waiting_on = self.wireless_topic
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("teleop_coordinator running")
            self.running_logged = True

    def on_wireless(self, msg: WirelessController) -> None:
        self.latest_keys = int(msg.keys) & 0xFFFF
        self.lx = self._apply_deadzone(float(msg.lx))
        self.ly = self._apply_deadzone(float(msg.ly))
        self.rx = self._apply_deadzone(float(msg.rx))
        self.ry = self._apply_deadzone(float(msg.ry))
        self.last_wireless_time = time.monotonic()
        self._update_status()

    def on_push_event_timer(self) -> None:
        if self._is_stale():
            self._publish_push_event(PushEvent.NO_PUSH)
            return
        self._publish_push_event(self._label_from_keys(self.latest_keys))

    def on_status_timer(self) -> None:
        self._publish_status()

    def on_timer(self) -> None:
        start_ns = time.perf_counter_ns()
        try:
            self._update_status()
            if not self._is_stale():
                msg = LocomotionCmd()
                msg.stamp = self.get_clock().now().to_msg()
                msg.x_vel = self._clamp(self.ly * self.scale_x, self.scale_x)
                msg.y_vel = self._clamp(self.lx * self.scale_y, self.scale_y)
                msg.yaw_rate = self._clamp(self.rx * self.scale_yaw, self.scale_yaw)
                msg.z_pos = self.z_pos
                self.pub_cmd.publish(msg)

            self._publish_arm_task(self._compute_z_motion_direction())
        finally:
            loop_time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._record_loop_time_ms(loop_time_ms)


def main() -> None:
    rclpy.init()
    node = TeleopCoordinator()
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

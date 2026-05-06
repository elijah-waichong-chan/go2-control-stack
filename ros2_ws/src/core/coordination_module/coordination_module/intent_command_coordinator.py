#!/usr/bin/env python3
"""Publish coordinated locomotion commands and arm tasks from intent signals."""

from __future__ import annotations

from collections import deque
import math
import time

import rclpy
from hq_pcot_msgs.msg import ArmTask, LocomotionCmd, LoopStatus
from icon_lab_d1_ros2.msg import ServoCommand, ServoFeedback
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32
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


class AutonomousCoordinator(Node):
    """Generate base commands and arm tasks from filtered transport intents."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    L1_BUTTON_MASK = 1 << 1
    B_BUTTON_MASK = 1 << 9
    X_BUTTON_MASK = 1 << 10
    FRONT_BACK_NEUTRAL_JOINT0_DEG = 90.0
    UP_DOWN_IDLE_INTENT_LABEL = 0
    UP_DOWN_INCREASE_INTENT_LABEL = 5
    UP_DOWN_DECREASE_INTENT_LABEL = 6

    def __init__(self) -> None:
        super().__init__("autonomous_coordinator")

        self.declare_parameter("forward_backward_intent_topic", "/direction_intent/front_back/label")
        self.declare_parameter("left_right_intent_topic", "/direction_intent/left_right/label")
        self.declare_parameter("up_down_intent_topic", "/direction_intent/up_down/label")
        self.declare_parameter("arm_feedback_topic", "/arm/servo_feedback")
        self.declare_parameter("wireless_topic", "/wirelesscontroller")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("arm_task_topic", "/arm_task")
        self.declare_parameter("gripper_command_topic", "/arm/servo_command_input")
        self.declare_parameter("gripper_servo_index", 6)
        self.declare_parameter("gripper_open_angle_deg", 50.0)
        self.declare_parameter("gripper_close_angle_deg", -20.0)
        self.declare_parameter("gripper_velocity_deg_s", 60.0)
        self.declare_parameter("status_topic", "/status/coordination_module")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("forward_x_vel", 0.5)
        self.declare_parameter("backward_x_vel", -0.5)
        self.declare_parameter("front_back_x_vel_kp", 0.02)
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
        self.declare_parameter("left_y_vel", 0.3)
        self.declare_parameter("right_y_vel", -0.3)
        self.declare_parameter("left_right_intent_timeout_s", 0.5)
        self.declare_parameter("up_down_intent_timeout_s", 0.5)
        self.declare_parameter("wireless_timeout_s", 0.5)

        self.forward_backward_intent_topic = str(
            self.get_parameter("forward_backward_intent_topic").value
        )
        self.left_right_intent_topic = str(
            self.get_parameter("left_right_intent_topic").value
        )
        self.up_down_intent_topic = str(self.get_parameter("up_down_intent_topic").value)
        self.arm_feedback_topic = str(self.get_parameter("arm_feedback_topic").value)
        self.wireless_topic = str(self.get_parameter("wireless_topic").value)
        self.locomotion_cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.arm_task_topic = str(self.get_parameter("arm_task_topic").value)
        self.gripper_command_topic = str(self.get_parameter("gripper_command_topic").value)
        self.gripper_servo_index = int(self.get_parameter("gripper_servo_index").value)
        self.gripper_open_angle_deg = float(self.get_parameter("gripper_open_angle_deg").value)
        self.gripper_close_angle_deg = float(self.get_parameter("gripper_close_angle_deg").value)
        self.gripper_velocity_deg_s = float(self.get_parameter("gripper_velocity_deg_s").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.forward_x_vel = float(self.get_parameter("forward_x_vel").value)
        self.backward_x_vel = float(self.get_parameter("backward_x_vel").value)
        self.front_back_x_vel_kp = float(self.get_parameter("front_back_x_vel_kp").value)
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
        self.up_down_intent_timeout_s = max(
            0.0, float(self.get_parameter("up_down_intent_timeout_s").value)
        )
        self.wireless_timeout_s = max(
            0.0, float(self.get_parameter("wireless_timeout_s").value)
        )
        if self.gripper_servo_index < 0 or self.gripper_servo_index >= 7:
            raise ValueError("gripper_servo_index must be between 0 and 6")
        if self.gripper_velocity_deg_s <= 0.0:
            raise ValueError("gripper_velocity_deg_s must be > 0")
        self.loop_budget_ms = 1000.0 / max(self.publish_hz, 1.0)
        self.loop_stats_window = max(100, int(self.publish_hz * 10.0))

        self.latest_forward_backward_intent: int | None = None
        self.last_forward_backward_intent_time_ns: int | None = None
        self.latest_left_right_intent: int | None = None
        self.last_left_right_intent_time_ns: int | None = None
        self.left_right_recent: deque[int] = deque(maxlen=self.left_right_window_size)
        self.active_y_vel = self.base_y_vel
        self.zero_hold_until_ns: int | None = None

        self.latest_up_down_intent: int | None = None
        self.last_up_down_intent_time: float | None = None
        self.latest_arm_angles_deg: list[float] | None = None
        self.latest_wireless_keys = 0
        self.previous_wireless_keys = 0
        self.last_wireless_time: float | None = None
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.loop_times_ms: deque[float] = deque(maxlen=self.loop_stats_window)
        self.deadline_flags: deque[int] = deque(maxlen=self.loop_stats_window)

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
        wireless_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
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
        self.sub_up_down_intent = self.create_subscription(
            Int32,
            self.up_down_intent_topic,
            self.on_up_down_intent,
            qos,
        )
        self.sub_arm_feedback = self.create_subscription(
            ServoFeedback,
            self.arm_feedback_topic,
            self.on_arm_feedback,
            qos,
        )
        self.sub_wireless = self.create_subscription(
            WirelessController,
            self.wireless_topic,
            self.on_wireless,
            wireless_qos,
        )
        self.pub_cmd = self.create_publisher(LocomotionCmd, self.locomotion_cmd_topic, qos)
        self.pub_arm_task = self.create_publisher(ArmTask, self.arm_task_topic, qos)
        self.pub_gripper_cmd = self.create_publisher(
            ServoCommand, self.gripper_command_topic, qos
        )
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)
        self.status_timer = self.create_timer(1.0 / self.status_hz, self.on_status_timer)
        self._update_status()
        self.get_logger().info(
            "autonomous_coordinator ready: "
            f"{self.forward_backward_intent_topic} + {self.left_right_intent_topic} + "
            f"{self.up_down_intent_topic} + {self.arm_feedback_topic} + {self.wireless_topic} -> "
            f"{self.locomotion_cmd_topic} + {self.arm_task_topic} + {self.gripper_command_topic}, "
            f"gripper(index={self.gripper_servo_index}, open={self.gripper_open_angle_deg:.1f}deg, "
            f"close={self.gripper_close_angle_deg:.1f}deg), "
            f"publish={self.publish_hz:.1f}Hz"
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

    def on_up_down_intent(self, msg: Int32) -> None:
        self.latest_up_down_intent = int(msg.data)
        self.last_up_down_intent_time = time.monotonic()

    def on_arm_feedback(self, msg: ServoFeedback) -> None:
        if len(msg.angle_deg) < 6:
            return
        self.latest_arm_angles_deg = [float(value) for value in msg.angle_deg[:6]]
        self._update_status()

    def on_wireless(self, msg: WirelessController) -> None:
        keys = int(msg.keys) & 0xFFFF
        pressed = keys & ~self.previous_wireless_keys
        if pressed & self.X_BUTTON_MASK:
            self._publish_gripper_command(self.gripper_open_angle_deg)
        if pressed & self.B_BUTTON_MASK:
            self._publish_gripper_command(self.gripper_close_angle_deg)
        self.previous_wireless_keys = keys
        self.latest_wireless_keys = keys
        self.last_wireless_time = time.monotonic()

    def _set_status(self, status_code: int) -> None:
        status_code = int(status_code)
        if status_code == self.status_code:
            return
        self.status_code = status_code
        self._publish_status()

    def on_status_timer(self) -> None:
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
        if self.latest_arm_angles_deg is None:
            if self.waiting_on != self.arm_feedback_topic:
                self.get_logger().info(
                    f"autonomous_coordinator waiting for {self.arm_feedback_topic}..."
                )
                self.waiting_on = self.arm_feedback_topic
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("autonomous_coordinator running")
            self.running_logged = True

    def _is_l1_held(self) -> bool:
        if self.last_wireless_time is None:
            return False
        if (time.monotonic() - self.last_wireless_time) > self.wireless_timeout_s:
            return False
        return bool(self.latest_wireless_keys & self.L1_BUTTON_MASK)

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

        if self.latest_arm_angles_deg is not None and self.latest_arm_angles_deg:
            joint0_deg = float(self.latest_arm_angles_deg[0])
            joint0_error_deg = self.FRONT_BACK_NEUTRAL_JOINT0_DEG - joint0_deg
            commanded_x_vel = self.front_back_x_vel_kp * joint0_error_deg

            if self.latest_forward_backward_intent == self.forward_intent_label:
                return max(0.0, min(self.forward_x_vel, commanded_x_vel))
            if self.latest_forward_backward_intent == self.backward_intent_label:
                return min(0.0, max(self.backward_x_vel, commanded_x_vel))
            if self.latest_forward_backward_intent == self.idle_intent_label:
                return 0.0
            return 0.0

        if self.latest_forward_backward_intent == self.forward_intent_label:
            return self.forward_x_vel
        if self.latest_forward_backward_intent == self.backward_intent_label:
            return self.backward_x_vel
        if self.latest_forward_backward_intent == self.idle_intent_label:
            return 0.0
        return 0.0

    def _build_tracking_task(self, task_type: int, z_motion_direction: int = 0) -> ArmTask:
        msg = ArmTask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "autonomous_coordinator"
        msg.task = int(task_type)
        msg.z_motion_direction = int(z_motion_direction)
        return msg

    def _publish_gripper_command(self, target_angle_deg: float) -> None:
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.mode = [ServoCommand.MODE_IGNORE] * 7
        command.mode[self.gripper_servo_index] = ServoCommand.MODE_POSITION_VELOCITY
        command.angle_deg = [math.nan] * 7
        command.angle_deg[self.gripper_servo_index] = float(target_angle_deg)
        command.velocity_deg_s = [math.nan] * 7
        command.velocity_deg_s[self.gripper_servo_index] = float(self.gripper_velocity_deg_s)
        command.damping_power_mw = [0.0] * 7
        command.interval_ms = 0
        self.pub_gripper_cmd.publish(command)

    def _compute_z_motion_direction(self) -> int:
        now = time.monotonic()
        if (
            self.latest_up_down_intent is None
            or self.last_up_down_intent_time is None
            or (now - self.last_up_down_intent_time) > self.up_down_intent_timeout_s
        ):
            return 0
        if self.latest_up_down_intent == self.UP_DOWN_INCREASE_INTENT_LABEL:
            return 1
        if self.latest_up_down_intent == self.UP_DOWN_DECREASE_INTENT_LABEL:
            return -1
        return 0

    def _compute_arm_task(self) -> ArmTask | None:
        if self.latest_arm_angles_deg is None:
            return None

        return self._build_tracking_task(
            ArmTask.TASK_UP_DOWN_TRACK,
            z_motion_direction=self._compute_z_motion_direction(),
        )

    def on_timer(self) -> None:
        start_ns = time.perf_counter_ns()
        try:
            self._update_status()
            locomotion_msg = LocomotionCmd()
            locomotion_msg.stamp = self.get_clock().now().to_msg()
            locomotion_msg.x_vel = self.compute_x_vel()
            locomotion_msg.y_vel = self.compute_y_vel()
            locomotion_msg.z_pos = self.z_pos
            locomotion_msg.yaw_rate = 0.0
            if not self._is_l1_held():
                self.pub_cmd.publish(locomotion_msg)

            arm_task = self._compute_arm_task()
            if arm_task is not None:
                self.pub_arm_task.publish(arm_task)
        finally:
            loop_time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._record_loop_time_ms(loop_time_ms)


def main() -> None:
    rclpy.init()
    node = AutonomousCoordinator()
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

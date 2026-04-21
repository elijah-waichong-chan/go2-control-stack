#!/usr/bin/env python3
"""Publish coordinated locomotion commands and arm tasks from intent signals."""

from __future__ import annotations

from collections import deque
import time

import rclpy
from hq_pcot_msgs.msg import ArmState, ArmTask, LocomotionCmd
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32


class IntentCommandCoordinator(Node):
    """Generate base commands and arm tasks from filtered transport intents."""

    FRONT_BACK_ARM_POSE = [90.0, -10.0, 10.0, -85.0, -3.0, -90.0]
    FRONT_BACK_PRESET_MASK = (1, 1, 1, 1, 1, 1)
    FRONT_BACK_MODE_RANGE_DEG = (-85.0, -105.0)

    UP_DOWN_ARM_POSE = [None, None, None, 5.0, None, -90.0]
    UP_DOWN_PRESET_MASK = (0, 0, 0, 1, 0, 1)
    UP_DOWN_IDLE_INTENT_LABEL = 0
    UP_DOWN_INCREASE_INTENT_LABEL = 5
    UP_DOWN_DECREASE_INTENT_LABEL = 6
    UP_DOWN_MODE_RANGE_DEG = (-5.0, 5.0)

    MODE_SWITCH_JOINT5_CURRENT_THRESHOLD = 30.0
    MOTION_SETTLE_DELAY_S = 2.0

    def __init__(self) -> None:
        super().__init__("intent_command_coordinator")

        self.declare_parameter("forward_backward_intent_topic", "/direction_intent/front_back")
        self.declare_parameter("left_right_intent_topic", "/direction_intent/left_right")
        self.declare_parameter("up_down_intent_topic", "/direction_intent/up_down")
        self.declare_parameter("arm_state_topic", "/arm/state")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("arm_task_topic", "/arm_task")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("forward_x_vel", 0.5)
        self.declare_parameter("backward_x_vel", -0.5)
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

        self.forward_backward_intent_topic = str(
            self.get_parameter("forward_backward_intent_topic").value
        )
        self.left_right_intent_topic = str(
            self.get_parameter("left_right_intent_topic").value
        )
        self.up_down_intent_topic = str(self.get_parameter("up_down_intent_topic").value)
        self.arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self.locomotion_cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.arm_task_topic = str(self.get_parameter("arm_task_topic").value)
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
        self.up_down_intent_timeout_s = max(
            0.0, float(self.get_parameter("up_down_intent_timeout_s").value)
        )

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
        self.latest_arm_currents: list[float] | None = None
        self.active_arm_mode: str | None = None
        self.joint5_current_switch_latched = False
        self.next_mode_switch_time = 0.0

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
        self.sub_up_down_intent = self.create_subscription(
            Int32,
            self.up_down_intent_topic,
            self.on_up_down_intent,
            qos,
        )
        self.sub_arm_state = self.create_subscription(
            ArmState,
            self.arm_state_topic,
            self.on_arm_state,
            qos,
        )
        self.pub_cmd = self.create_publisher(LocomotionCmd, self.locomotion_cmd_topic, qos)
        self.pub_arm_task = self.create_publisher(ArmTask, self.arm_task_topic, qos)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)
        self.get_logger().info(
            "intent_command_coordinator ready: "
            f"{self.forward_backward_intent_topic} + {self.left_right_intent_topic} + "
            f"{self.up_down_intent_topic} + {self.arm_state_topic} -> "
            f"{self.locomotion_cmd_topic} + {self.arm_task_topic}, "
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

    def on_arm_state(self, msg: ArmState) -> None:
        if len(msg.angle_deg) < 6:
            return
        self.latest_arm_angles_deg = [float(value) for value in msg.angle_deg[:6]]
        self.latest_arm_currents = [float(value) for value in msg.current[:6]]

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

    def _infer_arm_mode(self) -> str | None:
        if self.latest_arm_angles_deg is None:
            return None
        joint5_angle = float(self.latest_arm_angles_deg[5])
        if min(self.FRONT_BACK_MODE_RANGE_DEG) <= joint5_angle <= max(
            self.FRONT_BACK_MODE_RANGE_DEG
        ):
            return "front_back_mode"
        if min(self.UP_DOWN_MODE_RANGE_DEG) <= joint5_angle <= max(
            self.UP_DOWN_MODE_RANGE_DEG
        ):
            return "up_down_mode"
        return "front_back_mode"

    def _materialize_partial_pose(
        self,
        partial_pose_deg: list[float | None],
    ) -> list[float] | None:
        if all(value is not None for value in partial_pose_deg):
            return [float(value) for value in partial_pose_deg]
        if self.latest_arm_angles_deg is None:
            return None
        target_deg = list(self.latest_arm_angles_deg)
        for joint_index, value in enumerate(partial_pose_deg):
            if value is not None:
                target_deg[joint_index] = float(value)
        return target_deg

    def _build_preset_task(
        self,
        task_type: int,
        preset_joint_configuration_deg: list[float],
        preset_joint_update_mask: tuple[int, int, int, int, int, int],
    ) -> ArmTask:
        msg = ArmTask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "intent_command_coordinator"
        msg.task = int(task_type)
        msg.z_motion_direction = 0
        msg.preset_joint_configuration_deg = [
            float(value) for value in preset_joint_configuration_deg
        ]
        msg.preset_joint_update_mask = [int(value) for value in preset_joint_update_mask]
        return msg

    def _build_tracking_task(self, task_type: int, z_motion_direction: int = 0) -> ArmTask:
        msg = ArmTask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "intent_command_coordinator"
        msg.task = int(task_type)
        msg.z_motion_direction = int(z_motion_direction)
        return msg

    def _set_next_arm_mode(self, next_mode: str, now: float) -> ArmTask | None:
        if next_mode == "up_down_mode":
            target_deg = self._materialize_partial_pose(self.UP_DOWN_ARM_POSE)
            if target_deg is None:
                return None
            task_msg = self._build_preset_task(
                ArmTask.TASK_UP_DOWN_PRESET,
                target_deg,
                self.UP_DOWN_PRESET_MASK,
            )
        else:
            target_deg = self._materialize_partial_pose(self.FRONT_BACK_ARM_POSE)
            if target_deg is None:
                return None
            task_msg = self._build_preset_task(
                ArmTask.TASK_FRONT_BACK_PRESET,
                target_deg,
                self.FRONT_BACK_PRESET_MASK,
            )

        self.active_arm_mode = next_mode
        self.next_mode_switch_time = now + self.MOTION_SETTLE_DELAY_S
        return task_msg

    def _update_arm_mode_from_currents(self, now: float) -> ArmTask | None:
        if self.latest_arm_currents is None:
            return None

        current_mode = self.active_arm_mode
        if current_mode is None:
            current_mode = self._infer_arm_mode()
            if current_mode is None:
                return None
            self.active_arm_mode = current_mode

        next_mode = current_mode
        joint5_current = (
            float(self.latest_arm_currents[5])
            if len(self.latest_arm_currents) > 5
            else 0.0
        )
        if joint5_current > self.MODE_SWITCH_JOINT5_CURRENT_THRESHOLD:
            if (
                not self.joint5_current_switch_latched
                and now >= self.next_mode_switch_time
            ):
                next_mode = (
                    "up_down_mode"
                    if current_mode == "front_back_mode"
                    else "front_back_mode"
                )
                self.joint5_current_switch_latched = True
        else:
            self.joint5_current_switch_latched = False

        if current_mode != next_mode:
            return self._set_next_arm_mode(next_mode, now)
        return None

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

        now = time.monotonic()
        preset_task = self._update_arm_mode_from_currents(now)
        if preset_task is not None:
            return preset_task

        if self.active_arm_mode == "up_down_mode":
            return self._build_tracking_task(
                ArmTask.TASK_UP_DOWN_TRACK,
                z_motion_direction=self._compute_z_motion_direction(),
            )
        return self._build_tracking_task(ArmTask.TASK_FRONT_BACK_TRACK)

    def on_timer(self) -> None:
        locomotion_msg = LocomotionCmd()
        locomotion_msg.stamp = self.get_clock().now().to_msg()
        locomotion_msg.x_vel = self.compute_x_vel()
        locomotion_msg.y_vel = self.compute_y_vel()
        locomotion_msg.z_pos = self.z_pos
        locomotion_msg.yaw_rate = 0.0
        self.pub_cmd.publish(locomotion_msg)

        arm_task = self._compute_arm_task()
        if arm_task is not None:
            self.pub_arm_task.publish(arm_task)


def main() -> None:
    rclpy.init()
    node = IntentCommandCoordinator()
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

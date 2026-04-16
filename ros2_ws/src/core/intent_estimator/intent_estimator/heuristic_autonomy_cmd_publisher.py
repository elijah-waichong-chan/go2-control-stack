#!/usr/bin/env python3
"""Publish locomotion commands from the first arm joint angle heuristic."""

from __future__ import annotations

from collections import deque
import math

import rclpy
from go2_msgs.msg import ArmAngles, LocomotionCmd
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32
from unitree_go.msg import LowState


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class HeuristicAutonomyCmdPublisher(Node):
    """Drive `/locomotion_cmd` from heuristic arm and intent signals."""

    def __init__(self) -> None:
        super().__init__("heuristic_autonomy_cmd_publisher")

        self.declare_parameter("arm_angles_topic", "/arm_angles")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("left_right_intent_topic", "/direction_intent/left_right")
        self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("forward_x_vel", 0.5)
        self.declare_parameter("backward_x_vel", -0.5)
        self.declare_parameter("forward_min_deg", 30.0)
        self.declare_parameter("forward_max_deg", 85.0)
        self.declare_parameter("backward_min_deg", 95.0)
        self.declare_parameter("backward_max_deg", 150.0)
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
        self.declare_parameter("heading_hold_kp", 1.2)
        self.declare_parameter("heading_hold_kd", 0.0)
        self.declare_parameter("heading_hold_max_yaw_rate", 0.6)
        self.declare_parameter("heading_hold_deadband_deg", 3.0)
        self.declare_parameter("lowstate_timeout_s", 0.5)
        self.declare_parameter("yaw_source", "rpy")

        self.arm_angles_topic = str(self.get_parameter("arm_angles_topic").value)
        self.lowstate_topic = str(self.get_parameter("lowstate_topic").value)
        self.left_right_intent_topic = str(
            self.get_parameter("left_right_intent_topic").value
        )
        self.locomotion_cmd_topic = str(self.get_parameter("locomotion_cmd_topic").value)
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.forward_x_vel = float(self.get_parameter("forward_x_vel").value)
        self.backward_x_vel = float(self.get_parameter("backward_x_vel").value)
        self.forward_min_deg = float(self.get_parameter("forward_min_deg").value)
        self.forward_max_deg = float(self.get_parameter("forward_max_deg").value)
        self.backward_min_deg = float(self.get_parameter("backward_min_deg").value)
        self.backward_max_deg = float(self.get_parameter("backward_max_deg").value)
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
        self.heading_hold_kp = float(self.get_parameter("heading_hold_kp").value)
        self.heading_hold_kd = float(self.get_parameter("heading_hold_kd").value)
        self.heading_hold_max_yaw_rate = abs(
            float(self.get_parameter("heading_hold_max_yaw_rate").value)
        )
        self.heading_hold_deadband_rad = math.radians(
            float(self.get_parameter("heading_hold_deadband_deg").value)
        )
        self.lowstate_timeout_s = max(
            0.0, float(self.get_parameter("lowstate_timeout_s").value)
        )
        self.yaw_source = str(self.get_parameter("yaw_source").value).strip().lower()

        self.x_vel = 0.0
        self.initial_yaw_rad: float | None = None
        self.latest_yaw_rad: float | None = None
        self.latest_yaw_rate_rad_s = 0.0
        self.latest_yaw_source = "unknown"
        self.last_lowstate_time_ns: int | None = None
        self.latest_left_right_intent: int | None = None
        self.last_left_right_intent_time_ns: int | None = None
        self.left_right_recent: deque[int] = deque(maxlen=self.left_right_window_size)
        self.active_y_vel = self.base_y_vel
        self.zero_hold_until_ns: int | None = None
        self.lowstate_stale_logged = False
        self.last_debug_log_time_ns: int | None = None
        self.debug_log_period_s = 0.5

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            self.arm_angles_topic,
            self.on_arm_angles,
            qos,
        )
        self.sub_lowstate = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.on_lowstate,
            sensor_qos,
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
            "heuristic_autonomy_cmd_publisher ready: "
            f"{self.arm_angles_topic} -> {self.locomotion_cmd_topic}, "
            f"publish={self.publish_hz:.1f}Hz, "
            f"joint0 forward in [{self.forward_min_deg:.1f}, {self.forward_max_deg:.1f}] -> {self.forward_x_vel:.2f}, "
            f"backward in [{self.backward_min_deg:.1f}, {self.backward_max_deg:.1f}] -> {self.backward_x_vel:.2f}, "
            f"left/right intent {self.left_right_intent_topic}: "
            f"{self.left_intent_label}->{self.left_y_vel:+.2f}m/s, "
            f"{self.right_intent_label}->{self.right_y_vel:+.2f}m/s, "
            f"left needs {self.left_window_size} matches, right needs {self.right_window_size} matches, "
            f"{self.zero_window_size} zeros -> hold 0.0 for {self.zero_hold_s:.2f}s, "
            f"yaw_source={self.yaw_source}, "
            f"heading hold from {self.lowstate_topic} with kp={self.heading_hold_kp:.2f}, "
            f"kd={self.heading_hold_kd:.2f}, max_yaw_rate={self.heading_hold_max_yaw_rate:.2f}"
        )

    def on_arm_angles(self, msg: ArmAngles) -> None:
        if len(msg.angle_deg) == 0:
            return

        joint0_deg = float(msg.angle_deg[0])
        if self.forward_min_deg <= joint0_deg <= self.forward_max_deg:
            self.x_vel = self.forward_x_vel
        elif self.backward_min_deg <= joint0_deg <= self.backward_max_deg:
            self.x_vel = self.backward_x_vel
        else:
            self.x_vel = 0.0

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

    def on_lowstate(self, msg: LowState) -> None:
        yaw_rad, yaw_source = self.extract_yaw(msg)
        self.latest_yaw_rad = yaw_rad
        self.latest_yaw_source = yaw_source
        self.latest_yaw_rate_rad_s = float(msg.imu_state.gyroscope[2])
        self.last_lowstate_time_ns = self.get_clock().now().nanoseconds
        self.lowstate_stale_logged = False

        if self.initial_yaw_rad is None:
            self.initial_yaw_rad = yaw_rad
            self.get_logger().info(
                f"Captured initial yaw reference: {math.degrees(yaw_rad):.1f} deg ({yaw_source})"
            )

    def extract_yaw(self, msg: LowState) -> tuple[float, str]:
        use_rpy = self.yaw_source != "quaternion"
        yaw_rpy = float(msg.imu_state.rpy[2])
        if use_rpy and math.isfinite(yaw_rpy):
            return yaw_rpy, "rpy"

        q = msg.imu_state.quaternion
        yaw_quat = _quat_to_yaw(
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        )
        return yaw_quat, "quaternion"

    def compute_heading_hold_yaw_rate(self) -> float:
        if self.initial_yaw_rad is None or self.latest_yaw_rad is None:
            return 0.0

        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_lowstate_time_ns is None
            or (now_ns - self.last_lowstate_time_ns) * 1e-9 > self.lowstate_timeout_s
        ):
            if not self.lowstate_stale_logged:
                self.get_logger().warn(
                    "Lowstate is stale; disabling heading correction until IMU updates resume."
                )
                self.lowstate_stale_logged = True
            return 0.0

        yaw_error_rad = _wrap_angle_rad(self.initial_yaw_rad - self.latest_yaw_rad)
        if abs(yaw_error_rad) < self.heading_hold_deadband_rad:
            return 0.0

        correction = (
            self.heading_hold_kp * yaw_error_rad
            - self.heading_hold_kd * self.latest_yaw_rate_rad_s
        )
        return max(
            -self.heading_hold_max_yaw_rate,
            min(self.heading_hold_max_yaw_rate, correction),
        )

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

        # Once lateral motion is active, keep it latched until an explicit zero window arrives.
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

    def maybe_log_heading_debug(self, commanded_yaw_rate: float, commanded_y_vel: float) -> None:
        if self.initial_yaw_rad is None or self.latest_yaw_rad is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_debug_log_time_ns is not None
            and (now_ns - self.last_debug_log_time_ns) * 1e-9 < self.debug_log_period_s
        ):
            return

        yaw_error_rad = _wrap_angle_rad(self.initial_yaw_rad - self.latest_yaw_rad)
        self.get_logger().info(
            "heading_hold "
            f"src={self.latest_yaw_source} "
            f"yaw={math.degrees(self.latest_yaw_rad):.1f}deg "
            f"ref={math.degrees(self.initial_yaw_rad):.1f}deg "
            f"err={math.degrees(yaw_error_rad):+.1f}deg "
            f"lr_intent={self.latest_left_right_intent} "
            f"lr_recent={list(self.left_right_recent)} "
            f"zero_hold_active={bool(self.zero_hold_until_ns is not None and now_ns < self.zero_hold_until_ns)} "
            f"cmd_y={commanded_y_vel:+.3f}m/s "
            f"gyro_z={self.latest_yaw_rate_rad_s:+.3f}rad/s "
            f"cmd_yaw_rate={commanded_yaw_rate:+.3f}rad/s"
        )
        self.last_debug_log_time_ns = now_ns

    def on_timer(self) -> None:
        msg = LocomotionCmd()
        msg.stamp = self.get_clock().now().to_msg()
        msg.x_vel = self.x_vel
        msg.y_vel = self.compute_y_vel()
        msg.z_pos = self.z_pos
        msg.yaw_rate = self.compute_heading_hold_yaw_rate()
        self.maybe_log_heading_debug(msg.yaw_rate, msg.y_vel)
        self.pub_cmd.publish(msg)


def main() -> None:
    rclpy.init()
    node = HeuristicAutonomyCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

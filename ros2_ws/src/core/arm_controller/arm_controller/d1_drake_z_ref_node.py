#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from hq_pcot_msgs.msg import ArmState, ArmTarget
from unitree_arm.msg import ArmString
from unitree_go.msg import WirelessController

from arm_controller.d1_drake_ik_solver import D1DrakeIKSolver


class D1DrakeZReferenceNode(Node):
    JOINT_PUBLISH_THRESHOLD_DEG = 0.1
    COMMAND_PUBLISH_HZ = 10.0
    WIRELESS_CONTROL_HZ = 30.0
    WIRELESS_INPUT_TIMEOUT_S = 0.2
    IK_SOLVE_LOG_INTERVAL_S = 0.25
    Z_REFERENCE_LOG_INTERVAL_S = 0.25
    SINGLE_JOINT_INDICES = (1, 2, 4)
    STARTUP_ANGLES_DEG = (90.0, -10.0, 10.0, 5.0, -3.0, -90.0, -10.0)
    STARTUP_MODE_DELAY_S = 0.1
    STARTUP_DAMPING_DELAY_S = 3.0

    def __init__(self) -> None:
        super().__init__("d1_drake_z_ref_node")

        self.declare_parameter("arm_state_topic", "/arm/state")
        self.declare_parameter("arm_command_topic", "/arm_Command")
        self.declare_parameter("wireless_topic", "/wirelesscontroller")
        self.declare_parameter("z_step_m", 0.05)
        self.declare_parameter("z_velocity_mps", 0.20)
        self.declare_parameter("z_ref_min_m", -0.09)
        self.declare_parameter("z_ref_max_m", 0.56)
        self.declare_parameter("ry_deadzone", 0.10)

        arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        arm_command_topic = str(self.get_parameter("arm_command_topic").value)
        wireless_topic = str(self.get_parameter("wireless_topic").value)
        self.z_step_m = float(self.get_parameter("z_step_m").value)
        self.z_velocity_mps = abs(float(self.get_parameter("z_velocity_mps").value))
        self.z_ref_min_m = float(self.get_parameter("z_ref_min_m").value)
        self.z_ref_max_m = float(self.get_parameter("z_ref_max_m").value)
        self.ry_deadzone = min(1.0, max(0.0, float(self.get_parameter("ry_deadzone").value)))
        if self.z_ref_min_m > self.z_ref_max_m:
            self.z_ref_min_m, self.z_ref_max_m = self.z_ref_max_m, self.z_ref_min_m

        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        wireless_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.solver = D1DrakeIKSolver()
        self.q0_deg: np.ndarray | None = None
        self.q_out: np.ndarray | None = None
        self.latest_q_in: np.ndarray | None = None
        self.z_reference: float | None = None
        self._pending_q_out_solver: np.ndarray | None = None
        self._pending_q_out: np.ndarray | None = None
        self._q0_seeded = False
        self._q0_ready_at = time.monotonic() + self.STARTUP_DAMPING_DELAY_S
        self._latest_right_stick_y = 0.0
        self._last_wireless_message_time = 0.0
        self._last_wireless_control_time = time.monotonic()
        self._last_ik_solve_log_time = 0.0
        self._last_z_reference_log_time = 0.0
        self._startup_mode_timer = None
        self._pending_single_mode_timers: list = []

        self.pub_arm_command = self.create_publisher(
            ArmString,
            arm_command_topic,
            pub_qos,
        )
        self.sub_arm_angles = self.create_subscription(
            ArmState,
            arm_state_topic,
            self.on_arm_angles,
            state_qos,
        )
        self.sub_wireless = self.create_subscription(
            WirelessController,
            wireless_topic,
            self.on_wireless,
            wireless_qos,
        )
        self._startup_mode_timer = self.create_timer(
            self.STARTUP_MODE_DELAY_S,
            self.publish_startup_single_mode_command,
        )
        self._command_publish_timer = self.create_timer(
            1.0 / self.COMMAND_PUBLISH_HZ,
            self.on_command_publish_timer,
        )
        self._wireless_control_timer = self.create_timer(
            1.0 / self.WIRELESS_CONTROL_HZ,
            self.on_wireless_control_timer,
        )

        self.get_logger().info(
            "d1_drake_z_ref_node running: "
            f"{arm_state_topic} + {wireless_topic} -> {arm_command_topic}; "
            f"ry controls z_ref in [{self.z_ref_min_m:.2f}, {self.z_ref_max_m:.2f}] m "
            f"at {self.z_velocity_mps:.2f} m/s"
        )

    def on_arm_angles(self, msg: ArmState) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]
        self.latest_q_in = q_in.copy()

        if not self._q0_seeded:
            if time.monotonic() < self._q0_ready_at:
                return
            if not self.save_current_configuration_as_q0():
                return

    def on_wireless(self, msg: WirelessController) -> None:
        self._latest_right_stick_y = float(msg.ry)
        self._last_wireless_message_time = time.monotonic()

    def on_wireless_control_timer(self) -> None:
        now = time.monotonic()
        dt = now - self._last_wireless_control_time
        self._last_wireless_control_time = now
        if dt <= 0.0 or self.z_reference is None:
            return

        right_stick_y = self.get_deadzone_filtered_right_stick_y(now)
        if right_stick_y == 0.0:
            return

        updated_z_reference = self.clamp_z_reference(
            self.z_reference + right_stick_y * self.z_velocity_mps * dt
        )
        if abs(updated_z_reference - self.z_reference) <= 1e-6:
            return

        self.z_reference = updated_z_reference
        if (now - self._last_z_reference_log_time) >= self.Z_REFERENCE_LOG_INTERVAL_S:
            self._last_z_reference_log_time = now
            self.get_logger().info(f"Updated arm z reference to {self.z_reference:.4f} m")
        self.solve_and_publish_for_current_reference()

    def on_command_publish_timer(self) -> None:
        if (
            self.latest_q_in is None
            or self._pending_q_out_solver is None
            or self._pending_q_out is None
        ):
            return

        if not self.publish_arm_commands(
            self.latest_q_in.copy(),
            self._pending_q_out_solver,
            self._pending_q_out,
        ):
            self._pending_q_out_solver = None
            self._pending_q_out = None

    def solve_and_publish_for_current_reference(self) -> None:
        if self.latest_q_in is None or self.z_reference is None or self.q0_deg is None:
            return

        q_in = self.latest_q_in.copy()
        self._pending_q_out_solver = None
        self._pending_q_out = None
        target = self.build_up_down_target()

        try:
            q_out_solver = self.solver.solve_up_down(q_in, target, self.q0_deg)
            solved_pose = self.solver.get_end_effector_pose(q_out_solver)
            solved_z = float(solved_pose.translation()[2])
            z_error = solved_z - self.z_reference
            q_out = q_out_solver.copy()
            q_out[0] = -q_out[0]
            q_out[3] = -q_out[3]
            self.q_out = q_out
            self._pending_q_out_solver = q_out_solver
            self._pending_q_out = q_out
            now = time.monotonic()
            if (now - self._last_ik_solve_log_time) >= self.IK_SOLVE_LOG_INTERVAL_S:
                self._last_ik_solve_log_time = now
                self.get_logger().info(
                    "Manual Drake IK solved z target: "
                    f"target={self.z_reference:.4f} m, "
                    f"solved={solved_z:.4f} m, "
                    f"error={z_error:.4f} m"
                )
        except RuntimeError as exc:
            self.get_logger().warning(f"Drake IK solve failed: {exc}")

    def save_current_configuration_as_q0(self) -> bool:
        if self.latest_q_in is None:
            self.get_logger().warning(
                "Cannot save q0 arm configuration before receiving /arm/state"
            )
            return False

        self.q0_deg = self.latest_q_in.copy()
        q0_pose = self.solver.get_end_effector_pose(self.q0_deg)
        self.z_reference = self.clamp_z_reference(float(q0_pose.translation()[2]))
        self._q0_seeded = True
        self.get_logger().info(
            "Saved current arm configuration as q0: "
            + np.array2string(self.q0_deg, precision=2, separator=", ")
            + f"; z_ref={self.z_reference:.4f}"
        )
        return True

    def build_up_down_target(self) -> ArmTarget:
        q0_pose = self.solver.get_end_effector_pose(self.q0_deg)
        target = ArmTarget()
        target.command_type = ArmTarget.COMMAND_TYPE_END_EFFECTOR_POSE
        target.header.frame_id = self.solver.BASE_FRAME
        target.position_m = [
            float(q0_pose.translation()[0]),
            float(q0_pose.translation()[1]),
            float(self.z_reference),
        ]
        target.orientation_xyzw = [0.0, 0.0, 0.0, 0.0]
        return target

    def clamp_z_reference(self, z_reference: float) -> float:
        return min(self.z_ref_max_m, max(self.z_ref_min_m, z_reference))

    def get_deadzone_filtered_right_stick_y(self, now: float) -> float:
        if (now - self._last_wireless_message_time) > self.WIRELESS_INPUT_TIMEOUT_S:
            return 0.0
        right_stick_y = max(-1.0, min(1.0, self._latest_right_stick_y))
        if abs(right_stick_y) < self.ry_deadzone:
            return 0.0
        return right_stick_y

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
    ) -> bool:
        joints_to_update: list[int] = []

        for joint_index in self.SINGLE_JOINT_INDICES:
            if (
                abs(float(q_in[joint_index]) - float(q_out_solver[joint_index]))
                <= self.JOINT_PUBLISH_THRESHOLD_DEG
            ):
                continue
            joints_to_update.append(joint_index)

        if not joints_to_update:
            return False

        self.publish_group_joint_command(q_out)
        return True

    def publish_group_joint_command(self, q_out: np.ndarray) -> None:
        target_angles_deg = np.asarray(self.STARTUP_ANGLES_DEG, dtype=float).copy()
        for joint_index in self.SINGLE_JOINT_INDICES:
            target_angles_deg[joint_index] = float(q_out[joint_index])

        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 2,
            "data": {
                "mode": 0,
                "angle0": float(target_angles_deg[0]),
                "angle1": float(target_angles_deg[1]),
                "angle2": float(target_angles_deg[2]),
                "angle3": float(target_angles_deg[3]),
                "angle4": float(target_angles_deg[4]),
                "angle5": float(target_angles_deg[5]),
                "angle6": float(target_angles_deg[6]),
                "delay_ms": 0,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def schedule_single_mode_command(self, delay_s: float) -> None:
        timer_holder = {"timer": None}

        def _publish_when_ready() -> None:
            timer = timer_holder["timer"]
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
                if timer in self._pending_single_mode_timers:
                    self._pending_single_mode_timers.remove(timer)
            self.publish_single_mode_command()

        timer = self.create_timer(delay_s, _publish_when_ready)
        timer_holder["timer"] = timer
        self._pending_single_mode_timers.append(timer)

    def publish_single_mode_command(self) -> None:
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 4,
            "data": {
                "id": 0,
                "mode": 0,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_startup_arm_command(self) -> None:
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 2,
            "data": {
                "mode": 0,
                "angle0": self.STARTUP_ANGLES_DEG[0],
                "angle1": self.STARTUP_ANGLES_DEG[1],
                "angle2": self.STARTUP_ANGLES_DEG[2],
                "angle3": self.STARTUP_ANGLES_DEG[3],
                "angle4": self.STARTUP_ANGLES_DEG[4],
                "angle5": self.STARTUP_ANGLES_DEG[5],
                "angle6": self.STARTUP_ANGLES_DEG[6],
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_startup_single_mode_command(self) -> None:
        if self._startup_mode_timer is not None:
            self._startup_mode_timer.cancel()
            self.destroy_timer(self._startup_mode_timer)
            self._startup_mode_timer = None

        self.publish_startup_arm_command()
        self.schedule_single_mode_command(self.STARTUP_DAMPING_DELAY_S)
        self.get_logger().info(
            "Published startup arm command and scheduled joint0 single mode setup"
        )


def main() -> None:
    rclpy.init()
    node = D1DrakeZReferenceNode()
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

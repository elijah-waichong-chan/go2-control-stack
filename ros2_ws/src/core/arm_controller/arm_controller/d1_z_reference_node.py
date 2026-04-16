#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from go2_msgs.msg import ArmAngles
from unitree_arm.msg import ArmString
from unitree_go.msg import WirelessController

from arm_controller.d1_ik_solver import D1IKSolver


class D1ZReferenceNode(Node):
    COMMAND_THRESHOLD_DEG = 0.1
    COMMAND_COOLDOWN_S = 0.0
    COMMAND_PUBLISH_HZ = 10.0
    WIRELESS_CONTROL_HZ = 30.0
    WIRELESS_INPUT_TIMEOUT_S = 0.2
    IK_SOLVE_LOG_INTERVAL_S = 0.25
    Z_REFERENCE_LOG_INTERVAL_S = 0.25
    SINGLE_JOINT_INDICES = (1, 2, 4)
    COMMAND_SEQ = 4
    COMMAND_ADDRESS = 1
    COMMAND_FUNCODE_FULL_ARM_ANGLE = 2
    COMMAND_FUNCODE_SINGLE_MODE = 4
    COMMAND_DELAY_MS = 0
    COMMAND_MODE_BY_SERVO_ID = {
        0: 0,
    }
    STARTUP_ARM_COMMAND = {
        "seq": 4,
        "address": 1,
        "funcode": 2,
        "data": {
            "mode": 0,
            "angle0": 90,
            "angle1": -10,
            "angle2": 10,
            "angle3": 5,
            "angle4": -3,
            "angle5": -90,
            "angle6": -10,
        },
    }
    STARTUP_MODE_DELAY_S = 0.1
    STARTUP_DAMPING_DELAY_S = 3.0
    JOINT0_MODE_DELAY_S = 1.0

    def __init__(self) -> None:
        super().__init__("d1_z_reference_node")

        self.declare_parameter("arm_angles_topic", "/arm_angles")
        self.declare_parameter("arm_command_topic", "/arm_Command")
        self.declare_parameter("wireless_topic", "/wirelesscontroller")
        self.declare_parameter("z_step_m", 0.05)
        self.declare_parameter("z_velocity_mps", 0.20)
        self.declare_parameter("z_ref_min_m", -0.09)
        self.declare_parameter("z_ref_max_m", 0.56)
        self.declare_parameter("ry_deadzone", 0.10)

        arm_angles_topic = str(self.get_parameter("arm_angles_topic").value)
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

        self.solver = D1IKSolver()
        self.q_out: np.ndarray | None = None
        self.latest_q_in: np.ndarray | None = None
        self.z_reference: float | None = None
        self._pending_q_out_solver: np.ndarray | None = None
        self._pending_q_out: np.ndarray | None = None
        self._nominal_seeded = False
        self._nominal_seed_ready_at = time.monotonic() + self.STARTUP_DAMPING_DELAY_S
        self._last_command_time_by_joint: dict[int, float] = {}
        self._pending_single_mode_timers: list = []
        self._latest_right_stick_y = 0.0
        self._last_wireless_message_time = 0.0
        self._last_wireless_control_time = time.monotonic()
        self._last_ik_solve_log_time = 0.0
        self._last_z_reference_log_time = 0.0

        self.pub_arm_command = self.create_publisher(
            ArmString,
            arm_command_topic,
            pub_qos,
        )
        self.pub_arm_ik_debug = self.create_publisher(
            String,
            "/arm_ik_debug",
            pub_qos,
        )
        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            arm_angles_topic,
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
            "d1_z_reference_node running: "
            f"{arm_angles_topic} + {wireless_topic} -> {arm_command_topic}; "
            f"ry controls z_ref in [{self.z_ref_min_m:.2f}, {self.z_ref_max_m:.2f}] m "
            f"at {self.z_velocity_mps:.2f} m/s"
        )

    def on_arm_angles(self, msg: ArmAngles) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]
        self.latest_q_in = q_in.copy()

        if not self._nominal_seeded:
            if time.monotonic() < self._nominal_seed_ready_at:
                return
            if not self.save_current_configuration_as_nominal():
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
        if self.latest_q_in is None or self.z_reference is None:
            return

        q_in = self.latest_q_in.copy()
        self._pending_q_out_solver = None
        self._pending_q_out = None
        try:
            q_out_solver = self.solver.solve_with_z_reference(q_in, self.z_reference)
            solved_z = self.solver.get_end_effector_z(q_out_solver)
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
                    "Manual IK solved z target: "
                    f"target={self.z_reference:.4f} m, "
                    f"solved={solved_z:.4f} m, "
                    f"error={z_error:.4f} m"
                )
            self.publish_arm_ik_debug(q_out)
        except RuntimeError as exc:
            self.get_logger().warning(f"IK solve failed: {exc}")

    def save_current_configuration_as_nominal(self) -> bool:
        """Use the latest measured arm configuration as the nominal pose."""
        if self.latest_q_in is None:
            self.get_logger().warning(
                "Cannot save nominal arm configuration before receiving /arm_angles"
            )
            return False

        self.solver.save_current_configuration_as_nominal(self.latest_q_in)
        nominal_z_reference = float(self.solver.nominal_end_effector_z)
        self.z_reference = self.clamp_z_reference(nominal_z_reference)
        self._nominal_seeded = True
        self.get_logger().info(
            "Saved current arm configuration as nominal: "
            + np.array2string(self.latest_q_in, precision=2, separator=", ")
            + f"; z_ref={self.z_reference:.4f}"
        )
        return True

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
        now = time.monotonic()
        joints_to_update: list[int] = []

        for joint_index in self.SINGLE_JOINT_INDICES:
            last_command_time = self._last_command_time_by_joint.get(joint_index)
            if (
                last_command_time is not None
                and (now - last_command_time) < self.COMMAND_COOLDOWN_S
            ):
                continue
            if (
                abs(float(q_in[joint_index]) - float(q_out_solver[joint_index]))
                <= self.COMMAND_THRESHOLD_DEG
            ):
                continue
            joints_to_update.append(joint_index)

        if not joints_to_update:
            return False

        self.publish_group_joint_command(q_out)

        for joint_index in joints_to_update:
            self._last_command_time_by_joint[joint_index] = now

        return True

    def publish_group_joint_command(
        self,
        q_out: np.ndarray,
    ) -> None:
        startup_angles = self.STARTUP_ARM_COMMAND["data"]
        target_angles_deg = np.asarray(
            [
                startup_angles["angle0"],
                startup_angles["angle1"],
                startup_angles["angle2"],
                startup_angles["angle3"],
                startup_angles["angle4"],
                startup_angles["angle5"],
                startup_angles["angle6"],
            ],
            dtype=float,
        )

        for joint_index in self.SINGLE_JOINT_INDICES:
            target_angles_deg[joint_index] = float(q_out[joint_index])

        payload = {
            "seq": self.COMMAND_SEQ,
            "address": self.COMMAND_ADDRESS,
            "funcode": self.COMMAND_FUNCODE_FULL_ARM_ANGLE,
            "data": {
                "mode": 0,
                "angle0": float(target_angles_deg[0]),
                "angle1": float(target_angles_deg[1]),
                "angle2": float(target_angles_deg[2]),
                "angle3": float(target_angles_deg[3]),
                "angle4": float(target_angles_deg[4]),
                "angle5": float(target_angles_deg[5]),
                "angle6": float(target_angles_deg[6]),
                "delay_ms": self.COMMAND_DELAY_MS,
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
        for servo_id, mode_value in self.COMMAND_MODE_BY_SERVO_ID.items():
            payload = {
                "seq": self.COMMAND_SEQ,
                "address": self.COMMAND_ADDRESS,
                "funcode": self.COMMAND_FUNCODE_SINGLE_MODE,
                "data": {
                    "id": servo_id,
                    "mode": mode_value,
                },
            }
            msg = ArmString()
            msg.data = json.dumps(payload, separators=(",", ":"))
            self.pub_arm_command.publish(msg)

    def publish_startup_arm_command(self) -> None:
        msg = ArmString()
        msg.data = json.dumps(self.STARTUP_ARM_COMMAND, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_startup_single_mode_command(self) -> None:
        if self._startup_mode_timer is not None:
            self._startup_mode_timer.cancel()
            self.destroy_timer(self._startup_mode_timer)
            self._startup_mode_timer = None

        self.publish_startup_arm_command()
        self.schedule_single_mode_command(self.STARTUP_DAMPING_DELAY_S)
        self.get_logger().info(
            "Published startup arm command: "
            + json.dumps(self.STARTUP_ARM_COMMAND, separators=(",", ":"))
            + f"; scheduled startup single mode commands in {self.STARTUP_DAMPING_DELAY_S:.1f}s: "
            + ", ".join(
                json.dumps(
                    {
                        "seq": self.COMMAND_SEQ,
                        "address": self.COMMAND_ADDRESS,
                        "funcode": self.COMMAND_FUNCODE_SINGLE_MODE,
                        "data": {
                            "id": servo_id,
                            "mode": mode_value,
                        },
                    },
                    separators=(",", ":"),
                )
                for servo_id, mode_value in self.COMMAND_MODE_BY_SERVO_ID.items()
            )
        )

    def publish_arm_ik_debug(self, q_out: np.ndarray) -> None:
        msg = String()
        nominal_q_str = np.array2string(self.solver.nominal_q, precision=2, separator=", ")
        q_out_str = np.array2string(q_out, precision=1, separator=", ")
        z_ref_str = "nan" if self.z_reference is None else f"{self.z_reference:.4f}"
        msg.data = f"q0={nominal_q_str}\nz_ref={z_ref_str}\nq_out={q_out_str}"
        self.pub_arm_ik_debug.publish(msg)


def main() -> None:
    rclpy.init()
    node = D1ZReferenceNode()
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

#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from hq_pcot_msgs.msg import ArmState, ArmTarget
from std_msgs.msg import Int32
from unitree_arm.msg import ArmString

from arm_controller.d1_drake_ik_solver import D1DrakeIKSolver


class DrakeArmControllerNode(Node):
    JOINT_PUBLISH_THRESHOLD_DEG = 0.1
    SINGLE_JOINT_INDICES = (1, 2, 3, 4, 5)
    UP_DOWN_IK_PUBLISH_JOINT_INDICES = (1, 2, 4)
    FIXED_GRIPPER_ANGLE_DEG = -20.0
    STARTUP_ANGLES_DEG = (90.0, -10.0, 10.0, 5.0, -3.0, 0.0, FIXED_GRIPPER_ANGLE_DEG)
    FIXED_JOINT1_JOINT5_RANGE_DEG = (-5.0, 5.0)
    UP_DOWN_JOINT5_RANGE_DEG = (-85.0, 105.0)
    JOINT4_CURRENT_TRANSITION_THRESHOLD = 100.0
    JOINT0_TARGET_DEG = 90.0
    JOINT5_CURRENT_SWITCH_THRESHOLD = 30.0
    MODE_SWITCH_COOLDOWN_S = 3.0
    MODE_SWITCH_UP_DOWN_JOINT5_TARGET_DEG = -90.0
    MODE_SWITCH_FRONT_BACK_JOINT5_TARGET_DEG = 0.0
    STARTUP_MODE_DELAY_S = 0.1
    STARTUP_DAMPING_DELAY_S = 3.0
    JOINT0_TARGET_DELAY_MS = 500
    UP_DOWN_Z_REFERENCE_LOG_INTERVAL_S = 0.25
    UP_DOWN_IDLE_INTENT_LABEL = 0
    UP_DOWN_INCREASE_INTENT_LABEL = 5
    UP_DOWN_DECREASE_INTENT_LABEL = 6

    def __init__(self) -> None:
        super().__init__("drake_arm_controller")

        self.declare_parameter("up_down_z_velocity_mps", 0.05)
        self.declare_parameter("up_down_z_ref_min_m", -0.09)
        self.declare_parameter("up_down_z_ref_max_m", 0.56)
        self.declare_parameter("up_down_intent_topic", "/direction_intent/up_down")
        self.declare_parameter("up_down_idle_intent_label", self.UP_DOWN_IDLE_INTENT_LABEL)
        self.declare_parameter(
            "up_down_increase_intent_label", self.UP_DOWN_INCREASE_INTENT_LABEL
        )
        self.declare_parameter(
            "up_down_decrease_intent_label", self.UP_DOWN_DECREASE_INTENT_LABEL
        )
        self.declare_parameter("up_down_intent_timeout_s", 0.5)

        self.up_down_z_velocity_mps = abs(
            float(self.get_parameter("up_down_z_velocity_mps").value)
        )
        self.up_down_z_ref_min_m = float(self.get_parameter("up_down_z_ref_min_m").value)
        self.up_down_z_ref_max_m = float(self.get_parameter("up_down_z_ref_max_m").value)
        self.up_down_intent_topic = str(self.get_parameter("up_down_intent_topic").value)
        self.up_down_idle_intent_label = int(
            self.get_parameter("up_down_idle_intent_label").value
        )
        self.up_down_increase_intent_label = int(
            self.get_parameter("up_down_increase_intent_label").value
        )
        self.up_down_decrease_intent_label = int(
            self.get_parameter("up_down_decrease_intent_label").value
        )
        self.up_down_intent_timeout_s = max(
            0.0, float(self.get_parameter("up_down_intent_timeout_s").value)
        )
        if self.up_down_z_ref_min_m > self.up_down_z_ref_max_m:
            self.up_down_z_ref_min_m, self.up_down_z_ref_max_m = (
                self.up_down_z_ref_max_m,
                self.up_down_z_ref_min_m,
            )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.solver = D1DrakeIKSolver()
        self.latest_q_in: np.ndarray | None = None
        self.latest_gripper_angle_deg: float | None = None
        self.q0_deg: np.ndarray | None = None
        self.q_out: np.ndarray | None = None
        self._active_solver_mode: str | None = None
        self._q0_ready_at = float("inf")
        self._q0_seeded = False
        self._joint0_is_damped = True
        self._joint5_current_switch_latched = False
        self._next_mode_switch_time = 0.0
        self._up_down_z_reference: float | None = None
        self._up_down_x_reference: float | None = None
        self._up_down_y_reference: float | None = None
        self._last_up_down_control_time: float | None = None
        self._last_up_down_z_reference_log_time = 0.0
        self._latest_up_down_intent: int | None = None
        self._last_up_down_intent_time: float | None = None

        self.pub_arm_command = self.create_publisher(ArmString, "/arm_Command", qos)
        self.sub_arm_state = self.create_subscription(
            ArmState,
            "/arm/state",
            self.on_arm_state,
            qos,
        )
        self.sub_up_down_intent = self.create_subscription(
            Int32,
            self.up_down_intent_topic,
            self.on_up_down_intent,
            10,
        )
        self._startup_mode_timer = self.create_timer(
            self.STARTUP_MODE_DELAY_S,
            self.publish_startup_single_mode_command,
        )

        self.get_logger().info(
            "drake_arm_controller listening on /arm/state and "
            f"{self.up_down_intent_topic}"
        )

    def on_arm_state(self, msg: ArmState) -> None:
        if len(msg.angle_deg) < 6:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        if len(msg.angle_deg) > 6:
            self.latest_gripper_angle_deg = float(msg.angle_deg[6])

        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]
        self.latest_q_in = q_in.copy()

        if not self._q0_seeded:
            if time.monotonic() < self._q0_ready_at:
                return
            if not self.save_current_configuration_as_q0():
                return
            self._q0_seeded = True

        try:
            joint4_current = float(msg.current[4]) if len(msg.current) > 4 else 0.0
            joint5_current = float(msg.current[5]) if len(msg.current) > 5 else 0.0
            self.handle_joint0_current_control(joint4_current)
            q_out_solver = self.solve_for_current_mode(q_in, joint5_current)
            if q_out_solver is None:
                return

            q_out = q_out_solver.copy()
            q_out[0] = -q_out[0]
            q_out[3] = -q_out[3]
            self.q_out = q_out
            self.publish_arm_commands(q_in, q_out_solver, q_out)
        except RuntimeError as exc:
            self.get_logger().warning(f"Drake IK solve failed: {exc}")

    def on_up_down_intent(self, msg: Int32) -> None:
        self._latest_up_down_intent = int(msg.data)
        self._last_up_down_intent_time = time.monotonic()

    def save_current_configuration_as_q0(self) -> bool:
        if self.latest_q_in is None:
            self.get_logger().warning("Cannot save q0 before receiving /arm/state")
            return False

        self.q0_deg = self.latest_q_in.copy()
        self.get_logger().info(
            "Saved current arm configuration as Drake q0: "
            + np.array2string(self.q0_deg, precision=2, separator=", ")
        )
        return True

    def capture_up_down_references(self) -> None:
        if self.q0_deg is None:
            return

        q0_pose = self.solver.get_end_effector_pose(self.q0_deg)
        q0_translation = np.asarray(q0_pose.translation(), dtype=float)
        self._up_down_x_reference = float(q0_translation[0])
        self._up_down_y_reference = float(q0_translation[1])
        self._up_down_z_reference = self.clamp_up_down_z_reference(float(q0_translation[2]))
        self._last_up_down_control_time = time.monotonic()
        self.get_logger().info(
            "Captured up_down references: "
            f"x_ref={self._up_down_x_reference:.4f} m, "
            f"y_ref={self._up_down_y_reference:.4f} m, "
            f"z_ref={self._up_down_z_reference:.4f} m"
        )

    def clear_up_down_references(self) -> None:
        self._up_down_x_reference = None
        self._up_down_y_reference = None
        self._up_down_z_reference = None
        self._last_up_down_control_time = None

    def clamp_up_down_z_reference(self, z_reference: float) -> float:
        return min(self.up_down_z_ref_max_m, max(self.up_down_z_ref_min_m, z_reference))

    def update_up_down_z_reference_from_intent(self, now: float) -> None:
        if self._up_down_z_reference is None:
            return
        if self._last_up_down_control_time is None:
            self._last_up_down_control_time = now
            return

        dt = now - self._last_up_down_control_time
        self._last_up_down_control_time = now
        if dt <= 0.0 or now < self._next_mode_switch_time or self.up_down_z_velocity_mps <= 0.0:
            return
        if self._latest_up_down_intent is None or self._last_up_down_intent_time is None:
            return
        if (now - self._last_up_down_intent_time) > self.up_down_intent_timeout_s:
            return

        direction = 0.0
        if self._latest_up_down_intent == self.up_down_increase_intent_label:
            direction = 1.0
        elif self._latest_up_down_intent == self.up_down_decrease_intent_label:
            direction = -1.0
        elif self._latest_up_down_intent == self.up_down_idle_intent_label:
            direction = 0.0
        if direction == 0.0:
            return

        updated_z_reference = self.clamp_up_down_z_reference(
            self._up_down_z_reference + direction * self.up_down_z_velocity_mps * dt
        )
        if abs(updated_z_reference - self._up_down_z_reference) <= 1e-6:
            return

        self._up_down_z_reference = updated_z_reference
        if (now - self._last_up_down_z_reference_log_time) >= self.UP_DOWN_Z_REFERENCE_LOG_INTERVAL_S:
            self._last_up_down_z_reference_log_time = now
            self.get_logger().info(
                "Updated up_down z reference from intent: "
                f"intent={self._latest_up_down_intent}, "
                f"z_ref={self._up_down_z_reference:.4f} m"
            )

    def solve_for_current_mode(
        self,
        q_in: np.ndarray,
        joint5_current: float,
    ) -> np.ndarray | None:
        if self.q0_deg is None:
            return None

        current_mode = self._active_solver_mode
        if current_mode is None:
            current_mode = self._infer_mode_from_joint5_angle(float(q_in[5]))
        if current_mode is None:
            current_mode = "front_back_mode"

        next_solver_mode = current_mode
        mode_switch_joint5_target: float | None = None
        now = time.monotonic()

        if joint5_current > self.JOINT5_CURRENT_SWITCH_THRESHOLD:
            if not self._joint5_current_switch_latched and now >= self._next_mode_switch_time:
                if current_mode == "front_back_mode":
                    next_solver_mode = "up_down_mode"
                    mode_switch_joint5_target = self.MODE_SWITCH_UP_DOWN_JOINT5_TARGET_DEG
                else:
                    next_solver_mode = "front_back_mode"
                    mode_switch_joint5_target = self.MODE_SWITCH_FRONT_BACK_JOINT5_TARGET_DEG
                self._joint5_current_switch_latched = True
        else:
            self._joint5_current_switch_latched = False

        if current_mode != next_solver_mode:
            if mode_switch_joint5_target is not None:
                self.publish_single_joint_command(5, mode_switch_joint5_target)
            self.save_current_configuration_as_q0()
            if next_solver_mode == "up_down_mode":
                self.capture_up_down_references()
            else:
                self.clear_up_down_references()
            self._next_mode_switch_time = now + self.MODE_SWITCH_COOLDOWN_S

        self._active_solver_mode = next_solver_mode
        if next_solver_mode == "front_back_mode":
            return self.solver.solve_front_back(
                q_in,
                self.build_front_back_target(q_in),
                self.q0_deg,
            )

        if (
            self._up_down_x_reference is None
            or self._up_down_y_reference is None
            or self._up_down_z_reference is None
        ):
            self.capture_up_down_references()
        self.update_up_down_z_reference_from_intent(now)
        return self.solver.solve_up_down(
            q_in,
            self.build_up_down_target(),
            self.q0_deg,
        )

    def build_front_back_target(self, q_in: np.ndarray) -> ArmTarget:
        q0_pose = self.solver.get_end_effector_pose(self.q0_deg)
        current_pose = self.solver.get_end_effector_pose(q_in)
        target = ArmTarget()
        target.command_type = ArmTarget.COMMAND_TYPE_END_EFFECTOR_POSE
        target.header.frame_id = self.solver.BASE_FRAME
        target.position_m = [
            float(current_pose.translation()[0]),
            float(q0_pose.translation()[1]),
            float(q0_pose.translation()[2]),
        ]
        target.orientation_xyzw = [0.0, 0.0, 0.0, 0.0]
        return target

    def build_up_down_target(self) -> ArmTarget:
        target = ArmTarget()
        target.command_type = ArmTarget.COMMAND_TYPE_END_EFFECTOR_POSE
        target.header.frame_id = self.solver.BASE_FRAME
        target.position_m = [
            float(self._up_down_x_reference),
            float(self._up_down_y_reference),
            float(self._up_down_z_reference),
        ]
        target.orientation_xyzw = [0.0, 0.0, 0.0, 0.0]
        return target

    def _infer_mode_from_joint5_angle(self, joint5_angle: float) -> str | None:
        if self._angle_in_range(joint5_angle, self.FIXED_JOINT1_JOINT5_RANGE_DEG):
            return "front_back_mode"
        if self._angle_in_range(joint5_angle, self.UP_DOWN_JOINT5_RANGE_DEG):
            return "up_down_mode"
        return None

    def _angle_in_range(
        self,
        angle_deg: float,
        bounds_deg: tuple[float, float],
    ) -> bool:
        return min(bounds_deg) <= angle_deg <= max(bounds_deg)

    def handle_joint0_current_control(self, joint4_current: float) -> None:
        if self._joint0_is_damped:
            if joint4_current < self.JOINT4_CURRENT_TRANSITION_THRESHOLD:
                self.publish_single_joint_command(
                    0,
                    self.JOINT0_TARGET_DEG,
                    delay_ms=self.JOINT0_TARGET_DELAY_MS,
                )
                self._joint0_is_damped = False
            return

        if joint4_current > self.JOINT4_CURRENT_TRANSITION_THRESHOLD:
            self.publish_single_mode_command_for_servo(0, 0)
            self._joint0_is_damped = True

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
    ) -> None:
        joints_to_update: list[int] = []
        joints_to_consider = self.SINGLE_JOINT_INDICES
        if self._active_solver_mode == "up_down_mode":
            joints_to_consider = self.UP_DOWN_IK_PUBLISH_JOINT_INDICES

        for joint_index in joints_to_consider:
            if abs(float(q_in[joint_index]) - float(q_out_solver[joint_index])) <= self.JOINT_PUBLISH_THRESHOLD_DEG:
                continue
            joints_to_update.append(joint_index)

        gripper_needs_update = (
            self.latest_gripper_angle_deg is None
            or abs(self.latest_gripper_angle_deg - self.FIXED_GRIPPER_ANGLE_DEG)
            > self.JOINT_PUBLISH_THRESHOLD_DEG
        )
        if not joints_to_update and not gripper_needs_update:
            return

        if 4 in joints_to_update:
            joints_to_update = [4] + [joint_index for joint_index in joints_to_update if joint_index != 4]

        for joint_index in joints_to_update:
            self.publish_single_joint_command(joint_index, float(q_out[joint_index]))

        if gripper_needs_update:
            self.publish_single_joint_command(6, self.FIXED_GRIPPER_ANGLE_DEG)

    def publish_single_joint_command(
        self,
        servo_id: int,
        angle_deg: float,
        *,
        delay_ms: int | None = None,
    ) -> None:
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 1,
            "data": {
                "id": servo_id,
                "angle": float(angle_deg),
                "delay_ms": 0 if delay_ms is None else int(delay_ms),
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_single_mode_command_for_servo(self, servo_id: int, mode_value: int) -> None:
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 4,
            "data": {
                "id": servo_id,
                "mode": mode_value,
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
        self.publish_single_mode_command_for_servo(0, 0)
        self._q0_ready_at = time.monotonic() + self.STARTUP_DAMPING_DELAY_S
        self.get_logger().info(
            "Published startup arm command and joint0 single-servo mode setup"
        )


def main() -> None:
    rclpy.init()
    node = DrakeArmControllerNode()
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

#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from go2_msgs.msg import ArmAngles
from std_msgs.msg import Int32
from unitree_arm.msg import ArmString

from arm_controller.d1_ik_solver import D1IKSolver


class ArmControllerNode(Node):
    COMMAND_THRESHOLD_DEG = 0.1
    COMMAND_COOLDOWN_S = 0
    SINGLE_JOINT_INDICES = (1, 2, 3, 4, 5)
    FIXED_GRIPPER_SERVO_ID = 6
    FIXED_GRIPPER_ANGLE_DEG = -20.0
    FIXED_JOINT1_JOINT5_RANGE_DEG = (-5.0, 5.0)
    UP_DOWN_JOINT5_RANGE_DEG = (-85.0, 105.0)
    UP_DOWN_IK_PUBLISH_JOINT_INDICES = (1, 2, 4)
    JOINT4_CURRENT_TRANSITION_THRESHOLD = 100.0
    JOINT0_TARGET_DEG = 90.0
    JOINT5_CURRENT_SWITCH_THRESHOLD = 30.0
    MODE_SWITCH_COOLDOWN_S = 3.0
    MODE_SWITCH_UP_DOWN_JOINT5_TARGET_DEG = -90.0
    MODE_SWITCH_FRONT_BACK_JOINT5_TARGET_DEG = 0.0
    FRONT_BACK_MODE = "front_back_mode"
    UP_DOWN_MODE = "up_down_mode"
    UP_DOWN_IDLE_INTENT_LABEL = 0
    UP_DOWN_INCREASE_INTENT_LABEL = 5
    UP_DOWN_DECREASE_INTENT_LABEL = 6
    COMMAND_SEQ = 4
    COMMAND_ADDRESS = 1
    COMMAND_FUNCODE_SINGLE_SERVO_ANGLE = 1
    COMMAND_FUNCODE_SINGLE_MODE = 4
    COMMAND_DELAY_MS = 0
    JOINT0_TARGET_DELAY_MS = 500
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
            "angle5": 0,
            "angle6": FIXED_GRIPPER_ANGLE_DEG,
        },
    }
    STARTUP_MODE_DELAY_S = 0.1
    STARTUP_DAMPING_DELAY_S = 3.0
    JOINT0_MODE_DELAY_S = 1.0
    UP_DOWN_Z_REFERENCE_LOG_INTERVAL_S = 0.25

    def __init__(self) -> None:
        super().__init__("arm_controller")

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
        self.up_down_z_ref_min_m = float(
            self.get_parameter("up_down_z_ref_min_m").value
        )
        self.up_down_z_ref_max_m = float(
            self.get_parameter("up_down_z_ref_max_m").value
        )
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

        self.solver = D1IKSolver()
        self.q_out: np.ndarray | None = None
        self.latest_q_in: np.ndarray | None = None
        self.latest_gripper_angle_deg: float | None = None
        self.x_vel = 0.0
        self._active_solver_mode: str | None = None
        self._nominal_seeded = False
        self._nominal_seed_ready_at = time.monotonic() + self.STARTUP_DAMPING_DELAY_S
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
        self._last_command_time_by_joint: dict[int, float] = {}
        self._pending_single_mode_timers: list = []

        self.pub_arm_command = self.create_publisher(
            ArmString,
            "/arm_Command",
            qos,
        )
        self.sub_arm_angles = self.create_subscription(
            ArmAngles,
            "/arm_angles",
            self.on_arm_angles,
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
            "arm_controller listening on /arm_angles and "
            f"{self.up_down_intent_topic}"
        )

    def on_arm_angles(self, msg: ArmAngles) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        if len(msg.angle_deg) > self.FIXED_GRIPPER_SERVO_ID:
            self.latest_gripper_angle_deg = float(msg.angle_deg[self.FIXED_GRIPPER_SERVO_ID])
        # self.handle_joint0_limits(current_q)
        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]
        self.latest_q_in = q_in.copy()
        if not self._nominal_seeded:
            if time.monotonic() < self._nominal_seed_ready_at:
                return
            self.save_current_configuration_as_nominal()
            self._nominal_seeded = True

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
            self.get_logger().warning(f"IK solve failed: {exc}")

    def on_up_down_intent(self, msg: Int32) -> None:
        self._latest_up_down_intent = int(msg.data)
        self._last_up_down_intent_time = time.monotonic()

    def save_current_configuration_as_nominal(self) -> bool:
        """Replace the solver nominal state with the latest measured arm state."""
        if self.latest_q_in is None:
            self.get_logger().warning(
                "Cannot save nominal arm configuration before receiving /arm_angles"
            )
            return False

        self.solver.save_current_configuration_as_nominal(self.latest_q_in)
        self.get_logger().info(
            "Saved current arm configuration as IK nominal q0: "
            + np.array2string(self.latest_q_in, precision=2, separator=", ")
        )
        return True

    def capture_up_down_references(self) -> None:
        self._up_down_x_reference = float(self.solver.nominal_end_effector_x)
        self._up_down_y_reference = float(self.solver.nominal_end_effector_y)
        self._up_down_z_reference = self.clamp_up_down_z_reference(
            float(self.solver.nominal_end_effector_z)
        )
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
        return min(
            self.up_down_z_ref_max_m,
            max(self.up_down_z_ref_min_m, z_reference),
        )

    def update_up_down_z_reference_from_intent(self, now: float) -> None:
        if self._up_down_z_reference is None:
            return

        if self._last_up_down_control_time is None:
            self._last_up_down_control_time = now
            return

        dt = now - self._last_up_down_control_time
        self._last_up_down_control_time = now
        if dt <= 0.0:
            return
        if now < self._next_mode_switch_time:
            return
        if self.up_down_z_velocity_mps <= 0.0:
            return

        if (
            self._latest_up_down_intent is None
            or self._last_up_down_intent_time is None
            or (now - self._last_up_down_intent_time) > self.up_down_intent_timeout_s
        ):
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
        if (
            now - self._last_up_down_z_reference_log_time
        ) >= self.UP_DOWN_Z_REFERENCE_LOG_INTERVAL_S:
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
        """Dispatch to the active solver mode based on joint-5 overcurrent events."""
        current_mode = self._active_solver_mode
        if current_mode is None:
            current_mode = self._infer_mode_from_joint5_angle(float(q_in[5]))
        if current_mode is None:
            current_mode = self.FRONT_BACK_MODE

        next_solver_mode = current_mode
        mode_switch_joint5_target: float | None = None
        now = time.monotonic()

        if joint5_current > self.JOINT5_CURRENT_SWITCH_THRESHOLD:
            if (
                not self._joint5_current_switch_latched
                and now >= self._next_mode_switch_time
            ):
                if current_mode == self.FRONT_BACK_MODE:
                    next_solver_mode = self.UP_DOWN_MODE
                    mode_switch_joint5_target = self.MODE_SWITCH_UP_DOWN_JOINT5_TARGET_DEG
                else:
                    next_solver_mode = self.FRONT_BACK_MODE
                    mode_switch_joint5_target = self.MODE_SWITCH_FRONT_BACK_JOINT5_TARGET_DEG
                self._joint5_current_switch_latched = True
        else:
            self._joint5_current_switch_latched = False

        if current_mode != next_solver_mode:
            if mode_switch_joint5_target is not None:
                self.publish_single_joint_command(
                    5,
                    mode_switch_joint5_target,
                )
            self.save_current_configuration_as_nominal()
            if next_solver_mode == self.UP_DOWN_MODE:
                self.capture_up_down_references()
            else:
                self.clear_up_down_references()
            self._next_mode_switch_time = now + self.MODE_SWITCH_COOLDOWN_S

        self._active_solver_mode = next_solver_mode
        if next_solver_mode == self.FRONT_BACK_MODE:
            return self.solver.solve_with_fixed_joint1(q_in)
        if next_solver_mode == self.UP_DOWN_MODE:
            if (
                self._up_down_x_reference is None
                or self._up_down_y_reference is None
                or self._up_down_z_reference is None
            ):
                self.capture_up_down_references()
            self.update_up_down_z_reference_from_intent(now)
            return self.solver.solve_up_down(
                q_in,
                self._up_down_z_reference,
                x_reference=self._up_down_x_reference,
                y_reference=self._up_down_y_reference,
            )
        return None

    def _infer_mode_from_joint5_angle(self, joint5_angle: float) -> str | None:
        """Infer a startup mode from the current joint-5 angle."""
        if self._angle_in_range(joint5_angle, self.FIXED_JOINT1_JOINT5_RANGE_DEG):
            return self.FRONT_BACK_MODE
        if self._angle_in_range(joint5_angle, self.UP_DOWN_JOINT5_RANGE_DEG):
            return self.UP_DOWN_MODE
        return None

    def _angle_in_range(
        self,
        angle_deg: float,
        bounds_deg: tuple[float, float],
    ) -> bool:
        lower_deg = min(bounds_deg)
        upper_deg = max(bounds_deg)
        return lower_deg <= angle_deg <= upper_deg

    def handle_joint0_limits(self, current_q: np.ndarray) -> None:
        joint0_angle = float(current_q[0])

        if -80.0 < joint0_angle < 0.0:
            self.publish_single_joint_command(0, -82.0)
        elif -180.0 < joint0_angle < -100.0:
            self.publish_single_joint_command(0, -98.0)
        else:
            self.x_vel = 0.0

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
            self.publish_single_mode_command_for_servo(
                0,
                self.COMMAND_MODE_BY_SERVO_ID[0],
            )
            self._joint0_is_damped = True

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
    ) -> None:
        now = time.monotonic()
        joints_to_update: list[int] = []
        gripper_needs_update = False
        joints_to_consider = self.SINGLE_JOINT_INDICES
        if self._active_solver_mode == self.UP_DOWN_MODE:
            joints_to_consider = self.UP_DOWN_IK_PUBLISH_JOINT_INDICES

        for joint_index in joints_to_consider:
            last_command_time = self._last_command_time_by_joint.get(joint_index)
            if (
                last_command_time is not None
                and (now - last_command_time) < self.COMMAND_COOLDOWN_S
            ):
                continue
            if abs(float(q_in[joint_index]) - float(q_out_solver[joint_index])) <= self.COMMAND_THRESHOLD_DEG:
                continue
            joints_to_update.append(joint_index)

        last_gripper_command_time = self._last_command_time_by_joint.get(
            self.FIXED_GRIPPER_SERVO_ID
        )
        if (
            last_gripper_command_time is None
            or (now - last_gripper_command_time) >= self.COMMAND_COOLDOWN_S
        ):
            if (
                self.latest_gripper_angle_deg is None
                or abs(self.latest_gripper_angle_deg - self.FIXED_GRIPPER_ANGLE_DEG)
                > self.COMMAND_THRESHOLD_DEG
            ):
                gripper_needs_update = True

        if not joints_to_update and not gripper_needs_update:
            return

        joints_to_publish = joints_to_update
        if 4 in joints_to_publish:
            joints_to_publish = [4] + [
                joint_index for joint_index in joints_to_publish if joint_index != 4
            ]
        if not joints_to_publish:
            if gripper_needs_update:
                self.publish_single_joint_command(
                    self.FIXED_GRIPPER_SERVO_ID,
                    self.FIXED_GRIPPER_ANGLE_DEG,
                )
                self._last_command_time_by_joint[self.FIXED_GRIPPER_SERVO_ID] = now
            return

        for joint_index in joints_to_publish:
            self.publish_single_joint_command(joint_index, float(q_out[joint_index]))

        if gripper_needs_update:
            self.publish_single_joint_command(
                self.FIXED_GRIPPER_SERVO_ID,
                self.FIXED_GRIPPER_ANGLE_DEG,
            )
            self._last_command_time_by_joint[self.FIXED_GRIPPER_SERVO_ID] = now

        for joint_index in joints_to_publish:
            self._last_command_time_by_joint[joint_index] = now

    def publish_single_joint_command(
        self,
        servo_id: int,
        angle_deg: float,
        *,
        delay_ms: int | None = None,
    ) -> None:
        payload = {
            "seq": self.COMMAND_SEQ,
            "address": self.COMMAND_ADDRESS,
            "funcode": self.COMMAND_FUNCODE_SINGLE_SERVO_ANGLE,
            "data": {
                "id": servo_id,
                "angle": float(angle_deg),
                "delay_ms": self.COMMAND_DELAY_MS if delay_ms is None else int(delay_ms),
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_command.publish(msg)

    def publish_single_mode_command_for_servo(
        self,
        servo_id: int,
        mode_value: int,
    ) -> None:
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
            self.publish_single_mode_command_for_servo(servo_id, mode_value)

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

def main() -> None:
    rclpy.init()
    node = ArmControllerNode()
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

#!/usr/bin/env python3

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from hq_pcot_msgs.msg import ArmState
from std_msgs.msg import Int32
from unitree_arm.msg import ArmString
from unitree_go.msg import WirelessController

from arm_controller.d1_ik_solver import D1IKSolver


class ArmControllerNode(Node):
    GRIPPER_CLOSED_ANGLE_DEG = -15.0
    GRIPPER_OPEN_ANGLE_DEG = 50.0

    FRONT_BACK_ARM_POSE = [90, -10, 10, -85, -3, -90]
    FRONT_BACK_IK_PUBLISH_JOINT_INDICES = (1, 2, 3, 4, 5)
    FRONT_BACK_MODE_RANGE_DEG = (-85.0, -105.0)
    FRONT_BACK_IK_UPDATE_JOINT0_THRESHOLD = 1.0
    FRONT_BACK_JOINT4_CURRENT_THRESHOLD = 100.0

    UP_DOWN_ARM_POSE = [None, None, None, 5.0, None, -90]
    UP_DOWN_IK_PUBLISH_JOINT_INDICES = (1, 2, 4)
    UP_DOWN_IDLE_INTENT_LABEL = 0
    UP_DOWN_INCREASE_INTENT_LABEL = 5
    UP_DOWN_DECREASE_INTENT_LABEL = 6
    UP_DOWN_MODE_RANGE_DEG = (-5.0, 5.0)
    UP_DOWN_Z_VELOCITY_MPS = 0.02
    UP_DOWN_Z_REF_MIN_M = -0.10
    UP_DOWN_Z_REF_MAX_M = 0.50
    UP_DOWN_INTENT_TOPIC = "/direction_intent/up_down"

    COMMAND_THRESHOLD_DEG = 0.1
    MODE_SWITCH_JOINT5_CURRENT_THRESHOLD = 30.0
    MOTION_SETTLE_DELAY_S = 2.0

    def __init__(self) -> None:
        super().__init__("arm_controller")

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.solver = D1IKSolver()
        self.latest_gripper_angle_deg: float | None = None
        self._gripper_target_angle_deg = self.GRIPPER_CLOSED_ANGLE_DEG
        self._active_solver_mode: str | None = None
        self._nominal_seeded = False
        self._nominal_seed_ready_at = time.monotonic() + self.MOTION_SETTLE_DELAY_S
        self._pending_nominal_seed_mode: str | None = None
        self._last_wireless_keys = 0
        self._joint0_is_damped = True
        self._joint5_current_switch_latched = False
        self._next_mode_switch_time = 0.0
        self._up_down_z_reference: float | None = None
        self._up_down_x_reference: float | None = None
        self._up_down_y_reference: float | None = None
        self._last_up_down_control_time: float | None = None
        self._latest_up_down_intent: int | None = None
        self._last_up_down_intent_time: float | None = None
        self._last_received_joint0_deg: float | None = None

        self.pub_arm_json = self.create_publisher(
            ArmString,
            "/arm_Command",
            qos,
        )
        self.sub_arm_angles = self.create_subscription(
            ArmState,
            "/arm/state",
            self.on_arm_angles,
            qos,
        )
        wireless_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.sub_wireless = self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self.on_wireless,
            wireless_qos,
        )
        self.sub_up_down_intent = self.create_subscription(
            Int32,
            self.UP_DOWN_INTENT_TOPIC,
            self.on_up_down_intent,
            10,
        )
        self.publish_arm_pose_target(self.FRONT_BACK_ARM_POSE)
        self.publish_gripper_target(self.GRIPPER_CLOSED_ANGLE_DEG)
        time.sleep(self.MOTION_SETTLE_DELAY_S)
        self.publish_joint0_damping_command()

    def on_arm_angles(self, msg: ArmState) -> None:
        if len(msg.angle_deg) == 0:
            return

        current_q = np.asarray(msg.angle_deg[:6], dtype=float)
        current_joint0_deg = float(current_q[0])
        previous_joint0_deg = self._last_received_joint0_deg
        self._last_received_joint0_deg = current_joint0_deg
        if len(msg.angle_deg) > 6:
            self.latest_gripper_angle_deg = float(msg.angle_deg[6])

        q_in = current_q.copy()
        q_in[0] = -q_in[0]
        q_in[3] = -q_in[3]

        if not self._nominal_seeded:
            if time.monotonic() < self._nominal_seed_ready_at:
                return
            self._finalize_nominal_seed(q_in)

        joint4_current = float(msg.current[4]) if len(msg.current) > 4 else 0.0
        joint5_current = float(msg.current[5]) if len(msg.current) > 5 else 0.0

        self.handle_joint0_current_control(joint4_current)
        self.update_solver_mode_from_joint5_current(q_in, joint5_current)

        if (
            self._active_solver_mode != "up_down_mode"
            and previous_joint0_deg is not None
            and abs(current_joint0_deg - previous_joint0_deg) < self.FRONT_BACK_IK_UPDATE_JOINT0_THRESHOLD
        ):
            return

        try:
            q_out_solver = self.solve_ik(q_in)
            if q_out_solver is None:
                return
            q_out = q_out_solver.copy()
            q_out[0] = -q_out[0]
            q_out[3] = -q_out[3]
            self.publish_arm_commands(q_in, q_out_solver, q_out)
        except RuntimeError:
            return

    def on_up_down_intent(self, msg: Int32) -> None:
        self._latest_up_down_intent = int(msg.data)
        self._last_up_down_intent_time = time.monotonic()

    def on_wireless(self, msg: WirelessController) -> None:
        current_keys = int(msg.keys)
        previous_keys = self._last_wireless_keys
        self._last_wireless_keys = current_keys

        x_pressed = bool(current_keys & (1 << 10)) and not bool(previous_keys & (1 << 10))
        b_pressed = bool(current_keys & (1 << 9)) and not bool(previous_keys & (1 << 9))

        if x_pressed:
            self.publish_gripper_target(self.GRIPPER_OPEN_ANGLE_DEG)
        elif b_pressed:
            self.publish_gripper_target(self.GRIPPER_CLOSED_ANGLE_DEG)

    def _finalize_nominal_seed(self, q_in: np.ndarray) -> None:
        self.solver.save_current_configuration_as_q0(q_in)

        if self._pending_nominal_seed_mode == "up_down_mode":
            self.capture_up_down_references()
        elif self._pending_nominal_seed_mode == "front_back_mode":
            self._up_down_x_reference = None
            self._up_down_y_reference = None
            self._up_down_z_reference = None
            self._last_up_down_control_time = None

        self._pending_nominal_seed_mode = None
        self._nominal_seeded = True

    def capture_up_down_references(self) -> None:
        self._up_down_x_reference = float(self.solver.nominal_end_effector_x)
        self._up_down_y_reference = float(self.solver.nominal_end_effector_y)
        self._up_down_z_reference = self.clamp_up_down_z_reference(
            float(self.solver.nominal_end_effector_z)
        )
        self._last_up_down_control_time = time.monotonic()

    def clamp_up_down_z_reference(self, z_reference: float) -> float:
        return min(
            self.UP_DOWN_Z_REF_MAX_M,
            max(self.UP_DOWN_Z_REF_MIN_M, z_reference),
        )

    def update_up_down_z_reference_from_intent(self, now: float) -> bool:
        dt = now - self._last_up_down_control_time
        self._last_up_down_control_time = now
        if dt <= 0.0:
            return False
        if now < self._next_mode_switch_time:
            return False

        if (
            self._latest_up_down_intent is None
            or self._last_up_down_intent_time is None
            or (now - self._last_up_down_intent_time) > 0.5
        ):
            return False

        direction = 0.0
        if self._latest_up_down_intent == self.UP_DOWN_INCREASE_INTENT_LABEL:
            direction = 1.0
        elif self._latest_up_down_intent == self.UP_DOWN_DECREASE_INTENT_LABEL:
            direction = -1.0
        elif self._latest_up_down_intent == self.UP_DOWN_IDLE_INTENT_LABEL:
            direction = 0.0

        if direction == 0.0:
            return False

        updated_z_reference = self.clamp_up_down_z_reference(
            self._up_down_z_reference + direction * self.UP_DOWN_Z_VELOCITY_MPS * dt
        )
        if abs(updated_z_reference - self._up_down_z_reference) <= 1e-6:
            return False

        self._up_down_z_reference = updated_z_reference
        return True

    def update_solver_mode_from_joint5_current(
        self,
        q_in: np.ndarray,
        joint5_current: float,
    ) -> None:
        """Check joint-5 overcurrent on every state callback and update active mode."""
        current_mode = self._active_solver_mode
        if current_mode is None:
            joint5_angle = float(q_in[5])
            if min(self.FRONT_BACK_MODE_RANGE_DEG) <= joint5_angle <= max(
                self.FRONT_BACK_MODE_RANGE_DEG
            ):
                current_mode = "front_back_mode"
            elif min(self.UP_DOWN_MODE_RANGE_DEG) <= joint5_angle <= max(
                self.UP_DOWN_MODE_RANGE_DEG
            ):
                current_mode = "up_down_mode"
        if current_mode is None:
            current_mode = "front_back_mode"

        next_solver_mode = current_mode
        now = time.monotonic()

        if joint5_current > self.MODE_SWITCH_JOINT5_CURRENT_THRESHOLD:
            if (
                not self._joint5_current_switch_latched
                and now >= self._next_mode_switch_time
            ):
                if current_mode == "front_back_mode":
                    next_solver_mode = "up_down_mode"
                else:
                    next_solver_mode = "front_back_mode"
                self._joint5_current_switch_latched = True
        else:
            self._joint5_current_switch_latched = False

        if current_mode != next_solver_mode:
            if next_solver_mode == "up_down_mode":
                self.publish_arm_pose_target(
                    self.UP_DOWN_ARM_POSE
                )
            elif next_solver_mode == "front_back_mode":
                self.publish_arm_pose_target(
                    self.FRONT_BACK_ARM_POSE
                )
            self._pending_nominal_seed_mode = next_solver_mode
            self._nominal_seeded = False
            self._nominal_seed_ready_at = now + self.MOTION_SETTLE_DELAY_S
            self._next_mode_switch_time = now + self.MOTION_SETTLE_DELAY_S

        self._active_solver_mode = next_solver_mode

    def solve_ik(
        self,
        q_in: np.ndarray,
    ) -> np.ndarray | None:
        if self._active_solver_mode == "up_down_mode":
            if (
                self._up_down_x_reference is None
                or self._up_down_y_reference is None
                or self._up_down_z_reference is None
            ):
                self.capture_up_down_references()
            if not self.update_up_down_z_reference_from_intent(time.monotonic()):
                return None
            return self.solver.solve_up_down(
                q_in,
                self._up_down_z_reference,
                self.solver.q0_deg,
                x_reference=self._up_down_x_reference,
                y_reference=self._up_down_y_reference,
            )

        return self.solver.solve_front_back(q_in, self.solver.q0_deg)

    def handle_joint0_current_control(self, joint4_current: float) -> None:
        if self._joint0_is_damped:
            if joint4_current < self.FRONT_BACK_JOINT4_CURRENT_THRESHOLD:
                self.publish_arm_pose_target(
                    [90.0, None, None, None, None, None],
                    delay_ms=500,
                )
                self._joint0_is_damped = False
            return

        if joint4_current > self.FRONT_BACK_JOINT4_CURRENT_THRESHOLD:
            self.publish_joint0_damping_command()
            self._joint0_is_damped = True

    def publish_arm_commands(
        self,
        q_in: np.ndarray,
        q_out_solver: np.ndarray,
        q_out: np.ndarray,
    ) -> None:
        joints_to_update: list[int] = []
        gripper_needs_update = False
        joints_to_consider = self.FRONT_BACK_IK_PUBLISH_JOINT_INDICES
        if self._active_solver_mode == "up_down_mode":
            joints_to_consider = self.UP_DOWN_IK_PUBLISH_JOINT_INDICES

        for joint_index in joints_to_consider:
            joint_delta = abs(float(q_in[joint_index]) - float(q_out_solver[joint_index]))
            if joint_delta <= self.COMMAND_THRESHOLD_DEG:
                continue
            joints_to_update.append(joint_index)

        if (
            self.latest_gripper_angle_deg is None
            or abs(self.latest_gripper_angle_deg - self._gripper_target_angle_deg)
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
                self.publish_gripper_target(self._gripper_target_angle_deg)
            return

        servo_targets_deg: list[float | None] = [None] * 6
        for joint_index in joints_to_publish:
            servo_targets_deg[joint_index] = float(q_out[joint_index])
        self.publish_arm_pose_target(servo_targets_deg)

        if gripper_needs_update:
            self.publish_gripper_target(self._gripper_target_angle_deg)

    def publish_joint0_damping_command(self) -> None:
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
        self.pub_arm_json.publish(msg)

    def publish_arm_pose_target(
        self,
        target_angles_deg: list[float | None],
        *,
        delay_ms: int | None = None,
    ) -> None:
        for servo_id, angle_deg in enumerate(target_angles_deg):
            if angle_deg is None:
                continue
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
            self.pub_arm_json.publish(msg)

    def publish_gripper_target(
        self,
        angle_deg: float,
    ) -> None:
        self._gripper_target_angle_deg = float(angle_deg)
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 1,
            "data": {
                "id": 6,
                "angle": self._gripper_target_angle_deg,
                "delay_ms": 0,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_json.publish(msg)

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

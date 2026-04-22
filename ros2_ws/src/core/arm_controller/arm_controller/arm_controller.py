#!/usr/bin/env python3
"""Execute motion-coordinator arm tasks through local IK and servo commands."""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32

from arm_controller.d1_ik_solver import D1IKSolver, IKSolveError
from hq_pcot_msgs.msg import ArmCommand, ArmState, ArmTask
from unitree_arm.msg import ArmString
from unitree_go.msg import WirelessController


class ArmControllerNode(Node):
    GRIPPER_CLOSED_ANGLE_DEG = -15.0
    GRIPPER_OPEN_ANGLE_DEG = 50.0
    COMMAND_STATE_PUBLISH_HZ = 10.0

    DEFAULT_ARM_POSE = [90.0, -10.0, 10.0, -85.0, -3.0, -90.0]
    FRONT_BACK_TRACK_UPDATE_MASK = (0, 1, 1, 1, 1, 1)
    FRONT_BACK_IK_UPDATE_JOINT0_THRESHOLD = 0.1
    FRONT_BACK_JOINT4_CURRENT_THRESHOLD = 100.0

    UP_DOWN_TRACK_UPDATE_MASK = (0, 1, 1, 0, 1, 0)
    UP_DOWN_Z_VELOCITY_MPS = 0.1
    UP_DOWN_Z_REF_MIN_M = 0.20
    UP_DOWN_Z_REF_MAX_M = 0.65

    COMMAND_THRESHOLD_DEG = 0.1
    MOTION_SETTLE_DELAY_S = 2.0

    def __init__(self) -> None:
        super().__init__("arm_controller")

        self.declare_parameter("arm_state_topic", "/arm/state")
        self.declare_parameter("arm_task_topic", "/arm_task")
        self.declare_parameter("arm_command_topic", "/arm_Command")
        self.declare_parameter("arm_command_state_topic", "/arm/commanded_angles")
        self.declare_parameter("arm_z_reference_topic", "/arm/z_reference")
        self.declare_parameter("wireless_topic", "/wirelesscontroller")

        self.arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self.arm_task_topic = str(self.get_parameter("arm_task_topic").value)
        self.arm_command_topic = str(self.get_parameter("arm_command_topic").value)
        self.arm_command_state_topic = str(
            self.get_parameter("arm_command_state_topic").value
        )
        self.arm_z_reference_topic = str(
            self.get_parameter("arm_z_reference_topic").value
        )
        self.wireless_topic = str(self.get_parameter("wireless_topic").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.solver = D1IKSolver()
        self.latest_gripper_angle_deg: float | None = None
        self.latest_target_joint_configuration_deg: np.ndarray | None = None
        self.latest_target_joint_update_mask: tuple[int, int, int, int, int, int] = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.latest_arm_angles_deg: np.ndarray | None = None
        self.gripper_target_angle_deg = self.GRIPPER_CLOSED_ANGLE_DEG
        self.last_wireless_keys = 0
        self.joint0_is_damped = True

        self.active_solver_mode: str | None = None
        self.active_tracking_task: int = ArmTask.TASK_UNSPECIFIED
        self.nominal_seeded = False
        self.nominal_seed_ready_at = 0.0
        self.pending_nominal_seed_mode: str | None = None
        self.up_down_z_motion_direction = 0
        self.up_down_z_reference: float | None = None
        self.up_down_x_reference: float | None = None
        self.up_down_y_reference: float | None = None
        self.last_up_down_control_time: float | None = None
        self.last_front_back_solve_joint0_deg: float | None = None
        self.last_commanded_angles_deg: np.ndarray | None = None

        self.pub_arm_json = self.create_publisher(
            ArmString,
            self.arm_command_topic,
            qos,
        )
        command_state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        z_reference_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub_arm_command_state = self.create_publisher(
            ArmCommand,
            self.arm_command_state_topic,
            command_state_qos,
        )
        self.pub_arm_z_reference = self.create_publisher(
            Float32,
            self.arm_z_reference_topic,
            z_reference_qos,
        )
        self.sub_arm_state = self.create_subscription(
            ArmState,
            self.arm_state_topic,
            self.on_arm_state,
            qos,
        )
        self.sub_arm_task = self.create_subscription(
            ArmTask,
            self.arm_task_topic,
            self.on_arm_task,
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
            self.wireless_topic,
            self.on_wireless,
            wireless_qos,
        )
        self.command_state_timer = self.create_timer(
            1.0 / self.COMMAND_STATE_PUBLISH_HZ,
            self.publish_command_state,
        )

        self.publish_arm_pose_target(self.DEFAULT_ARM_POSE)
        self.publish_gripper_target(self.GRIPPER_CLOSED_ANGLE_DEG)
        time.sleep(self.MOTION_SETTLE_DELAY_S)
        self.publish_joint0_damping_command()

    def on_arm_task(self, msg: ArmTask) -> None:
        task = int(msg.task)
        if task == ArmTask.TASK_FRONT_BACK_PRESET:
            self._apply_preset_task(
                mode="front_back_mode",
                target_deg=msg.preset_joint_configuration_deg[:6],
                update_mask=msg.preset_joint_update_mask[:6],
            )
            self.active_tracking_task = ArmTask.TASK_FRONT_BACK_TRACK
            return
        if task == ArmTask.TASK_UP_DOWN_PRESET:
            self._apply_preset_task(
                mode="up_down_mode",
                target_deg=msg.preset_joint_configuration_deg[:6],
                update_mask=msg.preset_joint_update_mask[:6],
            )
            self.active_tracking_task = ArmTask.TASK_UP_DOWN_TRACK
            self.up_down_z_motion_direction = 0
            return
        if task == ArmTask.TASK_FRONT_BACK_TRACK:
            if self.active_solver_mode != "front_back_mode":
                self._begin_solver_mode("front_back_mode", settle_delay_s=0.0)
            self.active_tracking_task = ArmTask.TASK_FRONT_BACK_TRACK
            return
        if task == ArmTask.TASK_UP_DOWN_TRACK:
            if self.active_solver_mode != "up_down_mode":
                self._begin_solver_mode("up_down_mode", settle_delay_s=0.0)
            self.active_tracking_task = ArmTask.TASK_UP_DOWN_TRACK
            self.up_down_z_motion_direction = max(-1, min(1, int(msg.z_motion_direction)))

    def on_arm_state(self, msg: ArmState) -> None:
        if len(msg.angle_deg) < 6:
            return

        current_q_deg = np.asarray(msg.angle_deg[:6], dtype=float)
        self.latest_arm_angles_deg = current_q_deg
        current_joint0_deg = float(current_q_deg[0])

        if len(msg.angle_deg) > 6:
            self.latest_gripper_angle_deg = float(msg.angle_deg[6])

        joint4_current = float(msg.current[4]) if len(msg.current) > 4 else 0.0
        self.handle_joint0_current_control(joint4_current)

        current_q_solver_deg = self._to_solver_configuration(current_q_deg)

        if not self.nominal_seeded:
            self._maybe_finalize_nominal_seed(current_q_solver_deg)
            self.publish_arm_commands(current_q_deg)
            return

        try:
            self._maybe_update_tracking_target(
                current_q_solver_deg=current_q_solver_deg,
                current_joint0_deg=current_joint0_deg,
            )
        except IKSolveError as exc:
            self.get_logger().warning(
                "IK solve failed; keeping node alive and reusing the last valid target. "
                f"debug: {exc}"
            )
        except Exception as exc:
            self.get_logger().error(f"Unexpected tracking update failure: {exc}")
        self.publish_arm_commands(current_q_deg)

    def on_wireless(self, msg: WirelessController) -> None:
        current_keys = int(msg.keys)
        previous_keys = self.last_wireless_keys
        self.last_wireless_keys = current_keys

        x_pressed = bool(current_keys & (1 << 10)) and not bool(previous_keys & (1 << 10))
        b_pressed = bool(current_keys & (1 << 9)) and not bool(previous_keys & (1 << 9))

        if x_pressed:
            self.publish_gripper_target(self.GRIPPER_OPEN_ANGLE_DEG)
        elif b_pressed:
            self.publish_gripper_target(self.GRIPPER_CLOSED_ANGLE_DEG)

    def _apply_preset_task(
        self,
        *,
        mode: str,
        target_deg: list[float],
        update_mask: list[int],
    ) -> None:
        materialized_target_deg = np.asarray(target_deg[:6], dtype=float)
        materialized_update_mask = tuple(int(value) for value in update_mask[:6])
        self.latest_target_joint_configuration_deg = materialized_target_deg
        self.latest_target_joint_update_mask = materialized_update_mask
        servo_targets_deg: list[float | None] = [None] * 6
        for joint_index, should_update in enumerate(materialized_update_mask):
            if should_update:
                servo_targets_deg[joint_index] = float(materialized_target_deg[joint_index])
        self.publish_arm_pose_target(servo_targets_deg)
        self._begin_solver_mode(mode, settle_delay_s=self.MOTION_SETTLE_DELAY_S)

    def _begin_solver_mode(self, mode: str, *, settle_delay_s: float) -> None:
        self.active_solver_mode = mode
        self.pending_nominal_seed_mode = mode
        self.nominal_seeded = False
        self.nominal_seed_ready_at = time.monotonic() + max(0.0, settle_delay_s)
        if mode == "front_back_mode":
            self.last_front_back_solve_joint0_deg = None

    def _maybe_finalize_nominal_seed(self, current_q_solver_deg: np.ndarray) -> None:
        if self.pending_nominal_seed_mode is None:
            return
        if time.monotonic() < self.nominal_seed_ready_at:
            return

        self.solver.save_current_configuration_as_q0(current_q_solver_deg)
        if self.pending_nominal_seed_mode == "up_down_mode":
            self.capture_up_down_references()
        else:
            self.up_down_x_reference = None
            self.up_down_y_reference = None
            self.up_down_z_reference = None
            self.last_up_down_control_time = None

        self.pending_nominal_seed_mode = None
        self.nominal_seeded = True

    def capture_up_down_references(self) -> None:
        self.up_down_x_reference = float(self.solver.nominal_end_effector_x)
        self.up_down_y_reference = float(self.solver.nominal_end_effector_y)
        self.up_down_z_reference = self.clamp_up_down_z_reference(
            float(self.solver.nominal_end_effector_z)
        )
        self.last_up_down_control_time = time.monotonic()
        self.publish_z_reference()

    def clamp_up_down_z_reference(self, z_reference: float) -> float:
        return min(
            self.UP_DOWN_Z_REF_MAX_M,
            max(self.UP_DOWN_Z_REF_MIN_M, z_reference),
        )

    def _maybe_update_tracking_target(
        self,
        *,
        current_q_solver_deg: np.ndarray,
        current_joint0_deg: float,
    ) -> None:
        if self.active_tracking_task == ArmTask.TASK_UP_DOWN_TRACK:
            self._update_up_down_target(current_q_solver_deg)
            return
        if self.active_tracking_task != ArmTask.TASK_FRONT_BACK_TRACK:
            return

        solve_reasons: list[str] = []
        if self.latest_target_joint_configuration_deg is None:
            solve_reasons.append("no_target")
        if self.latest_target_joint_update_mask != self.FRONT_BACK_TRACK_UPDATE_MASK:
            solve_reasons.append("update_mask_mismatch")
        delta_since_last_solve = None
        if self.last_front_back_solve_joint0_deg is None:
            solve_reasons.append("no_previous_front_back_solve")
        else:
            delta_since_last_solve = abs(
                current_joint0_deg - self.last_front_back_solve_joint0_deg
            )
            if delta_since_last_solve >= self.FRONT_BACK_IK_UPDATE_JOINT0_THRESHOLD:
                solve_reasons.append("joint0_threshold_reached")

        should_solve = bool(solve_reasons)
        if not should_solve:
            return

        q_out_solver_deg = self.solver.solve_front_back(
            current_q_solver_deg,
            self.solver.q0_deg,
        )
        self.latest_target_joint_configuration_deg = self._from_solver_configuration(
            q_out_solver_deg
        )
        self.latest_target_joint_update_mask = self.FRONT_BACK_TRACK_UPDATE_MASK
        self.last_front_back_solve_joint0_deg = current_joint0_deg

    def _update_up_down_target(self, current_q_solver_deg: np.ndarray) -> None:
        if (
            self.up_down_x_reference is None
            or self.up_down_y_reference is None
            or self.up_down_z_reference is None
        ):
            self.capture_up_down_references()

        should_solve = self._update_up_down_z_reference(time.monotonic())
        if (
            self.latest_target_joint_configuration_deg is None
            or self.latest_target_joint_update_mask != self.UP_DOWN_TRACK_UPDATE_MASK
        ):
            should_solve = True
        if not should_solve:
            return

        q_out_solver_deg = self.solver.solve_up_down(
            current_q_solver_deg,
            self.up_down_z_reference,
            self.solver.q0_deg,
            x_reference=self.up_down_x_reference,
            y_reference=self.up_down_y_reference,
        )
        self.latest_target_joint_configuration_deg = self._from_solver_configuration(
            q_out_solver_deg
        )
        self.latest_target_joint_update_mask = self.UP_DOWN_TRACK_UPDATE_MASK

    def _update_up_down_z_reference(self, now: float) -> bool:
        if self.last_up_down_control_time is None:
            self.last_up_down_control_time = now
            return False

        dt = now - self.last_up_down_control_time
        self.last_up_down_control_time = now
        if dt <= 0.0 or self.up_down_z_reference is None:
            return False
        if self.up_down_z_motion_direction == 0:
            return False

        updated_z_reference = self.clamp_up_down_z_reference(
            self.up_down_z_reference
            + float(self.up_down_z_motion_direction) * self.UP_DOWN_Z_VELOCITY_MPS * dt
        )
        if abs(updated_z_reference - self.up_down_z_reference) <= 1e-6:
            return False

        self.up_down_z_reference = updated_z_reference
        self.publish_z_reference()
        return True

    def _to_solver_configuration(self, joint_configuration_deg: np.ndarray) -> np.ndarray:
        q_solver_deg = np.asarray(joint_configuration_deg, dtype=float).copy()
        q_solver_deg[0] = -q_solver_deg[0]
        q_solver_deg[3] = -q_solver_deg[3]
        return q_solver_deg

    def _from_solver_configuration(self, joint_configuration_deg: np.ndarray) -> np.ndarray:
        q_target_deg = np.asarray(joint_configuration_deg, dtype=float).copy()
        q_target_deg[0] = -q_target_deg[0]
        q_target_deg[3] = -q_target_deg[3]
        return q_target_deg

    def handle_joint0_current_control(self, joint4_current: float) -> None:
        if self.joint0_is_damped:
            if joint4_current < self.FRONT_BACK_JOINT4_CURRENT_THRESHOLD:
                self.publish_arm_pose_target(
                    [90.0, None, None, None, None, None],
                    delay_ms=500,
                )
                self.joint0_is_damped = False
            return

        if joint4_current > self.FRONT_BACK_JOINT4_CURRENT_THRESHOLD:
            self.publish_joint0_damping_command()
            self.joint0_is_damped = True

    def publish_arm_commands(self, current_q_deg: np.ndarray) -> None:
        joints_to_update: list[int] = []
        gripper_needs_update = False

        if self.latest_target_joint_configuration_deg is not None:
            for joint_index, should_update in enumerate(self.latest_target_joint_update_mask):
                if not should_update:
                    continue
                if joint_index == 0 and self.joint0_is_damped:
                    continue
                joint_delta = abs(
                    float(current_q_deg[joint_index])
                    - float(self.latest_target_joint_configuration_deg[joint_index])
                )
                if joint_delta <= self.COMMAND_THRESHOLD_DEG:
                    continue
                joints_to_update.append(joint_index)

        if (
            self.latest_gripper_angle_deg is None
            or abs(self.latest_gripper_angle_deg - self.gripper_target_angle_deg)
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

        if joints_to_publish and self.latest_target_joint_configuration_deg is not None:
            servo_targets_deg: list[float | None] = [None] * 6
            for joint_index in joints_to_publish:
                servo_targets_deg[joint_index] = float(
                    self.latest_target_joint_configuration_deg[joint_index]
                )
            self.publish_arm_pose_target(servo_targets_deg)

        if gripper_needs_update:
            self.publish_gripper_target(self.gripper_target_angle_deg)

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

    def publish_command_state(self) -> None:
        if self.last_commanded_angles_deg is None:
            self.publish_z_reference()
            return

        msg = ArmCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.angle_deg = [float(value) for value in self.last_commanded_angles_deg]
        self.pub_arm_command_state.publish(msg)
        self.publish_z_reference()

    def publish_z_reference(self) -> None:
        if self.up_down_z_reference is None:
            return
        if self.active_tracking_task != ArmTask.TASK_UP_DOWN_TRACK:
            return

        msg = Float32()
        msg.data = float(self.up_down_z_reference)
        self.pub_arm_z_reference.publish(msg)

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
            self._update_last_commanded_angle(servo_id, float(angle_deg))

    def publish_gripper_target(self, angle_deg: float) -> None:
        self.gripper_target_angle_deg = float(angle_deg)
        payload = {
            "seq": 4,
            "address": 1,
            "funcode": 1,
            "data": {
                "id": 6,
                "angle": self.gripper_target_angle_deg,
                "delay_ms": 0,
            },
        }
        msg = ArmString()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_arm_json.publish(msg)
        self._update_last_commanded_angle(6, self.gripper_target_angle_deg)

    def _update_last_commanded_angle(self, joint_index: int, angle_deg: float) -> None:
        if self.last_commanded_angles_deg is None:
            self.last_commanded_angles_deg = np.zeros(7, dtype=np.float32)
        self.last_commanded_angles_deg[joint_index] = np.float32(angle_deg)


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

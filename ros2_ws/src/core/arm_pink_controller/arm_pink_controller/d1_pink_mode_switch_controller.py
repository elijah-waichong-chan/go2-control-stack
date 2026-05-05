from __future__ import annotations

from collections import deque
import math
from enum import Enum
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
from std_msgs.msg import Bool, Float32, String

from hq_pcot_msgs.msg import ArmTask, LoopStatus
from icon_lab_d1_ros2.msg import ServoCommand, ServoFeedback


SERVO_COUNT = 7
ARM_JOINT_COUNT = 6


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


class ControllerMode(str, Enum):
    FRONT_BACK = "front_back_mode"
    UP_DOWN = "up_down_mode"


class D1PinkModeSwitchController(Node):
    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    STATUS_STARTUP = 3
    DEFAULT_STARTUP_POSE_DEG = [90.0, -10.0, 10.0, 5.0, -3.0, 0.0]
    DEFAULT_FRONT_BACK_POSE_DEG = [math.nan, math.nan, math.nan, math.nan, math.nan, 0.0]
    DEFAULT_UP_DOWN_POSE_DEG = [math.nan, math.nan, math.nan, 0.0, math.nan, 90.0]
    DEFAULT_IK_Q0_SEED_DEG = [90.0, -10.0, 10.0, 0.0, -3.0, 90.0]
    DEFAULT_FRONT_BACK_MODE_RANGE_DEG = [-5.0, 5.0]
    DEFAULT_UP_DOWN_MODE_RANGE_DEG = [85.0, 95.0]
    DEFAULT_STARTUP_VELOCITY_DEG_S = 100.0
    DEFAULT_SWITCH_VELOCITY_DEG_S = 100.0
    DEFAULT_FRONT_BACK_JOINT0_TRIGGER_HIGH_VELOCITY_DEG_S = 20.0
    DEFAULT_FRONT_BACK_JOINT0_TRIGGER_LOW_VELOCITY_DEG_S = 1.0
    DEFAULT_FRONT_BACK_JOINT0_COMMAND_VELOCITY_DEG_S = 50.0
    DEFAULT_FRONT_BACK_JOINT0_RETURN_TARGET_DEG = 90.0
    DEFAULT_JOINT0_DAMPING_POWER_MW = 0.0
    DEFAULT_UP_DOWN_Z_COMMAND_VELOCITY_MPS = 0.1

    def __init__(self) -> None:
        super().__init__("d1_pink_mode_switch_controller")

        self.declare_parameter("feedback_topic", "/arm/servo_feedback")
        self.declare_parameter("command_topic", "/arm/servo_command_input")
        self.declare_parameter("arm_task_topic", "/arm_task")
        self.declare_parameter("mode_topic", "/d1_pink/mode")
        self.declare_parameter("z_ref_enabled_topic", "/d1_pink/enabled")
        self.declare_parameter("z_velocity_topic", "/d1_pink/z_velocity")
        self.declare_parameter(
            "startup_complete_topic", "/d1_pink/mode_switch_startup_complete"
        )
        self.declare_parameter("status_topic", "/status/arm_controller")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("startup_pose_deg", self.DEFAULT_STARTUP_POSE_DEG)
        self.declare_parameter("position_tolerance_deg", 2.0)
        self.declare_parameter(
            "startup_velocity_deg_s", self.DEFAULT_STARTUP_VELOCITY_DEG_S
        )
        self.declare_parameter(
            "switch_velocity_deg_s", self.DEFAULT_SWITCH_VELOCITY_DEG_S
        )
        self.declare_parameter("current_joint_index", 5)
        self.declare_parameter("current_threshold_ma", 30.0)
        self.declare_parameter(
            "front_back_joint0_trigger_high_velocity_deg_s",
            self.DEFAULT_FRONT_BACK_JOINT0_TRIGGER_HIGH_VELOCITY_DEG_S,
        )
        self.declare_parameter(
            "front_back_joint0_trigger_low_velocity_deg_s",
            self.DEFAULT_FRONT_BACK_JOINT0_TRIGGER_LOW_VELOCITY_DEG_S,
        )
        self.declare_parameter(
            "front_back_joint0_command_velocity_deg_s",
            self.DEFAULT_FRONT_BACK_JOINT0_COMMAND_VELOCITY_DEG_S,
        )
        self.declare_parameter(
            "front_back_joint0_return_target_deg",
            self.DEFAULT_FRONT_BACK_JOINT0_RETURN_TARGET_DEG,
        )
        self.declare_parameter(
            "joint0_damping_power_mw",
            self.DEFAULT_JOINT0_DAMPING_POWER_MW,
        )
        self.declare_parameter(
            "up_down_z_command_velocity_mps",
            self.DEFAULT_UP_DOWN_Z_COMMAND_VELOCITY_MPS,
        )
        self.declare_parameter(
            "front_back_pose_deg",
            self.DEFAULT_FRONT_BACK_POSE_DEG,
        )
        self.declare_parameter(
            "up_down_pose_deg",
            self.DEFAULT_UP_DOWN_POSE_DEG,
        )
        self.declare_parameter(
            "front_back_mode_range_deg",
            self.DEFAULT_FRONT_BACK_MODE_RANGE_DEG,
        )
        self.declare_parameter(
            "up_down_mode_range_deg",
            self.DEFAULT_UP_DOWN_MODE_RANGE_DEG,
        )
        self.declare_parameter("initial_mode", ControllerMode.FRONT_BACK.value)

        feedback_topic = str(self.get_parameter("feedback_topic").value)
        command_topic = str(self.get_parameter("command_topic").value)
        arm_task_topic = str(self.get_parameter("arm_task_topic").value)
        mode_topic = str(self.get_parameter("mode_topic").value)
        z_ref_enabled_topic = str(self.get_parameter("z_ref_enabled_topic").value)
        z_velocity_topic = str(self.get_parameter("z_velocity_topic").value)
        startup_complete_topic = str(self.get_parameter("startup_complete_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.startup_pose_deg = self._read_pose_parameter("startup_pose_deg")
        self.position_tolerance_deg = float(
            self.get_parameter("position_tolerance_deg").value
        )
        self.startup_velocity_deg_s = float(
            self.get_parameter("startup_velocity_deg_s").value
        )
        self.switch_velocity_deg_s = float(
            self.get_parameter("switch_velocity_deg_s").value
        )
        self.current_joint_index = int(self.get_parameter("current_joint_index").value)
        self.current_threshold_ma = float(
            self.get_parameter("current_threshold_ma").value
        )
        self.front_back_joint0_trigger_high_velocity_deg_s = float(
            self.get_parameter("front_back_joint0_trigger_high_velocity_deg_s").value
        )
        self.front_back_joint0_trigger_low_velocity_deg_s = float(
            self.get_parameter("front_back_joint0_trigger_low_velocity_deg_s").value
        )
        self.front_back_joint0_command_velocity_deg_s = float(
            self.get_parameter("front_back_joint0_command_velocity_deg_s").value
        )
        self.front_back_joint0_return_target_deg = float(
            self.get_parameter("front_back_joint0_return_target_deg").value
        )
        self.joint0_damping_power_mw = float(
            self.get_parameter("joint0_damping_power_mw").value
        )
        self.up_down_z_command_velocity_mps = float(
            self.get_parameter("up_down_z_command_velocity_mps").value
        )
        self.front_back_pose_deg = self._read_pose_parameter("front_back_pose_deg")
        self.up_down_pose_deg = self._read_pose_parameter("up_down_pose_deg")
        self.front_back_mode_range_deg = self._read_range_parameter(
            "front_back_mode_range_deg"
        )
        self.up_down_mode_range_deg = self._read_range_parameter(
            "up_down_mode_range_deg"
        )
        self.initial_mode = self._read_mode_parameter("initial_mode")

        if self.current_joint_index < 0 or self.current_joint_index >= ARM_JOINT_COUNT:
            raise ValueError("current_joint_index must be between 0 and 5")
        if self.position_tolerance_deg <= 0.0:
            raise ValueError("position_tolerance_deg must be > 0")
        if self.startup_velocity_deg_s <= 0.0:
            raise ValueError("startup_velocity_deg_s must be > 0")
        if self.switch_velocity_deg_s <= 0.0:
            raise ValueError("switch_velocity_deg_s must be > 0")
        if self.front_back_joint0_command_velocity_deg_s <= 0.0:
            raise ValueError("front_back_joint0_command_velocity_deg_s must be > 0")
        if not math.isfinite(self.front_back_joint0_return_target_deg):
            raise ValueError("front_back_joint0_return_target_deg must be finite")
        if self.front_back_joint0_trigger_low_velocity_deg_s < 0.0:
            raise ValueError("front_back_joint0_trigger_low_velocity_deg_s must be >= 0")
        if (
            self.front_back_joint0_trigger_high_velocity_deg_s
            <= self.front_back_joint0_trigger_low_velocity_deg_s
        ):
            raise ValueError(
                "front_back_joint0_trigger_high_velocity_deg_s must be > "
                "front_back_joint0_trigger_low_velocity_deg_s"
            )
        if self.joint0_damping_power_mw < 0.0:
            raise ValueError("joint0_damping_power_mw must be >= 0")
        if self.up_down_z_command_velocity_mps < 0.0:
            raise ValueError("up_down_z_command_velocity_mps must be >= 0")
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be > 0")
        self.loop_budget_ms = 1000.0 / max(self.control_rate_hz, 1.0)
        self.loop_stats_window = max(100, int(self.control_rate_hz * 10.0))

        self.feedback_sub = self.create_subscription(
            ServoFeedback,
            feedback_topic,
            self._handle_feedback,
            10,
        )
        self.arm_task_sub = self.create_subscription(
            ArmTask,
            arm_task_topic,
            self._handle_arm_task,
            10,
        )
        self.command_pub = self.create_publisher(ServoCommand, command_topic, 10)
        self.mode_pub = self.create_publisher(String, mode_topic, 10)
        self.z_ref_enabled_pub = self.create_publisher(Bool, z_ref_enabled_topic, 10)
        self.z_velocity_pub = self.create_publisher(Float32, z_velocity_topic, 10)
        self.startup_complete_pub = self.create_publisher(Bool, startup_complete_topic, 10)
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(LoopStatus, self.status_topic, status_qos)

        self.latest_feedback_deg: np.ndarray | None = None
        self.latest_current_ma: np.ndarray | None = None
        self.latest_velocity_deg_s: np.ndarray | None = None
        self.commanded_arm_pose_deg = self.startup_pose_deg.copy()
        self.current_mode: ControllerMode | None = None
        self.startup_started = False
        self.startup_complete = False
        self.current_switch_latched = False
        self.joint0_is_damped = False
        self.joint0_velocity_was_above_high_threshold = False
        self.pending_joint0_target_deg: float | None = None
        self.pending_motion_target_deg: np.ndarray | None = None
        self.pending_motion_mask = np.zeros(ARM_JOINT_COUNT, dtype=bool)
        self.pending_motion_description: str | None = None
        self.z_ref_enabled = False
        self.last_commanded_z_velocity_mps = 0.0
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.loop_times_ms: deque[float] = deque(maxlen=self.loop_stats_window)
        self.deadline_flags: deque[int] = deque(maxlen=self.loop_stats_window)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._on_timer)
        self.status_timer = self.create_timer(1.0 / self.status_hz, self._on_status_timer)
        self._update_status()

        self.get_logger().info(
            "Pink mode switch controller running on "
            f"{feedback_topic} -> {command_topic}. "
            f"Startup pose={self.startup_pose_deg.tolist()}, "
            f"startup_velocity={self.startup_velocity_deg_s:.1f} deg/s, "
            f"switch_velocity={self.switch_velocity_deg_s:.1f} deg/s, "
            f"position_tolerance={self.position_tolerance_deg:.1f} deg, "
            f"joint0 velocity trigger high={self.front_back_joint0_trigger_high_velocity_deg_s:.1f} deg/s, "
            f"joint0 velocity trigger low={self.front_back_joint0_trigger_low_velocity_deg_s:.1f} deg/s, "
            f"joint0 command velocity={self.front_back_joint0_command_velocity_deg_s:.1f} deg/s, "
            f"joint0 return target={self.front_back_joint0_return_target_deg:.1f} deg, "
            f"front_back pose={self.front_back_pose_deg.tolist()}, "
            f"up_down pose={self.up_down_pose_deg.tolist()}, "
            f"current joint={self.current_joint_index}, "
            f"threshold={self.current_threshold_ma:.1f} mA, "
            f"arm task topic={arm_task_topic}, "
            f"z_ref enabled topic={z_ref_enabled_topic}, "
            f"z_velocity topic={z_velocity_topic}, "
            f"up_down z command velocity={self.up_down_z_command_velocity_mps:.3f} m/s, "
            f"startup complete topic={startup_complete_topic}"
        )

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
            self.status_pub.publish(
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
        self.status_pub.publish(
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

    def _on_status_timer(self) -> None:
        self._publish_status()

    def _update_status(self) -> None:
        if (
            self.latest_feedback_deg is None
            or self.latest_current_ma is None
            or self.latest_velocity_deg_s is None
        ):
            if self.waiting_on != "feedback":
                self.get_logger().info("d1_pink_mode_switch_controller waiting for /arm/servo_feedback...")
                self.waiting_on = "feedback"
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        if not self.startup_complete:
            self.waiting_on = None
            self._set_status(self.STATUS_STARTUP)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("d1_pink_mode_switch_controller running")
            self.running_logged = True

    def _read_pose_parameter(self, name: str) -> np.ndarray:
        values = np.asarray(self.get_parameter(name).value, dtype=float).reshape(-1)
        if values.shape[0] != ARM_JOINT_COUNT:
            raise ValueError(f"{name} must have exactly {ARM_JOINT_COUNT} values")
        return values.copy()

    def _read_range_parameter(self, name: str) -> tuple[float, float]:
        values = np.asarray(self.get_parameter(name).value, dtype=float).reshape(-1)
        if values.shape[0] != 2:
            raise ValueError(f"{name} must contain exactly 2 values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must be finite")
        return float(values[0]), float(values[1])

    def _read_partial_pose(self, source_pose_deg: np.ndarray) -> np.ndarray:
        values = source_pose_deg.copy()
        if values.shape[0] != ARM_JOINT_COUNT:
            raise ValueError(
                f"pose must have exactly {ARM_JOINT_COUNT} values after materialization"
            )
        return values.copy()

    def _read_mode_parameter(self, name: str) -> ControllerMode:
        raw_value = str(self.get_parameter(name).value).strip().lower()
        try:
            return ControllerMode(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be one of {[mode.value for mode in ControllerMode]}"
            ) from exc

    def _handle_feedback(self, msg: ServoFeedback) -> None:
        angle_values = np.asarray(msg.angle_deg, dtype=float)
        current_values = np.asarray(msg.current_ma, dtype=float)
        velocity_values = np.asarray(msg.velocity_deg_s, dtype=float)
        if (
            angle_values.shape[0] < ARM_JOINT_COUNT
            or current_values.shape[0] < ARM_JOINT_COUNT
            or velocity_values.shape[0] < ARM_JOINT_COUNT
        ):
            return
        if not np.all(np.isfinite(angle_values[:ARM_JOINT_COUNT])):
            return
        if not np.all(np.isfinite(current_values[:ARM_JOINT_COUNT])):
            return
        if not np.all(np.isfinite(velocity_values[:ARM_JOINT_COUNT])):
            return

        self.latest_feedback_deg = angle_values[:ARM_JOINT_COUNT].copy()
        self.latest_current_ma = current_values[:ARM_JOINT_COUNT].copy()
        self.latest_velocity_deg_s = velocity_values[:ARM_JOINT_COUNT].copy()
        self._update_status()

    def _handle_arm_task(self, msg: ArmTask) -> None:
        if self.current_mode != ControllerMode.UP_DOWN or not self.z_ref_enabled:
            self._publish_z_velocity_command(0.0)
            return

        z_motion_direction = max(-1, min(1, int(msg.z_motion_direction)))
        self._publish_z_velocity_command(
            float(z_motion_direction) * self.up_down_z_command_velocity_mps
        )

    def _infer_mode(self, feedback_deg: np.ndarray) -> ControllerMode:
        joint5_angle_deg = float(feedback_deg[5])
        if min(self.front_back_mode_range_deg) <= joint5_angle_deg <= max(
            self.front_back_mode_range_deg
        ):
            return ControllerMode.FRONT_BACK
        if min(self.up_down_mode_range_deg) <= joint5_angle_deg <= max(
            self.up_down_mode_range_deg
        ):
            return ControllerMode.UP_DOWN
        return self.initial_mode

    def _target_reached(
        self,
        target_pose_deg: np.ndarray | None,
        active_mask: np.ndarray,
    ) -> bool:
        if (
            self.latest_feedback_deg is None
            or target_pose_deg is None
            or not np.any(active_mask)
        ):
            return False

        position_error_deg = np.abs(self.latest_feedback_deg - target_pose_deg)
        return bool(np.all(position_error_deg[active_mask] <= self.position_tolerance_deg))

    def _clear_pending_motion(self) -> None:
        self.pending_motion_target_deg = None
        self.pending_motion_mask = np.zeros(ARM_JOINT_COUNT, dtype=bool)
        self.pending_motion_description = None

    def _joint0_target_reached(self) -> bool:
        if self.latest_feedback_deg is None or self.pending_joint0_target_deg is None:
            return False

        return bool(
            abs(float(self.latest_feedback_deg[0]) - self.pending_joint0_target_deg)
            <= self.position_tolerance_deg
        )

    def _clear_pending_joint0_target(self) -> None:
        self.pending_joint0_target_deg = None

    def _set_z_ref_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.z_ref_enabled:
            return

        self.z_ref_enabled = enabled
        if not enabled:
            self._publish_z_velocity_command(0.0)
        msg = Bool()
        msg.data = enabled
        self.z_ref_enabled_pub.publish(msg)
        self.get_logger().info(
            f"Pink z_ref {'enabled' if enabled else 'disabled'} for {self.current_mode.value if self.current_mode is not None else 'startup'}"
        )

    def _publish_startup_complete(self) -> None:
        msg = Bool()
        msg.data = bool(self.startup_complete)
        self.startup_complete_pub.publish(msg)

    def _on_timer(self) -> None:
        loop_start = time.perf_counter()
        try:
            if (
                self.latest_feedback_deg is None
                or self.latest_current_ma is None
                or self.latest_velocity_deg_s is None
            ):
                self._update_status()
                return

            self._publish_startup_complete()

            if not self.startup_started:
                self.startup_started = True
                self._set_z_ref_enabled(False)
                self.commanded_arm_pose_deg = self.startup_pose_deg.copy()
                self.pending_motion_target_deg = self.commanded_arm_pose_deg.copy()
                self.pending_motion_mask = np.ones(ARM_JOINT_COUNT, dtype=bool)
                self.pending_motion_description = "startup pose"
                self._publish_pose_command(
                    self.commanded_arm_pose_deg,
                    velocity_deg_s=self.startup_velocity_deg_s,
                )
                self.get_logger().info(
                    "Published startup pose and waiting for position convergence "
                    f"within {self.position_tolerance_deg:.1f} deg before mode switching"
                )
                self._update_status()
                return

            if not self.startup_complete:
                if not self._target_reached(
                    self.pending_motion_target_deg,
                    self.pending_motion_mask,
                ):
                    self._update_status()
                    return

                self.startup_complete = True
                self._clear_pending_motion()
                self.current_mode = self._infer_mode(self.latest_feedback_deg)
                self._publish_mode()
                self._set_z_ref_enabled(
                    self.current_mode == ControllerMode.UP_DOWN
                    and self.pending_motion_target_deg is None
                )
                self.get_logger().info(
                    "Startup pose reached. "
                    f"Entering {self.current_mode.value} mode"
                )
                self._update_status()
                return

            if self.current_mode == ControllerMode.FRONT_BACK:
                self._update_front_back_joint0_control()

            if self.pending_motion_target_deg is not None:
                if not self._target_reached(
                    self.pending_motion_target_deg,
                    self.pending_motion_mask,
                ):
                    self._update_status()
                    return

                self.get_logger().info(
                    f"{self.pending_motion_description} reached within {self.position_tolerance_deg:.1f} deg"
                )
                if self.current_mode == ControllerMode.FRONT_BACK:
                    self._redamp_joint0_after_mode_target()
                self._clear_pending_motion()
                self._set_z_ref_enabled(self.current_mode == ControllerMode.UP_DOWN)

            current_value_ma = float(self.latest_current_ma[self.current_joint_index])
            if current_value_ma > self.current_threshold_ma:
                if not self.current_switch_latched:
                    next_mode = (
                        ControllerMode.UP_DOWN
                        if self.current_mode == ControllerMode.FRONT_BACK
                        else ControllerMode.FRONT_BACK
                    )
                    self.current_switch_latched = True
                    self.current_mode = next_mode
                    if self.current_mode != ControllerMode.FRONT_BACK:
                        self._clear_pending_joint0_target()
                        self.joint0_velocity_was_above_high_threshold = False
                        self.joint0_is_damped = False
                    self._apply_mode_targets(next_mode)
                    self._set_z_ref_enabled(False)
                    self._publish_mode()
                    self.get_logger().info(
                        f"Current {current_value_ma:.1f} mA exceeded threshold; switched to {self.current_mode.value}"
                    )
                self._update_status()
                return

            self.current_switch_latched = False
            self._update_status()
        finally:
            loop_time_ms = (time.perf_counter() - loop_start) * 1000.0
            self._record_loop_time_ms(loop_time_ms)

    def _apply_mode_targets(self, mode: ControllerMode) -> None:
        if mode == ControllerMode.FRONT_BACK:
            preset_pose_deg = self.front_back_pose_deg
        else:
            preset_pose_deg = self.up_down_pose_deg

        target_pose_deg = self.latest_feedback_deg.copy()
        finite_mask = np.isfinite(preset_pose_deg)
        if mode == ControllerMode.FRONT_BACK:
            finite_mask[0] = False
        target_pose_deg[finite_mask] = preset_pose_deg[finite_mask]

        self.commanded_arm_pose_deg = self._read_partial_pose(target_pose_deg)
        self.pending_motion_target_deg = self.commanded_arm_pose_deg.copy()
        self.pending_motion_mask = finite_mask.copy()
        self.pending_motion_description = f"{mode.value} target"
        self._publish_pose_command(
            self.commanded_arm_pose_deg,
            velocity_deg_s=self.switch_velocity_deg_s,
        )

    def _redamp_joint0_after_mode_target(self) -> None:
        self._clear_pending_joint0_target()
        self.joint0_velocity_was_above_high_threshold = False
        self._publish_single_joint_damping_command(
            joint_index=0,
            damping_power_mw=self.joint0_damping_power_mw,
        )
        self.joint0_is_damped = True

    def _update_front_back_joint0_control(self) -> None:
        if self.pending_joint0_target_deg is not None:
            if not self._joint0_target_reached():
                return
            self.get_logger().info(
                f"joint0 target reached within {self.position_tolerance_deg:.1f} deg"
            )
            self._clear_pending_joint0_target()
            self._publish_single_joint_damping_command(
                joint_index=0,
                damping_power_mw=self.joint0_damping_power_mw,
            )
            self.joint0_is_damped = True
            self.joint0_velocity_was_above_high_threshold = False
            return

        if not self.joint0_is_damped:
            self._publish_single_joint_damping_command(
                joint_index=0,
                damping_power_mw=self.joint0_damping_power_mw,
            )
            self.joint0_is_damped = True
            self.joint0_velocity_was_above_high_threshold = False
            return

        joint0_abs_velocity_deg_s = abs(float(self.latest_velocity_deg_s[0]))
        if joint0_abs_velocity_deg_s > self.front_back_joint0_trigger_high_velocity_deg_s:
            self.joint0_velocity_was_above_high_threshold = True
            return

        if (
            self.joint0_velocity_was_above_high_threshold
            and joint0_abs_velocity_deg_s < self.front_back_joint0_trigger_low_velocity_deg_s
        ):
            self._publish_single_joint_velocity_command(
                joint_index=0,
                target_angle_deg=self.front_back_joint0_return_target_deg,
                velocity_deg_s=self.front_back_joint0_command_velocity_deg_s,
            )
            self.pending_joint0_target_deg = self.front_back_joint0_return_target_deg
            self.joint0_is_damped = False
            self.joint0_velocity_was_above_high_threshold = False

    def _ensure_joint0_damped(self) -> None:
        self._clear_pending_joint0_target()
        self.joint0_velocity_was_above_high_threshold = False
        if self.joint0_is_damped:
            return

        self._publish_single_joint_damping_command(
            joint_index=0,
            damping_power_mw=self.joint0_damping_power_mw,
        )
        self.joint0_is_damped = True

    def _publish_single_joint_velocity_command(
        self,
        *,
        joint_index: int,
        target_angle_deg: float,
        velocity_deg_s: float,
    ) -> None:
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.mode = [ServoCommand.MODE_IGNORE] * SERVO_COUNT
        command.mode[joint_index] = ServoCommand.MODE_POSITION_VELOCITY
        command.angle_deg = [math.nan] * SERVO_COUNT
        command.angle_deg[joint_index] = float(target_angle_deg)
        command.velocity_deg_s = [math.nan] * SERVO_COUNT
        command.velocity_deg_s[joint_index] = float(velocity_deg_s)
        command.damping_power_mw = [0.0] * SERVO_COUNT
        command.interval_ms = 0
        self.command_pub.publish(command)

    def _publish_single_joint_damping_command(
        self,
        *,
        joint_index: int,
        damping_power_mw: float,
    ) -> None:
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.mode = [ServoCommand.MODE_IGNORE] * SERVO_COUNT
        command.mode[joint_index] = ServoCommand.MODE_DAMPING
        command.angle_deg = [math.nan] * SERVO_COUNT
        command.velocity_deg_s = [math.nan] * SERVO_COUNT
        command.damping_power_mw = [0.0] * SERVO_COUNT
        command.damping_power_mw[joint_index] = float(damping_power_mw)
        command.interval_ms = 0
        self.command_pub.publish(command)

    def _publish_pose_command(
        self,
        target_pose_deg: np.ndarray,
        *,
        interval_ms: int | None = None,
        velocity_deg_s: float | None = None,
    ) -> None:
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        if velocity_deg_s is not None:
            command.mode = [ServoCommand.MODE_POSITION_VELOCITY] * ARM_JOINT_COUNT + [
                ServoCommand.MODE_IGNORE
            ]
            command.velocity_deg_s = [float(velocity_deg_s)] * ARM_JOINT_COUNT + [math.nan]
            command.interval_ms = 0
        else:
            if interval_ms is None:
                raise ValueError("interval_ms is required for interval mode commands")
            command.mode = [ServoCommand.MODE_POSITION_INTERVAL] * ARM_JOINT_COUNT + [
                ServoCommand.MODE_IGNORE
            ]
            command.velocity_deg_s = [math.nan] * (ARM_JOINT_COUNT + 1)
            command.interval_ms = int(interval_ms)
        command.angle_deg = [float(value) for value in target_pose_deg[:ARM_JOINT_COUNT]] + [
            math.nan
        ]
        command.damping_power_mw = [0.0] * (ARM_JOINT_COUNT + 1)
        self.command_pub.publish(command)

    def _publish_mode(self) -> None:
        msg = String()
        msg.data = self.current_mode.value
        self.mode_pub.publish(msg)

    def _publish_z_velocity_command(self, z_velocity_mps: float) -> None:
        z_velocity_mps = float(z_velocity_mps)
        if abs(z_velocity_mps - self.last_commanded_z_velocity_mps) <= 1e-9:
            return

        self.last_commanded_z_velocity_mps = z_velocity_mps
        msg = Float32()
        msg.data = z_velocity_mps
        self.z_velocity_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = D1PinkModeSwitchController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

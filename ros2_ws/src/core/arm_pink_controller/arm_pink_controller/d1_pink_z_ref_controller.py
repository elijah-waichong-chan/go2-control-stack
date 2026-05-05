from __future__ import annotations

from collections import deque
import math
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
from std_msgs.msg import Bool, Float32

from hq_pcot_msgs.msg import LoopStatus
from icon_lab_d1_ros2.msg import ServoCommand, ServoFeedback

from arm_pink_controller.pink_solver import D1PinkSolver, PinkUnavailableError


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


class D1PinkZRefController(Node):
    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    STATUS_STARTUP = 3
    STARTUP_HOME_DEG = np.array([90.0, -10.0, 10.0, 0.0, -3.0, 90.0], dtype=float)
    STARTUP_VELOCITY_DEG_S = 80.0
    POSITION_TOLERANCE_DEG = 2.0
    TRACKING_COMMAND_DEADBAND_DEG = 0.5

    def __init__(self) -> None:
        super().__init__("d1_pink_z_ref_controller")

        self.declare_parameter("feedback_topic", "/arm/servo_feedback")
        self.declare_parameter("command_topic", "/arm/servo_command_input")
        self.declare_parameter("z_ref_topic", "/d1_pink/z_ref")
        self.declare_parameter("z_velocity_topic", "/d1_pink/z_velocity")
        self.declare_parameter("current_z_topic", "/d1_pink/current_z")
        self.declare_parameter("solver_target_z_topic", "/d1_pink/solver_target_z")
        self.declare_parameter("solver_current_z_topic", "/d1_pink/solver_current_z")
        self.declare_parameter("enabled_topic", "/d1_pink/enabled")
        self.declare_parameter("startup_complete_topic", "/d1_pink/startup_complete")
        self.declare_parameter("status_topic", "/status/arm_z_ref_controller")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("z_ref_min_m", 0.15)
        self.declare_parameter("z_ref_max_m", 0.50)
        self.declare_parameter("max_abs_z_velocity_mps", 0.20)
        self.declare_parameter("startup_velocity_deg_s", self.STARTUP_VELOCITY_DEG_S)
        self.declare_parameter("position_tolerance_deg", self.POSITION_TOLERANCE_DEG)
        self.declare_parameter("defer_startup_until_enabled", False)
        self.declare_parameter(
            "tracking_command_deadband_deg", self.TRACKING_COMMAND_DEADBAND_DEG
        )

        feedback_topic = self.get_parameter("feedback_topic").value
        command_topic = self.get_parameter("command_topic").value
        z_ref_topic = self.get_parameter("z_ref_topic").value
        z_velocity_topic = self.get_parameter("z_velocity_topic").value
        current_z_topic = self.get_parameter("current_z_topic").value
        solver_target_z_topic = self.get_parameter("solver_target_z_topic").value
        solver_current_z_topic = self.get_parameter("solver_current_z_topic").value
        enabled_topic = self.get_parameter("enabled_topic").value
        startup_complete_topic = self.get_parameter("startup_complete_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.tracking_command_interval_ms = max(1, int(round(1000.0 / self.control_rate_hz)))
        z_ref_min_m = float(self.get_parameter("z_ref_min_m").value)
        z_ref_max_m = float(self.get_parameter("z_ref_max_m").value)
        self.max_abs_z_velocity_mps = float(self.get_parameter("max_abs_z_velocity_mps").value)
        self.startup_velocity_deg_s = float(self.get_parameter("startup_velocity_deg_s").value)
        self.position_tolerance_deg = float(self.get_parameter("position_tolerance_deg").value)
        self.defer_startup_until_enabled = bool(
            self.get_parameter("defer_startup_until_enabled").value
        )
        self.tracking_command_deadband_deg = float(
            self.get_parameter("tracking_command_deadband_deg").value
        )

        if self.startup_velocity_deg_s <= 0.0:
            raise ValueError("startup_velocity_deg_s must be > 0")
        if self.position_tolerance_deg <= 0.0:
            raise ValueError("position_tolerance_deg must be > 0")
        if self.tracking_command_deadband_deg < 0.0:
            raise ValueError("tracking_command_deadband_deg must be >= 0")
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be > 0")
        self.loop_budget_ms = 1000.0 / max(self.control_rate_hz, 1.0)
        self.loop_stats_window = max(100, int(self.control_rate_hz * 10.0))

        try:
            self.solver = D1PinkSolver(z_ref_min_m=z_ref_min_m, z_ref_max_m=z_ref_max_m)
        except PinkUnavailableError as exc:
            self.get_logger().error(str(exc))
            raise

        self.feedback_sub = self.create_subscription(
            ServoFeedback, feedback_topic, self._handle_feedback, 10
        )
        self.z_velocity_sub = self.create_subscription(
            Float32, z_velocity_topic, self._handle_z_velocity, 10
        )
        self.enabled_sub = self.create_subscription(Bool, enabled_topic, self._handle_enabled, 10)
        self.command_pub = self.create_publisher(ServoCommand, command_topic, 10)
        self.z_ref_pub = self.create_publisher(Float32, z_ref_topic, 10)
        self.current_z_pub = self.create_publisher(Float32, current_z_topic, 10)
        self.solver_target_z_pub = self.create_publisher(Float32, solver_target_z_topic, 10)
        self.solver_current_z_pub = self.create_publisher(Float32, solver_current_z_topic, 10)
        self.startup_complete_pub = self.create_publisher(Bool, startup_complete_topic, 10)
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(LoopStatus, self.status_topic, status_qos)

        self.latest_servo_feedback_deg: np.ndarray | None = None
        self.desired_z_reference_m: float | None = None
        self.manual_z_velocity_mps = 0.0
        self.ik_enabled = False
        self.startup_home_command_sent = False
        self.startup_complete = False
        self.last_tracking_command_deg: np.ndarray | None = None
        self.last_control_time_sec: float | None = None
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.loop_times_ms: deque[float] = deque(maxlen=self.loop_stats_window)
        self.deadline_flags: deque[int] = deque(maxlen=self.loop_stats_window)
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._on_timer)
        self.status_timer = self.create_timer(1.0 / self.status_hz, self._on_status_timer)
        self._update_status()

        self.get_logger().info(
            f"Pink z_ref controller running on {feedback_topic} -> {command_topic} "
            f"publishing z_ref state topic {z_ref_topic}, z velocity input topic {z_velocity_topic} "
            f"solver target z topic {solver_target_z_topic}, "
            f"solver current z topic {solver_current_z_topic}, "
            f"at {self.control_rate_hz:.1f} Hz; "
            f"command interval {self.tracking_command_interval_ms} ms; "
            f"startup velocity {self.startup_velocity_deg_s:.1f} deg/s; "
            f"position tolerance {self.position_tolerance_deg:.1f} deg; "
            f"defer startup until enabled={self.defer_startup_until_enabled}; "
            f"tracking deadband {self.tracking_command_deadband_deg:.2f} deg; "
            f"IK enable topic {enabled_topic}; "
            f"startup complete topic {startup_complete_topic}; "
            f"status topic {self.status_topic}"
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
        if self.latest_servo_feedback_deg is None:
            if self.waiting_on != "feedback":
                self.get_logger().info("d1_pink_z_ref_controller waiting for /arm/servo_feedback...")
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
            self.get_logger().info("d1_pink_z_ref_controller running")
            self.running_logged = True

    def _startup_target_reached(self) -> bool:
        if self.latest_servo_feedback_deg is None:
            return False
        error_deg = np.abs(self.latest_servo_feedback_deg[:ARM_JOINT_COUNT] - self.STARTUP_HOME_DEG)
        return bool(np.all(error_deg <= self.position_tolerance_deg))

    def _handle_feedback(self, msg: ServoFeedback) -> None:
        values = np.asarray(msg.angle_deg, dtype=float)
        if values.shape[0] < SERVO_COUNT:
            return
        if not np.all(np.isfinite(values[:ARM_JOINT_COUNT])):
            return

        self.latest_servo_feedback_deg = values.copy()
        current_model_q = self.solver.servo_to_model(values)
        current_z = self.solver.get_current_z(current_model_q)
        self._publish_current_z(current_z)

    def _handle_z_velocity(self, msg: Float32) -> None:
        self.manual_z_velocity_mps = float(
            np.clip(msg.data, -self.max_abs_z_velocity_mps, self.max_abs_z_velocity_mps)
        )

    def _handle_enabled(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled == self.ik_enabled:
            return
        self.ik_enabled = enabled
        if self.defer_startup_until_enabled:
            self.startup_complete = False
            self.last_tracking_command_deg = None
        self.get_logger().info(f"Pink IK {'enabled' if enabled else 'disabled'}")

    def _publish_current_z(self, current_z: float) -> None:
        msg = Float32()
        msg.data = float(current_z)
        self.current_z_pub.publish(msg)

    def _publish_solver_current_z(self, current_z: float) -> None:
        msg = Float32()
        msg.data = float(current_z)
        self.solver_current_z_pub.publish(msg)

    def _publish_z_ref(self) -> None:
        if self.desired_z_reference_m is None:
            return
        msg = Float32()
        msg.data = float(self.desired_z_reference_m)
        self.z_ref_pub.publish(msg)

    def _publish_solver_target_z(self) -> None:
        if self.desired_z_reference_m is None:
            return
        msg = Float32()
        msg.data = float(self.desired_z_reference_m)
        self.solver_target_z_pub.publish(msg)

    def _publish_startup_complete(self) -> None:
        msg = Bool()
        msg.data = bool(self.startup_complete)
        self.startup_complete_pub.publish(msg)

    def _publish_home_command(self) -> None:
        if self.latest_servo_feedback_deg is None:
            return
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.mode = [ServoCommand.MODE_POSITION_VELOCITY] * ARM_JOINT_COUNT + [
            ServoCommand.MODE_IGNORE
        ]
        command.angle_deg = [float(value) for value in self.STARTUP_HOME_DEG] + [math.nan]
        command.velocity_deg_s = [self.startup_velocity_deg_s] * ARM_JOINT_COUNT + [math.nan]
        command.damping_power_mw = [0.0] * (ARM_JOINT_COUNT + 1)
        command.interval_ms = 0
        self.command_pub.publish(command)

    def _tracking_command_changed(self, servo_targets: np.ndarray) -> bool:
        if self.last_tracking_command_deg is None:
            return True
        delta_deg = np.abs(
            servo_targets[:ARM_JOINT_COUNT] - self.last_tracking_command_deg[:ARM_JOINT_COUNT]
        )
        return bool(np.any(delta_deg >= self.tracking_command_deadband_deg))

    def _on_timer(self) -> None:
        loop_start = time.perf_counter()
        if self.latest_servo_feedback_deg is None:
            self._update_status()
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_control_time_sec is None:
            dt = 1.0 / self.control_rate_hz
        else:
            dt = max(1e-3, now_sec - self.last_control_time_sec)
        self.last_control_time_sec = now_sec

        current_model_q = self.solver.servo_to_model(self.latest_servo_feedback_deg)
        current_z = self.solver.get_current_z(current_model_q)
        self._publish_current_z(current_z)
        self._publish_solver_current_z(current_z)
        self._publish_startup_complete()
        self._publish_z_ref()
        self._publish_solver_target_z()
        self._update_status()

        if self.defer_startup_until_enabled:
            if not self.ik_enabled:
                self._update_status()
                self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
                return

            if not self.startup_complete:
                seeded_z = self.solver.set_q0(current_model_q)
                if self.desired_z_reference_m is None:
                    self.desired_z_reference_m = current_z
                self.startup_complete = True
                self.last_tracking_command_deg = None
                self._publish_startup_complete()
                self._publish_z_ref()
                self._publish_solver_target_z()
                self.get_logger().info(
                    "Initialized Pink z_ref after external enable; "
                    f"reseeded q0 from current pose (nominal z={seeded_z:.4f} m) and kept z_ref={self.desired_z_reference_m:.4f} m"
                )
                self._update_status()
                self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
                return

        if not self.startup_home_command_sent:
            self.startup_home_command_sent = True
            self._publish_home_command()
            self.get_logger().info(
                "Publishing startup home pose in velocity mode and waiting until "
                f"all joints are within {self.position_tolerance_deg:.1f} deg"
            )
            self._update_status()
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return

        if not self.startup_complete:
            if not self._startup_target_reached():
                self._update_status()
                self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
                return

            settled_model_q = self.solver.servo_to_model(self.latest_servo_feedback_deg)
            settled_z = self.solver.set_q0(settled_model_q)
            self.desired_z_reference_m = current_z
            self.ik_enabled = True
            self.startup_complete = True
            self.last_tracking_command_deg = None
            self._publish_startup_complete()
            self._publish_z_ref()
            self._publish_solver_target_z()
            self.get_logger().info(
                "Startup settle complete. Seeded q0 from settled pose, "
                f"nominal z={settled_z:.4f} m, synced z_ref={self.desired_z_reference_m:.4f} m, and auto-enabled Pink IK"
            )
            self._update_status()
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return

        if not self.ik_enabled:
            self._update_status()
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return

        if self.desired_z_reference_m is None:
            self.desired_z_reference_m = current_z

        if abs(self.manual_z_velocity_mps) > 0.0:
            self.desired_z_reference_m = float(
                np.clip(
                    self.desired_z_reference_m + self.manual_z_velocity_mps * dt,
                    self.solver.z_ref_min_m,
                    self.solver.z_ref_max_m,
                )
            )
        self._publish_z_ref()
        self._publish_solver_target_z()

        try:
            result = self.solver.solve_step(
                current_q_deg=current_model_q,
                z_reference_m=self.desired_z_reference_m,
                dt=dt,
            )
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Pink solve failed: {exc}")
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return

        self._publish_current_z(result.current_z_m)
        self._publish_solver_current_z(result.current_z_m)
        servo_targets = self.solver.model_to_servo(
            result.joint_command_deg, self.latest_servo_feedback_deg
        )
        if not np.all(np.isfinite(servo_targets[:ARM_JOINT_COUNT])):
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return
        if not self._tracking_command_changed(servo_targets):
            self._update_status()
            self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)
            return

        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.mode = [ServoCommand.MODE_POSITION_INTERVAL] * ARM_JOINT_COUNT + [
            ServoCommand.MODE_IGNORE
        ]
        command.angle_deg = [float(value) for value in servo_targets[:ARM_JOINT_COUNT]] + [math.nan]
        command.velocity_deg_s = [math.nan] * (ARM_JOINT_COUNT + 1)
        command.damping_power_mw = [0.0] * (ARM_JOINT_COUNT + 1)
        command.interval_ms = self.tracking_command_interval_ms
        self.command_pub.publish(command)
        self.last_tracking_command_deg = servo_targets.copy()
        self._update_status()
        self._record_loop_time_ms((time.perf_counter() - loop_start) * 1000.0)


def main() -> None:
    rclpy.init()
    node = D1PinkZRefController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

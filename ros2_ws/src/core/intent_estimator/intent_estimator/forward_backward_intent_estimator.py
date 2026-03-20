#!/usr/bin/env python3
"""ROS 2 node for the 012 forward/backward direction intent model."""

from __future__ import annotations

from pathlib import Path
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from go2_msgs.msg import LoopStatus
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from unitree_go.msg import LowState

from intent_estimator.model_runtime import (
    SlidingWindowIntentModel,
    first_existing_path,
)


def _default_model_dir(bundle_name: str) -> Path:
    """Resolve the first usable model bundle path across install and source layouts."""
    module_path = Path(__file__).resolve()
    share_dir = Path(get_package_share_directory("intent_estimator"))
    return first_existing_path(
        [
            share_dir / "config" / "models" / bundle_name,
            module_path.parents[1] / "config" / "models" / bundle_name,
            module_path.parents[3] / "core" / "intent_estimator" / "config" / "models" / bundle_name,
        ]
    )


def _make_loop_status(status_code: int) -> LoopStatus:
    msg = LoopStatus()
    msg.status = int(status_code)
    msg.avg_loop_ms = -1.0
    msg.p99_loop_ms = -1.0
    msg.max_loop_ms = -1.0
    msg.budget_ms = -1.0
    msg.deadline_miss_count = -1
    msg.sample_count = -1
    return msg


class ForwardBackwardIntentEstimatorNode(Node):
    """Run the 012 model from foot force, IMU acceleration, and joint_states."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    LEG_JOINT_ORDER = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )

    def __init__(self) -> None:
        super().__init__("forward_backward_intent_estimator")

        default_model_dir = _default_model_dir("012")

        self.model_dir = default_model_dir.expanduser()
        self.lowstate_topic = "/lowstate"
        self.joint_states_topic = "/joint_states"
        self.output_topic = "/direction_intent/forward_backward"
        self.status_topic = "/status/intent_estimator/forward_backward"
        self.status_hz = 10.0
        self.sliding_window_ms = 300.0
        self.sampling_hz = 200.0
        self.publish_hz = 10.0

        self.model = SlidingWindowIntentModel(
            logger=self.get_logger(),
            model_dir=self.model_dir,
            sliding_window_ms=self.sliding_window_ms,
            sampling_hz=self.sampling_hz,
            publish_hz=self.publish_hz,
        )
        self.last_joint_state: JointState | None = None
        self.joint_index: dict[str, int] = {}
        self.have_lowstate = False
        self.have_joint_states = False
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_lowstate = self.create_subscription(
            LowState, self.lowstate_topic, self.on_lowstate, sensor_qos
        )
        self.sub_joint_states = self.create_subscription(
            JointState, self.joint_states_topic, self.on_joint_states, sensor_qos
        )
        self.pub_intent = self.create_publisher(Int32, self.output_topic, 10)
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)
        self.status_timer = self.create_timer(1.0 / max(1.0, self.status_hz), self.on_status_timer)
        self._update_status()

        self.get_logger().info(
            "forward_backward_intent_estimator ready: "
            f"{self.lowstate_topic} + {self.joint_states_topic} -> {self.output_topic}, "
            f"status={self.status_topic}, "
            f"model={self.model.metadata.model_path}, "
            f"window={self.sliding_window_ms:.0f}ms@{self.sampling_hz:.0f}Hz, "
            f"publish={self.publish_hz:.1f}Hz, "
            f"input={self.model.input_name}[batch,"
            f"{self.model.metadata.num_features},"
            f"{self.model.metadata.num_timesteps}], "
            f"output={self.model.output_name}[batch,{self.model.metadata.output_dim}], "
            f"labels={self.model.metadata.index_to_label}"
        )

    def _set_status(self, status_code: int) -> None:
        self.status_code = int(status_code)

    def on_status_timer(self) -> None:
        self.pub_status.publish(_make_loop_status(self.status_code))

    def _update_status(self) -> None:
        missing_topics = []
        if not self.have_lowstate:
            missing_topics.append(self.lowstate_topic)
        if not self.have_joint_states:
            missing_topics.append(self.joint_states_topic)

        if missing_topics:
            waiting_on = ", ".join(missing_topics)
            if waiting_on != self.waiting_on:
                self.get_logger().info(
                    "forward_backward_intent_estimator waiting for %s..."
                    % waiting_on
                )
                self.waiting_on = waiting_on
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("forward_backward_intent_estimator running")
            self.running_logged = True

    def on_joint_states(self, msg: JointState) -> None:
        """Cache the latest joint_states message for the next lowstate sample."""
        self.last_joint_state = msg
        self.joint_index = {name: i for i, name in enumerate(msg.name)}
        self.have_joint_states = True
        self._update_status()

    def on_lowstate(self, msg: LowState) -> None:
        """Build a 19-feature sample and publish the throttled predicted label."""
        self.have_lowstate = True
        self._update_status()
        if self.last_joint_state is None:
            return

        foot_force = [float(x) for x in msg.foot_force[:4]]
        accel = [float(x) for x in msg.imu_state.accelerometer[:3]]
        if len(foot_force) != 4:
            self.get_logger().error(
                f"Expected 4 foot-force values from /lowstate.foot_force, got {len(foot_force)}."
            )
            return
        if len(accel) != 3:
            self.get_logger().error(
                "Expected 3 accel values from /lowstate.imu_state.accelerometer, "
                f"got {len(accel)}."
            )
            return

        joint_dq: list[float] = []
        missing_joints: list[str] = []
        joint_state = self.last_joint_state
        for joint_name in self.LEG_JOINT_ORDER:
            joint_idx = self.joint_index.get(joint_name)
            if joint_idx is None or joint_idx >= len(joint_state.velocity):
                missing_joints.append(joint_name)
                continue
            joint_dq.append(float(joint_state.velocity[joint_idx]))

        if missing_joints:
            self.get_logger().error(
                "Expected leg joint velocities from /joint_states, missing: "
                f"{missing_joints}."
            )
            return

        features = foot_force + accel + joint_dq

        try:
            pred_label = self.model.push(features, time.monotonic())
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime inference errors
            self.get_logger().error(f"ONNX inference failed: {exc}")
            return

        if pred_label is None:
            return

        out = Int32()
        out.data = int(pred_label)
        self.pub_intent.publish(out)


def main() -> None:
    """Run the forward/backward direction intent estimator node."""
    rclpy.init()
    node = ForwardBackwardIntentEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ROS 2 node for the 034 left/right direction intent model."""

from __future__ import annotations

from pathlib import Path
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from go2_msgs.msg import ArmAngles, LoopStatus
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32

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


class LeftRightIntentEstimatorNode(Node):
    """Subscribe to arm angles, run ONNX inference, and publish intent labels."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2

    def __init__(self) -> None:
        super().__init__("left_right_intent_estimator")

        default_model_dir = _default_model_dir("034")

        self.model_dir = default_model_dir.expanduser()
        self.arm_angles_topic = "/arm_angles"
        self.output_topic = "/direction_intent/left_right"
        self.status_topic = "/status/intent_estimator/left_right"
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
        self.have_arm_angles = False
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

        self.sub_arm_angles = self.create_subscription(
            ArmAngles, self.arm_angles_topic, self.on_arm_angles, sensor_qos
        )
        self.pub_intent = self.create_publisher(Int32, self.output_topic, 10)
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)
        self.status_timer = self.create_timer(1.0 / max(1.0, self.status_hz), self.on_status_timer)
        self._update_status()

        self.get_logger().info(
            "left_right_intent_estimator ready: "
            f"{self.arm_angles_topic} -> {self.output_topic}, "
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
        if not self.have_arm_angles:
            if self.waiting_on != self.arm_angles_topic:
                self.get_logger().info(
                    f"left_right_intent_estimator waiting for {self.arm_angles_topic}..."
                )
                self.waiting_on = self.arm_angles_topic
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("left_right_intent_estimator running")
            self.running_logged = True

    def on_arm_angles(self, msg: ArmAngles) -> None:
        """Resample a 300 ms window and publish the throttled predicted label."""
        self.have_arm_angles = True
        self._update_status()
        try:
            pred_label = self.model.push(msg.angle_deg, time.monotonic())
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
    """Run the left/right direction intent estimator node."""
    rclpy.init()
    node = LeftRightIntentEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

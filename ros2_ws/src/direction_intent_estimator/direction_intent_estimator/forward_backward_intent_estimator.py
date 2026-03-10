#!/usr/bin/env python3
"""ROS 2 node for the 012 forward/backward direction intent model."""

from __future__ import annotations

from pathlib import Path
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from go2_msgs.msg import QDq
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32
from unitree_go.msg import LowState

from direction_intent_estimator.model_runtime import (
    SlidingWindowIntentModel,
    first_existing_path,
)


def _default_model_dir(bundle_name: str) -> Path:
    """Resolve the first usable model bundle path across install and source layouts."""
    module_path = Path(__file__).resolve()
    share_dir = Path(get_package_share_directory("direction_intent_estimator"))
    return first_existing_path(
        [
            share_dir / "config" / "models" / bundle_name,
            module_path.parents[1] / "config" / "models" / bundle_name,
            module_path.parents[3] / "src" / "direction_intent_estimator" / "config" / "models" / bundle_name,
        ]
    )


class ForwardBackwardIntentEstimatorNode(Node):
    """Run the 012 model from foot force, IMU acceleration, and qdq_est."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2

    def __init__(self) -> None:
        super().__init__("forward_backward_intent_estimator")

        default_model_dir = _default_model_dir("012")

        self.declare_parameter("model_dir", str(default_model_dir))
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("qdq_topic", "/qdq_est")
        self.declare_parameter("output_topic", "/direction_intent/forward_backward")
        self.declare_parameter("onnx_intra_threads", 1)
        self.declare_parameter("onnx_inter_threads", 1)
        self.declare_parameter("status_topic", "/status/intent_estimator/forward_backward")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("sliding_window_ms", 300.0)
        self.declare_parameter("sampling_hz", 200.0)
        self.declare_parameter("publish_hz", 10.0)

        self.model_dir = Path(str(self.get_parameter("model_dir").value)).expanduser()
        self.lowstate_topic = str(self.get_parameter("lowstate_topic").value)
        self.qdq_topic = str(self.get_parameter("qdq_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.onnx_intra_threads = int(self.get_parameter("onnx_intra_threads").value)
        self.onnx_inter_threads = int(self.get_parameter("onnx_inter_threads").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_hz = float(self.get_parameter("status_hz").value)
        self.sliding_window_ms = max(1.0, float(self.get_parameter("sliding_window_ms").value))
        self.sampling_hz = max(1.0, float(self.get_parameter("sampling_hz").value))
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))

        self.model = SlidingWindowIntentModel(
            logger=self.get_logger(),
            model_dir=self.model_dir,
            onnx_intra_threads=self.onnx_intra_threads,
            onnx_inter_threads=self.onnx_inter_threads,
            sliding_window_ms=self.sliding_window_ms,
            sampling_hz=self.sampling_hz,
            publish_hz=self.publish_hz,
        )
        self.last_qdq: QDq | None = None
        self.have_lowstate = False
        self.have_qdq = False
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
        self.sub_qdq = self.create_subscription(
            QDq, self.qdq_topic, self.on_qdq, sensor_qos
        )
        self.pub_intent = self.create_publisher(Int32, self.output_topic, 10)
        self.pub_status = self.create_publisher(Int32, self.status_topic, status_qos)
        self.status_timer = self.create_timer(1.0 / max(1.0, self.status_hz), self.on_status_timer)
        self._update_status()

        self.get_logger().info(
            "forward_backward_intent_estimator ready: "
            f"{self.lowstate_topic} + {self.qdq_topic} -> {self.output_topic}, "
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
        self.pub_status.publish(Int32(data=int(self.status_code)))

    def _update_status(self) -> None:
        missing_topics = []
        if not self.have_lowstate:
            missing_topics.append(self.lowstate_topic)
        if not self.have_qdq:
            missing_topics.append(self.qdq_topic)

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

    def on_qdq(self, msg: QDq) -> None:
        """Cache the latest qdq_est message for the next lowstate sample."""
        self.last_qdq = msg
        self.have_qdq = True
        self._update_status()

    def on_lowstate(self, msg: LowState) -> None:
        """Build a 19-feature sample and publish the throttled predicted label."""
        self.have_lowstate = True
        self._update_status()
        if self.last_qdq is None:
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

        joint_dq = [float(x) for x in self.last_qdq.dq[6:18]]
        if len(joint_dq) != 12:
            self.get_logger().error(
                f"Expected 12 joint dq values from /qdq_est, got {len(joint_dq)}."
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

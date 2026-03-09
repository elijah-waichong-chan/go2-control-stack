#!/usr/bin/env python3
"""ROS 2 node for the 034 left/right direction intent model."""

from __future__ import annotations

from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from go2_msgs.msg import ArmAngles
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32

from direction_intent_estimator.model_runtime import SlidingWindowIntentModel


class LeftRightIntentEstimatorNode(Node):
    """Subscribe to arm angles, run ONNX inference, and publish intent labels."""

    def __init__(self) -> None:
        super().__init__("left_right_intent_estimator")

        share_dir = Path(get_package_share_directory("direction_intent_estimator"))
        default_model_dir = share_dir / "config" / "models" / "034"

        self.declare_parameter("model_dir", str(default_model_dir))
        self.declare_parameter("arm_angles_topic", "/arm_angles")
        self.declare_parameter("output_topic", "/direction_intent/left_right")
        self.declare_parameter("onnx_intra_threads", 1)
        self.declare_parameter("onnx_inter_threads", 1)

        self.model_dir = Path(str(self.get_parameter("model_dir").value)).expanduser()
        self.arm_angles_topic = str(self.get_parameter("arm_angles_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.onnx_intra_threads = int(self.get_parameter("onnx_intra_threads").value)
        self.onnx_inter_threads = int(self.get_parameter("onnx_inter_threads").value)
        self.model = SlidingWindowIntentModel(
            logger=self.get_logger(),
            model_dir=self.model_dir,
            onnx_intra_threads=self.onnx_intra_threads,
            onnx_inter_threads=self.onnx_inter_threads,
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_arm_angles = self.create_subscription(
            ArmAngles, self.arm_angles_topic, self.on_arm_angles, qos
        )
        self.pub_intent = self.create_publisher(Int32, self.output_topic, qos)

        self.get_logger().info(
            "left_right_intent_estimator ready: "
            f"{self.arm_angles_topic} -> {self.output_topic}, "
            f"model={self.model.metadata.model_path}, "
            f"input={self.model.input_name}[batch,"
            f"{self.model.metadata.num_features},"
            f"{self.model.metadata.num_timesteps}], "
            f"output={self.model.output_name}[batch,{self.model.metadata.output_dim}], "
            f"labels={self.model.metadata.index_to_label}"
        )

    def on_arm_angles(self, msg: ArmAngles) -> None:
        """Accumulate a 60-step history window and publish the predicted label."""
        try:
            pred_label = self.model.push(msg.angle_deg)
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

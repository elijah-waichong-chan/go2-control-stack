#!/usr/bin/env python3
"""ROS 2 node for the 012 forward/backward direction intent model."""

from __future__ import annotations

from collections import deque
import math
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
    build_selected_feature_vector,
    first_existing_path,
    load_deploy_cfg,
    load_selected_features,
    resolve_deploy_path,
)


def _default_model_dir(bundle_name: str) -> Path:
    """Resolve the first usable model bundle path across install and source layouts."""
    module_path = Path(__file__).resolve()
    share_dir = Path(get_package_share_directory("intent_estimator"))
    return first_existing_path(
        [
            module_path.parents[1] / "config" / "models" / bundle_name,
            module_path.parents[3] / "core" / "intent_estimator" / "config" / "models" / bundle_name,
            share_dir / "config" / "models" / bundle_name,
        ]
    )


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


class ForwardBackwardIntentEstimatorNode(Node):
    """Run the 012 model from arm angles and arm currents."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    TRANSITION_CONFIRMATIONS = 2

    def __init__(self) -> None:
        super().__init__("forward_backward_intent_estimator")

        default_model_dir = _default_model_dir("012")

        self.model_dir = default_model_dir.expanduser()
        self.arm_angles_topic = "/arm_angles"
        self.output_topic = "/direction_intent/forward_backward"
        self.status_topic = "/status/intent_estimator/forward_backward"
        self.status_hz = 10.0
        self.sliding_window_ms = 300.0
        self.sampling_hz = 200.0
        self.publish_hz = 10.0
        self.loop_budget_ms = 1000.0 / max(self.publish_hz, 1.0)
        self.loop_stats_window = max(100, int(self.publish_hz * 10.0))

        deploy_cfg = load_deploy_cfg(resolve_deploy_path(self.model_dir))
        self.selected_features = load_selected_features(deploy_cfg)

        self.model = SlidingWindowIntentModel(
            logger=self.get_logger(),
            model_dir=self.model_dir,
            sliding_window_ms=self.sliding_window_ms,
            sampling_hz=self.sampling_hz,
            publish_hz=self.publish_hz,
        )
        selected_width = sum(feature.width for feature in self.selected_features)
        if selected_width != self.model.metadata.input_num_features:
            raise RuntimeError(
                "Selected feature widths sum to %d, but deploy preprocessing expects %d raw features."
                % (selected_width, self.model.metadata.input_num_features)
            )
        self.have_arm_angles = False
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.filtered_label = 0
        self.pending_label: int | None = None
        self.pending_count = 0
        self.loop_times_ms: deque[float] = deque(maxlen=self.loop_stats_window)
        self.deadline_flags: deque[int] = deque(maxlen=self.loop_stats_window)

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
            "forward_backward_intent_estimator ready: "
            f"{self.arm_angles_topic} -> {self.output_topic}, "
            f"status={self.status_topic}, "
            f"model={self.model.metadata.model_path}, "
            f"window={self.sliding_window_ms:.0f}ms@{self.sampling_hz:.0f}Hz, "
            f"publish={self.publish_hz:.1f}Hz, "
            f"transition_confirmations={self.TRANSITION_CONFIRMATIONS}, "
            f"raw_input={self.model.metadata.input_num_features}, "
            f"input={self.model.input_name}[batch,"
            f"{self.model.metadata.num_features},"
            f"{self.model.metadata.num_timesteps}], "
            f"output={self.model.output_name}[batch,{self.model.metadata.output_dim}], "
            f"labels={self.model.metadata.index_to_label}, "
            f"features={[feature.name for feature in self.selected_features]}"
        )

    def _filter_transition(self, candidate_label: int) -> int:
        candidate = int(candidate_label)
        if candidate == self.filtered_label:
            self.pending_label = None
            self.pending_count = 0
            return self.filtered_label

        if candidate == self.pending_label:
            self.pending_count += 1
        else:
            self.pending_label = candidate
            self.pending_count = 1

        if self.pending_count >= self.TRANSITION_CONFIRMATIONS:
            self.filtered_label = candidate
            self.pending_label = None
            self.pending_count = 0

        return self.filtered_label

    def _set_status(self, status_code: int) -> None:
        status_code = int(status_code)
        if status_code == self.status_code:
            return
        self.status_code = status_code
        self._publish_status()

    def on_status_timer(self) -> None:
        self._publish_status()

    def _record_loop_time_ms(self, loop_time_ms: float) -> None:
        self.loop_times_ms.append(float(loop_time_ms))
        self.deadline_flags.append(1 if loop_time_ms > self.loop_budget_ms else 0)

    def _publish_status(self) -> None:
        if not self.loop_times_ms:
            self.pub_status.publish(
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
        self.pub_status.publish(
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

    def _update_status(self) -> None:
        if not self.have_arm_angles:
            if self.waiting_on != self.arm_angles_topic:
                self.get_logger().info(
                    f"forward_backward_intent_estimator waiting for {self.arm_angles_topic}..."
                )
                self.waiting_on = self.arm_angles_topic
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("forward_backward_intent_estimator running")
            self.running_logged = True

    def on_arm_angles(self, msg: ArmAngles) -> None:
        """Build a deploy-configured feature vector and publish the predicted label."""
        start_ns = time.perf_counter_ns()
        try:
            self.have_arm_angles = True
            self._update_status()
            source_vectors = {
                "arm_angles": msg.angle_deg,
                "arm_currents": msg.current,
            }
            features = build_selected_feature_vector(self.selected_features, source_vectors)
            pred_label = self.model.push(features, time.monotonic())
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime inference errors
            self.get_logger().error(f"ONNX inference failed: {exc}")
            return
        finally:
            loop_time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._record_loop_time_ms(loop_time_ms)

        if pred_label is None:
            return

        filtered_label = self._filter_transition(int(pred_label))
        out = Int32()
        out.data = filtered_label
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

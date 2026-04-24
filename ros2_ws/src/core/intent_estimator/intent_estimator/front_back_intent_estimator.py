#!/usr/bin/env python3
"""ROS 2 node for heuristic front/back direction intent estimation."""

from __future__ import annotations

from collections import deque
import math
import time

import rclpy
from hq_pcot_msgs.msg import ArmState, LoopStatus
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32


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


class FrontBackIntentEstimatorNode(Node):
    """Publish front/back intent labels from joint0 arm-angle thresholds."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    TRANSITION_CONFIRMATIONS = 2

    def __init__(self) -> None:
        super().__init__("front_back_intent_estimator")

        self.arm_state_topic = "/arm/state"
        self.output_topic = "/direction_intent/front_back/label"
        self.status_topic = "/status/intent_estimator/front_back"
        self.status_hz = 10.0
        self.expected_input_hz = 50.0
        self.forward_min_deg = 30.0
        self.forward_max_deg = 80.0
        self.backward_min_deg = 100.0
        self.backward_max_deg = 150.0
        self.idle_label = 0
        self.forward_label = 1
        self.backward_label = 2
        self.loop_budget_ms = 1000.0 / max(self.expected_input_hz, 1.0)
        self.loop_stats_window = max(100, int(self.expected_input_hz * 10.0))

        self.have_arm_angles = False
        self.status_code = self.STATUS_WAITING_FOR_TOPICS
        self.waiting_on: str | None = None
        self.running_logged = False
        self.filtered_label = self.idle_label
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
            ArmState, self.arm_state_topic, self.on_arm_angles, sensor_qos
        )
        self.pub_intent = self.create_publisher(Int32, self.output_topic, 10)
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)
        self.status_timer = self.create_timer(
            1.0 / max(1.0, self.status_hz), self.on_status_timer
        )
        self._update_status()

        self.get_logger().info(
            "front_back_intent_estimator ready: "
            f"{self.arm_state_topic} -> {self.output_topic}, "
            f"status={self.status_topic}, "
            f"forward=[{self.forward_min_deg:.1f}, {self.forward_max_deg:.1f}]deg -> {self.forward_label}, "
            f"backward=[{self.backward_min_deg:.1f}, {self.backward_max_deg:.1f}]deg -> {self.backward_label}, "
            f"idle={self.idle_label}, "
            f"transition_confirmations={self.TRANSITION_CONFIRMATIONS}"
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
            if self.waiting_on != self.arm_state_topic:
                self.get_logger().info(
                    f"front_back_intent_estimator waiting for {self.arm_state_topic}..."
                )
                self.waiting_on = self.arm_state_topic
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("front_back_intent_estimator running")
            self.running_logged = True

    def _predict_label(self, joint0_deg: float) -> int:
        if self.forward_min_deg <= joint0_deg <= self.forward_max_deg:
            return self.forward_label
        if self.backward_min_deg <= joint0_deg <= self.backward_max_deg:
            return self.backward_label
        return self.idle_label

    def on_arm_angles(self, msg: ArmState) -> None:
        """Publish heuristic front/back intent labels from the first arm joint."""
        start_ns = time.perf_counter_ns()
        try:
            if len(msg.angle_deg) == 0:
                raise ValueError("Expected /arm/state to include at least one angle.")
            self.have_arm_angles = True
            self._update_status()
            pred_label = self._predict_label(float(msg.angle_deg[0]))
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        finally:
            loop_time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._record_loop_time_ms(loop_time_ms)

        filtered_label = self._filter_transition(pred_label)
        out = Int32()
        out.data = filtered_label
        self.pub_intent.publish(out)


def main() -> None:
    """Run the heuristic front/back intent estimator node."""
    rclpy.init()
    node = FrontBackIntentEstimatorNode()
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

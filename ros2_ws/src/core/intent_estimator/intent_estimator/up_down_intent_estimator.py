#!/usr/bin/env python3
"""ROS 2 node for the 056 up/down direction intent model."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from hq_pcot_msgs.msg import ArmState, LoopStatus
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32
from unitree_go.msg import LowState

from intent_estimator.onnx_runtime import (
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
            module_path.parents[3]
            / "core"
            / "intent_estimator"
            / "config"
            / "models"
            / bundle_name,
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


class UpDownIntentEstimatorNode(Node):
    """Run the 056 model from the features declared in its deploy bundle."""

    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_TOPICS = 2
    TRANSITION_CONFIRMATIONS = 3
    ZERO_TRANSITION_CONFIRMATIONS = 1
    SDK_TO_URDF_LEG_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def __init__(self) -> None:
        super().__init__("up_down_intent_estimator")

        default_model_dir = _default_model_dir("056")

        self.model_dir = default_model_dir.expanduser()
        self.lowstate_topic = "/lowstate"
        self.arm_state_topic = "/arm/state"
        self.output_topic = "/direction_intent/up_down"
        self.raw_output_topic = "/direction_intent/up_down/raw"
        self.status_topic = "/status/intent_estimator/up_down"
        self.status_hz = 10.0
        self.publish_hz = 10.0

        deploy_cfg = load_deploy_cfg(resolve_deploy_path(self.model_dir))
        self.selected_features = load_selected_features(deploy_cfg)
        deploy_model_cfg = deploy_cfg.get("model", {})
        self.sampling_hz = float(deploy_model_cfg.get("sampling_hz", 200.0))
        self.sliding_window_ms = float(
            deploy_model_cfg.get(
                "window_ms",
                1000.0 * float(deploy_model_cfg.get("num_timesteps", 0)) / self.sampling_hz,
            )
        )
        if self.sliding_window_ms <= 0.0:
            raise RuntimeError(
                "deploy.yaml must declare a positive model.window_ms or model.num_timesteps."
            )
        self.loop_budget_ms = 1000.0 / max(self.publish_hz, 1.0)
        self.loop_stats_window = max(100, int(self.publish_hz * 10.0))

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
        self.selected_feature_names = {feature.name for feature in self.selected_features}
        self.requires_lowstate = bool(
            self.selected_feature_names.intersection({"ff", "accel", "dq"})
        )
        self.requires_arm_angles = bool(
            self.selected_feature_names.intersection({"arm_angles", "arm_currents"})
        )
        self.input_topics: list[str] = []
        if self.requires_lowstate:
            self.input_topics.append(self.lowstate_topic)
        if self.requires_arm_angles:
            self.input_topics.append(self.arm_state_topic)
        self.have_lowstate = False
        self.have_arm_angles = False
        self.latest_arm_angles: list[float] = [0.0] * 7
        self.latest_arm_currents: list[float] = [0.0] * 7
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

        self.sub_lowstate = None
        if self.requires_lowstate:
            self.sub_lowstate = self.create_subscription(
                LowState, self.lowstate_topic, self.on_lowstate, sensor_qos
            )
        self.sub_arm_angles = None
        if self.requires_arm_angles:
            self.sub_arm_angles = self.create_subscription(
                ArmState, self.arm_state_topic, self.on_arm_angles, sensor_qos
            )
        self.pub_raw_intent = self.create_publisher(Int32, self.raw_output_topic, 10)
        self.pub_intent = self.create_publisher(Int32, self.output_topic, 10)
        self.pub_status = self.create_publisher(LoopStatus, self.status_topic, status_qos)
        self.status_timer = self.create_timer(
            1.0 / max(1.0, self.status_hz), self.on_status_timer
        )
        self._update_status()

        input_topics = " + ".join(self.input_topics) if self.input_topics else "(none)"
        self.get_logger().info(
            "up_down_intent_estimator ready: "
            f"{input_topics} -> {self.raw_output_topic} (raw), {self.output_topic} (filtered), "
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

        required_confirmations = self.TRANSITION_CONFIRMATIONS
        if candidate == 0 and self.filtered_label in (5, 6):
            required_confirmations = self.ZERO_TRANSITION_CONFIRMATIONS

        if self.pending_count >= required_confirmations:
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
        missing_topics = self._missing_topics()
        if missing_topics:
            waiting_topics = ", ".join(missing_topics)
            if self.waiting_on != waiting_topics:
                self.get_logger().info(
                    f"up_down_intent_estimator waiting for {waiting_topics}..."
                )
                self.waiting_on = waiting_topics
            self._set_status(self.STATUS_WAITING_FOR_TOPICS)
            return

        self.waiting_on = None
        self._set_status(self.STATUS_RUNNING)
        if not self.running_logged:
            self.get_logger().info("up_down_intent_estimator running")
            self.running_logged = True

    def _missing_topics(self) -> list[str]:
        missing_topics: list[str] = []
        if self.requires_lowstate and not self.have_lowstate:
            missing_topics.append(self.lowstate_topic)
        if self.requires_arm_angles and not self.have_arm_angles:
            missing_topics.append(self.arm_state_topic)
        return missing_topics

    def _build_source_vectors(self, lowstate_msg: LowState | None) -> dict[str, object]:
        source_vectors: dict[str, object] = {}
        if lowstate_msg is not None:
            leg_dq = [
                float(lowstate_msg.motor_state[index_sdk].dq)
                for index_sdk in self.SDK_TO_URDF_LEG_INDEX
            ]
            source_vectors.update(
                {
                    "ff": [float(value) for value in lowstate_msg.foot_force],
                    "accel": [
                        float(value) for value in lowstate_msg.imu_state.accelerometer
                    ],
                    "dq": leg_dq,
                }
            )
        if self.requires_arm_angles:
            source_vectors["arm_angles"] = self.latest_arm_angles
            source_vectors["arm_currents"] = self.latest_arm_currents
        for feature in self.selected_features:
            if feature.name in source_vectors:
                continue
            raise ValueError(f"Unsupported up_down deploy feature: {feature.name}")
        return source_vectors

    def on_arm_angles(self, msg: ArmState) -> None:
        self.latest_arm_angles = [float(value) for value in msg.angle_deg]
        self.latest_arm_currents = [float(value) for value in msg.current]
        self.have_arm_angles = True
        self._update_status()
        self._maybe_run_inference(None)

    def on_lowstate(self, msg: LowState) -> None:
        self.have_lowstate = True
        self._update_status()
        self._maybe_run_inference(msg)

    def _maybe_run_inference(self, lowstate_msg: LowState | None) -> None:
        """Build a deploy-configured feature vector and publish the predicted label."""
        start_ns = time.perf_counter_ns()
        try:
            if self.requires_lowstate and lowstate_msg is None:
                return
            if self.requires_lowstate and len(lowstate_msg.foot_force) < 4:
                raise ValueError(
                    "Expected /lowstate foot_force to have length >= 4, got "
                    f"{len(lowstate_msg.foot_force)}"
                )
            max_index = max(self.SDK_TO_URDF_LEG_INDEX)
            if self.requires_lowstate and len(lowstate_msg.motor_state) <= max_index:
                raise ValueError(
                    "Expected /lowstate motor_state to include at least "
                    f"{max_index + 1} entries, got {len(lowstate_msg.motor_state)}"
                )
            if self.requires_arm_angles and len(self.latest_arm_angles) != 7:
                raise ValueError(
                    "Expected latest /arm/state angle_deg to have length 7, got "
                    f"{len(self.latest_arm_angles)}"
                )
            if self.requires_arm_angles and len(self.latest_arm_currents) != 7:
                raise ValueError(
                    "Expected latest /arm/state current to have length 7, got "
                    f"{len(self.latest_arm_currents)}"
                )
            if self._missing_topics():
                return
            source_vectors = self._build_source_vectors(lowstate_msg)
            features = build_selected_feature_vector(
                self.selected_features, source_vectors
            )
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

        raw_out = Int32()
        raw_out.data = int(pred_label)
        self.pub_raw_intent.publish(raw_out)

        filtered_label = self._filter_transition(int(pred_label))
        out = Int32()
        out.data = filtered_label
        self.pub_intent.publish(out)


def main() -> None:
    """Run the up/down direction intent estimator node."""
    rclpy.init()
    node = UpDownIntentEstimatorNode()
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

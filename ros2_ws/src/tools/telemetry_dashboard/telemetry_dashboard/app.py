#!/usr/bin/env python3

from collections import deque
import os
import threading
import time
from typing import Deque, Dict, Tuple

import streamlit as st

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from hq_pcot_msgs.msg import ArmCommand, ArmState, LocomotionCmd, LoopStatus, QDq
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage
from unitree_go.msg import LowCmd, LowState
from unitree_arm.msg import ArmString
from telemetry_dashboard import launch_process_manager


_ROS_RUNTIME_LOCK = threading.Lock()
_PRIMARY_ROS_NODE_NAME = f"telemetry_dashboard_{os.getpid()}"
_INTENT_ESTIMATORS = {
    "front_back": {
        "label": "Front/Back Estimator",
        "process_name": "front_back_estimator",
        "executable": "front_back_intent_estimator",
        "status_key": "intent_estimator_front_back",
        "status_topic": "/status/intent_estimator/front_back",
    },
    "left_right": {
        "label": "Left/Right Estimator",
        "process_name": "left_right_estimator",
        "executable": "left_right_intent_estimator",
        "status_key": "intent_estimator_left_right",
        "status_topic": "/status/intent_estimator/left_right",
    },
    "up_down": {
        "label": "Up/Down Estimator",
        "process_name": "up_down_estimator",
        "executable": "up_down_intent_estimator",
        "status_key": "intent_estimator_up_down",
        "status_topic": "/status/intent_estimator/up_down",
    },
}


@st.cache_resource(show_spinner=False)
def _get_ros_runtimes() -> Dict[
    str,
    Tuple["TelemetryNode", rclpy.executors.SingleThreadedExecutor, threading.Thread],
]:
    # Keep the ROS runtime registry alive across Streamlit reruns in the same process.
    return {}


class TelemetryNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.status_timeout_s = float(self.declare_parameter("status_timeout_s", 3.0).value)
        self.graph_poll_hz = float(self.declare_parameter("graph_poll_hz", 1.0).value)
        self.topic_rate_window_s = 1.0

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self._lock = threading.Lock()

        self._status: Dict[str, Tuple[object, float]] = {}
        self._topic_available: Dict[str, bool] = {}
        self._topic_timestamps: Dict[str, Deque[float]] = {}
        self._topic_latest_msg: Dict[str, str] = {}
        self._node_names: list[str] = []
        self._topic_names = {
            "qdq_est": "/qdq_est",
            "imu": "/imu",
            "odometry_filtered": "/odometry/filtered",
            "lowstate": "/lowstate",
            "locomotion_cmd": "/locomotion_cmd",
            "lowcmd": "/lowcmd",
            "joint_states": "/joint_states",
            "tf": "/tf",
            "arm_state": "/arm/state",
            "arm_command_state": "/arm/commanded_angles",
            "arm_feedback": "/arm_Feedback",
            "arm_command": "/arm_Command",
            "intent_forward_backward": "/direction_intent/front_back/label",
            "intent_left_right": "/direction_intent/left_right/label",
            "intent_up_down": "/direction_intent/up_down/label",
        }

        self.create_subscription(QDq, self._topic_names["qdq_est"], self.on_qdq_est, qos)
        self.create_subscription(Imu, self._topic_names["imu"], self.on_imu, qos)
        self.create_subscription(
            Odometry, self._topic_names["odometry_filtered"], self.on_odometry_filtered, qos
        )
        self.create_subscription(LowState, self._topic_names["lowstate"], self.on_lowstate, qos)
        self.create_subscription(
            LocomotionCmd, self._topic_names["locomotion_cmd"], self.on_locomotion_cmd, qos
        )
        self.create_subscription(LowCmd, self._topic_names["lowcmd"], self.on_lowcmd, qos)
        self.create_subscription(
            JointState, self._topic_names["joint_states"], self.on_joint_states, qos
        )
        self.create_subscription(
            TFMessage, self._topic_names["tf"], self.on_tf, qos
        )
        self.create_subscription(ArmState, self._topic_names["arm_state"], self.on_arm_angles, qos)
        self.create_subscription(
            ArmCommand,
            self._topic_names["arm_command_state"],
            self.on_arm_command_state,
            qos,
        )
        self.create_subscription(
            ArmString, self._topic_names["arm_feedback"], self.on_arm_feedback, qos
        )
        self.create_subscription(
            ArmString, self._topic_names["arm_command"], self.on_arm_command, qos
        )
        self.create_subscription(
            Int32,
            self._topic_names["intent_forward_backward"],
            lambda m: self.on_intent("intent_forward_backward", m),
            qos,
        )
        self.create_subscription(
            Int32,
            self._topic_names["intent_left_right"],
            lambda m: self.on_intent("intent_left_right", m),
            qos,
        )
        self.create_subscription(
            Int32,
            self._topic_names["intent_up_down"],
            lambda m: self.on_intent("intent_up_down", m),
            qos,
        )

        self.create_subscription(
            LoopStatus,
            "/status/state_estimator",
            lambda m: self.on_loop_status("state_estimator", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/loco_ctrl",
            lambda m: self.on_loop_status("loco_ctrl", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/arm_parser",
            lambda m: self.on_loop_status("arm_parser", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/standing_init",
            lambda m: self.on_loop_status("standing_init", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/intent_estimator/front_back",
            lambda m: self.on_loop_status("intent_estimator_front_back", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/intent_estimator/left_right",
            lambda m: self.on_loop_status("intent_estimator_left_right", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/intent_estimator/up_down",
            lambda m: self.on_loop_status("intent_estimator_up_down", m),
            status_qos,
        )
        self._graph_timer = self.create_timer(
            1.0 / max(self.graph_poll_hz, 0.1),
            self._on_graph_timer,
        )

    def on_status_value(self, name: str, value: object) -> None:
        with self._lock:
            self._status[name] = (value, time.monotonic())

    def on_loop_status(self, name: str, msg: LoopStatus) -> None:
        self.on_status_value(
            name,
            {
                "status": int(msg.status),
                "avg_loop_ms": float(msg.avg_loop_ms),
                "p99_loop_ms": float(msg.p99_loop_ms),
                "max_loop_ms": float(msg.max_loop_ms),
                "budget_ms": float(msg.budget_ms),
                "deadline_miss_count": int(msg.deadline_miss_count),
                "sample_count": int(msg.sample_count),
            },
        )

    def on_qdq_est(self, msg: QDq) -> None:
        self._mark_topic("qdq_est")

    def on_imu(self, msg: Imu) -> None:
        self._mark_topic("imu")

    def on_odometry_filtered(self, msg: Odometry) -> None:
        self._mark_topic("odometry_filtered")

    def on_lowstate(self, msg: LowState) -> None:
        self._mark_topic("lowstate")

    def on_locomotion_cmd(self, msg: LocomotionCmd) -> None:
        self._mark_topic("locomotion_cmd")

    def on_lowcmd(self, msg: LowCmd) -> None:
        self._mark_topic("lowcmd")

    def on_joint_states(self, msg: JointState) -> None:
        self._mark_topic("joint_states")

    def on_tf(self, msg: TFMessage) -> None:
        self._mark_topic("tf")

    def on_arm_angles(self, msg: ArmState) -> None:
        self._mark_topic("arm_state")

    def on_arm_command_state(self, msg: ArmCommand) -> None:
        self._mark_topic("arm_command_state")

    def on_arm_feedback(self, msg: ArmString) -> None:
        self._mark_topic("arm_feedback")

    def on_arm_command(self, msg: ArmString) -> None:
        self._mark_topic("arm_command")

    def _mark_topic(self, key: str) -> None:
        with self._lock:
            t = time.monotonic()
            self._status[key] = (True, t)
            self._update_topic_rate(key, t)

    def on_intent(self, key: str, msg: Int32) -> None:
        with self._lock:
            t = time.monotonic()
            self._status[key] = (True, t)
            self._update_topic_rate(key, t)
            self._topic_latest_msg[key] = str(int(msg.data))

    def _update_topic_rate(self, key: str, t: float) -> None:
        samples = self._topic_timestamps.setdefault(key, deque())
        samples.append(t)
        cutoff = t - self.topic_rate_window_s
        while samples and samples[0] < cutoff:
            samples.popleft()

    def _snapshot_topic_rates_locked(self, now: float) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        cutoff = now - self.topic_rate_window_s
        for key, samples in self._topic_timestamps.items():
            while samples and samples[0] < cutoff:
                samples.popleft()
            if samples:
                rates[key] = len(samples) / self.topic_rate_window_s
        return rates

    def _update_topic_availability(self) -> None:
        try:
            topics = self.get_topic_names_and_types()
            available = {name for name, _types in topics}
            with self._lock:
                for _key, name in self._topic_names.items():
                    self._topic_available[name] = name in available
        except Exception:
            with self._lock:
                for _key, name in self._topic_names.items():
                    self._topic_available[name] = False

    def _update_node_list(self) -> None:
        try:
            nodes = self.get_node_names_and_namespaces()
            formatted: list[str] = []
            for name, namespace in nodes:
                if namespace in ("", "/"):
                    formatted.append(f"/{name}")
                else:
                    formatted.append(f"{namespace.rstrip('/')}/{name}")
            with self._lock:
                self._node_names = sorted(formatted)
        except Exception:
            with self._lock:
                self._node_names = []

    def _on_graph_timer(self) -> None:
        self._update_topic_availability()
        self._update_node_list()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = time.monotonic()
            return {
                "status": dict(self._status),
                "topic_available": dict(self._topic_available),
                "topic_rate": self._snapshot_topic_rates_locked(now),
                "topic_latest_msg": dict(self._topic_latest_msg),
                "topic_names": dict(self._topic_names),
                "node_names": list(self._node_names),
                "status_timeout_s": self.status_timeout_s,
            }

    def destroy_node(self) -> bool:
        return super().destroy_node()


def _register_ros_runtime(
    node_name: str,
    node: "TelemetryNode",
    executor: rclpy.executors.SingleThreadedExecutor,
    thread: threading.Thread,
) -> None:
    with _ROS_RUNTIME_LOCK:
        _get_ros_runtimes()[node_name] = (node, executor, thread)


def _get_registered_ros_runtime(
    node_name: str,
) -> Tuple["TelemetryNode", rclpy.executors.SingleThreadedExecutor, threading.Thread] | None:
    with _ROS_RUNTIME_LOCK:
        return _get_ros_runtimes().get(node_name)


def _shutdown_ros_runtime(node_name: str) -> bool:
    with _ROS_RUNTIME_LOCK:
        runtime = _get_ros_runtimes().pop(node_name, None)

    if runtime is None:
        return False

    node, executor, thread = runtime
    try:
        executor.shutdown(timeout_sec=1.0)
    except TypeError:
        executor.shutdown()
    except Exception:
        pass

    try:
        executor.remove_node(node)
    except Exception:
        pass

    try:
        node.destroy_node()
    except Exception:
        pass

    if thread.is_alive():
        thread.join(timeout=1.0)

    return True


def _shutdown_other_dashboard_nodes(current_node_name: str) -> int:
    with _ROS_RUNTIME_LOCK:
        other_node_names = [
            node_name for node_name in _get_ros_runtimes() if node_name != current_node_name
        ]

    stopped = 0
    for node_name in other_node_names:
        if _shutdown_ros_runtime(node_name):
            stopped += 1
    return stopped


def _normalize_ros_graph_node_name(node_name: str) -> str:
    return node_name.strip().lstrip("/")


def _collect_other_visible_ros_nodes(
    snapshot: Dict[str, object],
    current_node_name: str,
) -> list[str]:
    current = _normalize_ros_graph_node_name(current_node_name)
    other_nodes: list[str] = []
    seen: set[str] = set()
    for node_name in snapshot.get("node_names", []):
        normalized = _normalize_ros_graph_node_name(str(node_name))
        if not normalized or normalized == current or normalized in seen:
            continue
        seen.add(normalized)
        other_nodes.append(normalized)
    return other_nodes


def get_ros_node() -> TelemetryNode:
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
    except RuntimeError:
        # rclpy may already be initialized in this process
        pass

    node_name = _PRIMARY_ROS_NODE_NAME
    st.session_state["ros_node_name"] = node_name
    runtime = _get_registered_ros_runtime(node_name)
    if runtime is not None:
        node, executor, thread = runtime
        st.session_state["ros_executor"] = executor
        st.session_state["ros_spin_thread"] = thread
        st.session_state["ros_node"] = node
        return node

    node = TelemetryNode(node_name)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    _register_ros_runtime(node_name, node, executor, thread)
    st.session_state["ros_executor"] = executor
    st.session_state["ros_spin_thread"] = thread
    st.session_state["ros_node"] = node
    return node


def _get_fresh_status(status_map: Dict[str, Tuple[object, float]], key: str, now: float, timeout: float):
    entry = _get_fresh_status_entry(status_map, key, now, timeout)
    if entry is None:
        return None
    val, _timestamp = entry
    return val


def _get_fresh_status_entry(
    status_map: Dict[str, Tuple[object, float]],
    key: str,
    now: float,
    timeout: float,
):
    if key not in status_map:
        return None
    val, t = status_map[key]
    if (now - t) > timeout:
        return None
    return val, t


def _status_code(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("status")
        if raw is None:
            return None
        return int(raw)
    return int(value)


def _get_intent_estimator_status(
    status_map: Dict[str, Tuple[object, float]], now: float, timeout: float
):
    freshest_value = None
    freshest_topic = None
    freshest_timestamp = None
    for config in _INTENT_ESTIMATORS.values():
        entry = _get_fresh_status_entry(status_map, config["status_key"], now, timeout)
        if entry is None:
            continue
        value, timestamp = entry
        if freshest_timestamp is None or timestamp > freshest_timestamp:
            freshest_value = value
            freshest_topic = config["status_topic"]
            freshest_timestamp = timestamp
    return freshest_value, freshest_topic


def _format_timing_ms(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "not available"
    if value_f < 0.0:
        return "not recorded"
    return f"{value_f:.2f} ms"


def _format_count(value: object) -> str:
    try:
        value_i = int(value)
    except (TypeError, ValueError):
        return "not available"
    if value_i < 0:
        return "not recorded"
    return str(value_i)


def _module_summary(key: str, value: object) -> tuple[str, str]:
    if value is None:
        return "error", "not running"

    if key == "standing_init":
        standing_status = _status_code(value)
        if standing_status is None:
            return "error", "invalid status"
        if standing_status == 3:
            return "success", "complete"
        if standing_status == 2:
            return "warning", "waiting for /lowstate"
        if standing_status == 1:
            return "info", "running sequence"
        return "error", f"idle ({standing_status})"

    if key == "arm_parser":
        arm_parser_status = _status_code(value)
        if arm_parser_status is None:
            return "error", "invalid status"
        if arm_parser_status == 1:
            return "success", "publishing"
        if arm_parser_status == 2:
            return "warning", "waiting for /arm_Feedback"
        return "error", f"idle ({arm_parser_status})"

    if key == "rl_controller":
        loco_status = _status_code(value)
        if loco_status is None:
            return "error", "invalid status"
        if loco_status == 1:
            return "success", "running"
        if loco_status == 2:
            return "warning", "waiting for /lowstate"
        if loco_status == 3:
            return "warning", "waiting for standing-init"
        return "error", f"idle ({loco_status})"

    if key == "intent_estimator":
        intent_status = _status_code(value)
        if intent_status is None:
            return "error", "invalid status"
        if intent_status == 1:
            return "success", "running"
        if intent_status == 2:
            return "warning", "waiting for input"
        return "error", f"idle ({intent_status})"

    if bool(value):
        return "success", "running"
    return "error", "not running"


def _render_summary(label: str, severity: str, text: str) -> None:
    message = f"{label}: {text}"
    if severity == "success":
        st.success(message)
    elif severity == "warning":
        st.warning(message)
    elif severity == "info":
        st.info(message)
    else:
        st.error(message)


def _render_loop_status_details(topic: str, value: object) -> None:
    st.caption(f"Topic: `{topic}`")
    if value is None:
        st.caption("No recent status message.")
        return

    raw_status = _status_code(value)
    if raw_status is None:
        st.write("Raw status: unavailable")
    else:
        st.write(f"Raw status: `{raw_status}`")

    if not isinstance(value, dict):
        st.caption("Timing details unavailable.")
        return

    st.write(f"Avg loop: {_format_timing_ms(value.get('avg_loop_ms'))}")
    st.write(f"P99 loop: {_format_timing_ms(value.get('p99_loop_ms'))}")
    st.write(f"Max loop: {_format_timing_ms(value.get('max_loop_ms'))}")
    st.write(f"Budget: {_format_timing_ms(value.get('budget_ms'))}")
    st.write(f"Deadline misses: {_format_count(value.get('deadline_miss_count'))}")
    st.write(f"Sample count: {_format_count(value.get('sample_count'))}")


def render_status(snapshot: Dict[str, object], timeout_s: float) -> None:
    now = time.monotonic()
    timeout = float(timeout_s)
    status_map = snapshot["status"]

    labels = [
        ("standing_init", "Standing Init"),
        ("rl_controller", "RL Controller"),
        ("intent_estimator", "Intent Estimator"),
        ("arm_parser", "Arm Parser"),
    ]

    intent_estimator_value, intent_estimator_topic = _get_intent_estimator_status(
        status_map, now, timeout
    )

    status_view = {
        "standing_init": _get_fresh_status(status_map, "standing_init", now, timeout),
        "arm_parser": _get_fresh_status(status_map, "arm_parser", now, timeout),
        "rl_controller": _get_fresh_status(status_map, "loco_ctrl", now, timeout),
        "intent_estimator": intent_estimator_value,
    }
    status_topics = {
        "standing_init": "/status/standing_init",
        "arm_parser": "/status/arm_parser",
        "rl_controller": "/status/loco_ctrl",
        "intent_estimator": (
            intent_estimator_topic
            or next(iter(_INTENT_ESTIMATORS.values()))["status_topic"]
        ),
    }

    cols = st.columns(len(labels))
    for idx, (key, label) in enumerate(labels):
        with cols[idx]:
            val = status_view.get(key)
            severity, summary = _module_summary(key, val)
            _render_summary(label, severity, summary)

            with st.expander("Details", expanded=False):
                _render_loop_status_details(status_topics[key], val)


def _style_sidebar_buttons() -> None:
    start_green = "#2ecc71"
    start_border = "#27ae60"
    start_orange = "#f39c12"
    start_orange_border = "#d68910"
    stop_red = "#e74c3c"
    stop_border = "#c0392b"
    css_rules = f"""
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {{
            background-color: {start_green};
            color: #ffffff;
            border: 1px solid {start_border};
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {{
            background-color: {start_green};
            color: #ffffff;
            border: 1px solid {start_border};
            filter: brightness(0.96);
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:focus,
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:focus-visible {{
            background-color: {start_green};
            color: #ffffff;
            border: 1px solid {start_border};
            box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.12);
        }}
        [data-testid="stSidebar"] .st-key-toggle_foxglove [data-testid="stButton"] button[kind="secondary"],
        [data-testid="stSidebar"] .st-key-toggle_rosbag [data-testid="stButton"] button[kind="secondary"] {{
            background-color: {start_orange};
            color: #ffffff;
            border: 1px solid {start_orange_border};
        }}
        [data-testid="stSidebar"] .st-key-toggle_foxglove [data-testid="stButton"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] .st-key-toggle_rosbag [data-testid="stButton"] button[kind="secondary"]:hover {{
            background-color: {start_orange};
            color: #ffffff;
            border: 1px solid {start_orange_border};
            filter: brightness(0.96);
        }}
        [data-testid="stSidebar"] .st-key-toggle_foxglove [data-testid="stButton"] button[kind="secondary"]:focus,
        [data-testid="stSidebar"] .st-key-toggle_foxglove [data-testid="stButton"] button[kind="secondary"]:focus-visible,
        [data-testid="stSidebar"] .st-key-toggle_rosbag [data-testid="stButton"] button[kind="secondary"]:focus,
        [data-testid="stSidebar"] .st-key-toggle_rosbag [data-testid="stButton"] button[kind="secondary"]:focus-visible {{
            background-color: {start_orange};
            color: #ffffff;
            border: 1px solid {start_orange_border};
            box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.12);
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
            background-color: {stop_red};
            color: #ffffff;
            border: 1px solid {stop_border};
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {{
            background-color: {stop_red};
            color: #ffffff;
            border: 1px solid {stop_border};
            filter: brightness(0.96);
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:focus,
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:focus-visible {{
            background-color: {stop_red};
            color: #ffffff;
            border: 1px solid {stop_border};
            box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.12);
        }}
        """

    st.markdown(f"<style>{css_rules}</style>", unsafe_allow_html=True)


def _parse_intent_label(raw: object) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _intent_cell_classes(active: bool, active_cls: str, shape_cls: str) -> str:
    classes = ["intent-cell", shape_cls]
    if active:
        classes.append(active_cls)
    else:
        classes.append("intent-muted")
    return " ".join(classes)


def _render_left_right_intent_pad(
    *,
    lr_topic_name: str,
    lr_rate_hz: float | None,
    lr_available: bool,
    lr_fresh: bool,
    lr_label: int | None,
) -> str:
    active_left = active_right = active_center = False
    lr_effective_label = lr_label if lr_available and lr_fresh and lr_label is not None else None

    if lr_effective_label == 3:
        active_left = True
    elif lr_effective_label == 4:
        active_right = True

    active_center = lr_effective_label == 0

    if not lr_available:
        display_state = "Unavailable"
    elif not lr_fresh or lr_label is None:
        display_state = "Waiting"
    elif active_center:
        display_state = "Idle"
    elif active_left:
        display_state = "Left"
    elif active_right:
        display_state = "Right"
    else:
        display_state = f"Label {lr_label}"

    lr_rate_text = f"{lr_rate_hz:.1f} Hz" if lr_rate_hz is not None else "Hz --"

    return (
        "<div class=\"intent-card\">"
        "<div class=\"intent-card-title\">Left/Right Intent</div>"
        f"<div class=\"intent-card-caption\">{lr_topic_name} | {lr_rate_text}</div>"
        "<div class=\"intent-row\">"
        f"<div class=\"{_intent_cell_classes(active_left, 'intent-active-left', 'intent-arrow')}\">◀</div>"
        f"<div class=\"{_intent_cell_classes(active_center, 'intent-active-center', 'intent-center')}\">●</div>"
        f"<div class=\"{_intent_cell_classes(active_right, 'intent-active-right', 'intent-arrow')}\">▶</div>"
        "</div>"
        f"<div class=\"intent-card-state\">{display_state}</div>"
        "</div>"
    )


def _render_front_back_intent_pad(
    *,
    topic_name: str,
    rate_hz: float | None,
    available: bool,
    fresh: bool,
    label: int | None,
) -> str:
    display_state = "Waiting"
    active_forward = active_backward = active_center = False

    if available and fresh and label is not None:
        if label == 1:
            display_state = "Forward"
            active_forward = True
        elif label == 2:
            display_state = "Backward"
            active_backward = True
        elif label == 0:
            display_state = "Idle"
            active_center = True
        else:
            display_state = f"Label {label}"
    elif available:
        display_state = "No fresh intent"

    rate_text = f"{rate_hz:.1f} Hz" if rate_hz is not None else "Hz --"

    return (
        "<div class=\"intent-card\">"
        "<div class=\"intent-card-title\">Front/Back Intent</div>"
        f"<div class=\"intent-card-caption\">{topic_name} | {rate_text}</div>"
        "<div class=\"intent-stack\">"
        f"<div class=\"{_intent_cell_classes(active_forward, 'intent-active-up', 'intent-arrow')}\">▲</div>"
        f"<div class=\"{_intent_cell_classes(active_center, 'intent-active-center', 'intent-center')}\">●</div>"
        f"<div class=\"{_intent_cell_classes(active_backward, 'intent-active-down', 'intent-arrow')}\">▼</div>"
        "</div>"
        f"<div class=\"intent-card-state\">{display_state}</div>"
        "</div>"
    )


def _render_up_down_intent_pad(
    *,
    topic_name: str,
    rate_hz: float | None,
    available: bool,
    fresh: bool,
    label: int | None,
) -> str:
    display_state = "Waiting"
    active_up = active_down = active_center = False

    if available and fresh and label is not None:
        if label == 5:
            display_state = "Up"
            active_up = True
        elif label == 6:
            display_state = "Down"
            active_down = True
        elif label == 0:
            display_state = "Idle"
            active_center = True
        else:
            display_state = f"Label {label}"
    elif available:
        display_state = "No fresh intent"

    rate_text = f"{rate_hz:.1f} Hz" if rate_hz is not None else "Hz --"

    return (
        "<div class=\"intent-card\">"
        "<div class=\"intent-card-title\">Up/Down Intent</div>"
        f"<div class=\"intent-card-caption\">{topic_name} | {rate_text}</div>"
        "<div class=\"intent-stack\">"
        f"<div class=\"{_intent_cell_classes(active_up, 'intent-active-up', 'intent-arrow')}\">▲</div>"
        f"<div class=\"{_intent_cell_classes(active_center, 'intent-active-center', 'intent-center')}\">●</div>"
        f"<div class=\"{_intent_cell_classes(active_down, 'intent-active-down', 'intent-arrow')}\">▼</div>"
        "</div>"
        f"<div class=\"intent-card-state\">{display_state}</div>"
        "</div>"
    )


def _render_sidebar(node: TelemetryNode) -> None:
    snapshot = node.snapshot()
    current_node_name = st.session_state.get("ros_node_name", node.get_name())

    st.header("Settings")
    status_map = snapshot["status"]
    rl_ctrl_status = _get_fresh_status(
        status_map,
        "loco_ctrl",
        time.monotonic(),
        float(snapshot["status_timeout_s"]),
    )
    # Consider control stack active when the RL controller status stream is alive.
    locomotion_active = (
        (_status_code(rl_ctrl_status) is not None)
        or launch_process_manager.is_running("control_stack")
    )
    autonomy_active = launch_process_manager.is_running("autonomy")
    foxglove_active = launch_process_manager.is_running("foxglove_bridge")
    rosbag_active = launch_process_manager.is_running("rosbag_recording")
    active_intent_estimator_modes = [
        mode
        for mode, config in _INTENT_ESTIMATORS.items()
        if launch_process_manager.is_running(config["process_name"])
    ]
    state_converter_active = (
        launch_process_manager.is_running("state_converter_stack")
        or launch_process_manager.is_running("arm_feedback_parser")
    )
    arm_controller_active = launch_process_manager.is_arm_controller_running()
    active_arm_controller_mode = launch_process_manager.get_active_arm_controller_mode()
    control_label = (
        "Stop Locomotion" if locomotion_active else "Start Locomotion"
    )
    enable_wireless_cmd_bridge = st.session_state.get("ctrl_enable_wireless_cmd_bridge", True)
    arm_controller_mode = st.session_state.get(
        "arm_controller_mode",
        "current_ik_ipopt",
    )
    if arm_controller_mode == "current_ik":
        arm_controller_mode = "current_ik_ipopt"
    elif arm_controller_mode == "manual_control_ik":
        arm_controller_mode = "manual_control_ik_ipopt"
    elif arm_controller_mode not in {"current_ik_ipopt", "manual_control_ik_ipopt"}:
        arm_controller_mode = "current_ik_ipopt"
    arm_controller_mode_labels = {
        "current_ik_ipopt": "Current IK (IPOPT)",
        "manual_control_ik_ipopt": "Manual z-ref IK (IPOPT)",
    }
    enable_front_back_estimator = st.session_state.get(
        "ctrl_enable_front_back_estimator", True
    )
    enable_left_right_estimator = st.session_state.get(
        "ctrl_enable_left_right_estimator", False
    )
    enable_up_down_estimator = st.session_state.get(
        "ctrl_enable_up_down_estimator", False
    )
    selected_direction_estimators = []
    if enable_front_back_estimator:
        selected_direction_estimators.append("front_back")
    if enable_left_right_estimator:
        selected_direction_estimators.append("left_right")
    if enable_up_down_estimator:
        selected_direction_estimators.append("up_down")

    if st.button(
        control_label,
        key="toggle_ctrl",
        use_container_width=True,
        type="primary" if locomotion_active else "secondary",
    ):
        if locomotion_active:
            ok, msg = launch_process_manager.stop_launch("control_stack")
        else:
            ok, msg = launch_process_manager.start_launch(
                "control_stack",
                "hq_pcot",
                "locomotion.launch.py",
                launch_args={
                    "enable_wireless_cmd_bridge": str(enable_wireless_cmd_bridge).lower(),
                },
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    with st.expander("Locomotion Options", expanded=False):
        enable_wireless_cmd_bridge = st.checkbox(
            "Wireless Cmd Bridge",
            value=enable_wireless_cmd_bridge,
            key="ctrl_enable_wireless_cmd_bridge",
        )
    autonomy_label = "Stop Autonomy" if autonomy_active else "Start Autonomy"
    if st.button(
        autonomy_label,
        key="toggle_autonomy",
        use_container_width=True,
        type="primary" if autonomy_active else "secondary",
    ):
        if autonomy_active:
            ok, msg = launch_process_manager.stop_launch("autonomy")
        else:
            ok, msg = launch_process_manager.start_node(
                "autonomy",
                "coordination_module",
                "intent_command_coordinator",
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    if active_intent_estimator_modes:
        active_labels = ", ".join(
            _INTENT_ESTIMATORS[mode]["label"] for mode in active_intent_estimator_modes
        )
        st.caption(f"Active intent estimator: {active_labels}")
    direction_estimator_active = bool(active_intent_estimator_modes)
    direction_estimator_label = (
        "Stop Direction Estimator"
        if direction_estimator_active
        else "Start Direction Estimator"
    )
    if st.button(
        direction_estimator_label,
        key="toggle_intent_estimator",
        use_container_width=True,
        type="primary" if direction_estimator_active else "secondary",
    ):
        if direction_estimator_active:
            messages = []
            ok = True
            for active_mode in active_intent_estimator_modes:
                active_config = _INTENT_ESTIMATORS[active_mode]
                stop_ok, stop_msg = launch_process_manager.stop_launch(
                    active_config["process_name"]
                )
                messages.append(stop_msg)
                if not stop_ok:
                    ok = False
                    break
            msg = "; ".join(messages)
        else:
            if not selected_direction_estimators:
                ok = False
                msg = "select at least one direction estimator to start"
            else:
                messages = []
                ok = True
                for selected_mode in selected_direction_estimators:
                    selected_config = _INTENT_ESTIMATORS[selected_mode]
                    start_ok, start_msg = launch_process_manager.start_node(
                        selected_config["process_name"],
                        "intent_estimator",
                        selected_config["executable"],
                    )
                    messages.append(start_msg)
                    if not start_ok:
                        ok = False
                        break
                msg = "; ".join(messages)
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    with st.expander("Direction Estimator Options", expanded=False):
        enable_front_back_estimator = st.checkbox(
            "Front/Back Estimator",
            value=enable_front_back_estimator,
            key="ctrl_enable_front_back_estimator",
        )
        enable_left_right_estimator = st.checkbox(
            "Left/Right Estimator",
            value=enable_left_right_estimator,
            key="ctrl_enable_left_right_estimator",
        )
        enable_up_down_estimator = st.checkbox(
            "Up/Down Estimator",
            value=enable_up_down_estimator,
            key="ctrl_enable_up_down_estimator",
        )
    state_converter_label = (
        "Stop State Converter" if state_converter_active else "Start State Converter"
    )
    if st.button(
        state_converter_label,
        key="toggle_state_converter",
        use_container_width=True,
        type="primary" if state_converter_active else "secondary",
    ):
        if state_converter_active:
            ok, msg = launch_process_manager.stop_state_converter_stack()
        else:
            ok, msg = launch_process_manager.start_state_converter_stack()
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    arm_controller_label = (
        "Stop Arm Controller" if arm_controller_active else "Start Arm Controller"
    )
    if st.button(
        arm_controller_label,
        key="toggle_arm_controller",
        use_container_width=True,
        type="primary" if arm_controller_active else "secondary",
    ):
        if arm_controller_active:
            ok, msg = launch_process_manager.stop_arm_controller()
        else:
            ok, msg = launch_process_manager.start_arm_controller_mode(
                arm_controller_mode
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    with st.expander("Arm Controller Options", expanded=False):
        arm_controller_mode = st.selectbox(
            "Arm Controller Mode",
            options=list(arm_controller_mode_labels.keys()),
            index=list(arm_controller_mode_labels.keys()).index(
                arm_controller_mode
                if arm_controller_mode in arm_controller_mode_labels
                else "current_ik_ipopt"
            ),
            format_func=lambda value: arm_controller_mode_labels[value],
            key="arm_controller_mode",
        )
        if active_arm_controller_mode in arm_controller_mode_labels:
            st.caption(
                "Active arm controller: "
                + arm_controller_mode_labels[active_arm_controller_mode]
            )
    foxglove_label = (
        "Stop Foxglove Bridge"
        if foxglove_active
        else "Start Foxglove Bridge"
    )
    if st.button(
        foxglove_label,
        key="toggle_foxglove",
        use_container_width=True,
        type="primary" if foxglove_active else "secondary",
    ):
        if foxglove_active:
            ok, msg = launch_process_manager.stop_launch("foxglove_bridge")
        else:
            ok, msg = launch_process_manager.start_launch(
                "foxglove_bridge", "foxglove_bridge", "foxglove_bridge_launch.xml"
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    rosbag_label = (
        "Stop Rosbag Recording"
        if rosbag_active
        else "Start Rosbag Recording"
    )
    default_rosbag_topics = list(launch_process_manager.ROSBAG_TOPICS)
    valid_rosbag_topics = set(default_rosbag_topics)
    if "rosbag_topics" in st.session_state:
        selected_default_rosbag_topics = [
            topic
            for topic in st.session_state["rosbag_topics"]
            if topic in valid_rosbag_topics
        ]
        st.session_state["rosbag_topics"] = selected_default_rosbag_topics
    else:
        selected_default_rosbag_topics = default_rosbag_topics
    selected_rosbag_topics = st.multiselect(
        "Rosbag Topics",
        options=default_rosbag_topics,
        default=selected_default_rosbag_topics,
        key="rosbag_topics",
        disabled=rosbag_active,
        help="Choose which topics to record when starting rosbag. Defaults to all topics.",
    )
    if rosbag_active:
        st.caption("Rosbag topic selection is locked while recording is running.")
    if st.button(
        rosbag_label,
        key="toggle_rosbag",
        use_container_width=True,
        type="primary" if rosbag_active else "secondary",
    ):
        if rosbag_active:
            ok, msg = launch_process_manager.stop_launch("rosbag_recording")
        else:
            ok, msg = launch_process_manager.start_rosbag_recording(
                selected_rosbag_topics
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    st.caption("Danger zone")
    if st.button(
        "Destroy All Other ROS Nodes",
        key="destroy_other_ros_nodes",
        use_container_width=True,
        type="primary",
    ):
        visible_other_nodes = _collect_other_visible_ros_nodes(snapshot, current_node_name)
        other_dashboard_nodes = _shutdown_other_dashboard_nodes(current_node_name)
        launch_process_manager.stop_all()
        ok, msg = launch_process_manager.stop_ros_nodes(
            visible_other_nodes,
            exclude_pids=[os.getpid()],
        )
        if other_dashboard_nodes:
            msg = (
                f"destroyed {other_dashboard_nodes} extra dashboard node(s); "
                + msg
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    _style_sidebar_buttons()


def _render_dashboard(node: TelemetryNode) -> None:
    snapshot = node.snapshot()

    st.subheader("Modules")
    render_status(snapshot, snapshot["status_timeout_s"])
    st.subheader("Topics")
    topic_available = snapshot.get("topic_available", {})
    topic_latest_msg = snapshot.get("topic_latest_msg", {})
    topic_names = snapshot.get("topic_names", {})
    topic_rate = snapshot.get("topic_rate", {})
    topic_status_timeout = float(snapshot["status_timeout_s"])
    now = time.monotonic()

    def _topic_ok(topic_key: str, status_key: str) -> bool:
        return _get_fresh_status(snapshot["status"], status_key, now, topic_status_timeout) is True

    topic_groups = [
        (
            "Control",
            [
                ("lowstate", "lowstate"),
                ("locomotion_cmd", "locomotion_cmd"),
                ("lowcmd", "lowcmd"),
            ],
        ),
        (
            "Estimator",
            [
                ("imu", "imu"),
                ("odometry_filtered", "odometry_filtered"),
                ("joint_states", "joint_states"),
                ("tf", "tf"),
                ("qdq_est", "qdq_est"),
            ],
        ),
        (
            "Arm",
            [
                ("arm_feedback", "arm_feedback"),
                ("arm_state", "arm_state"),
                ("arm_command_state", "arm_command_state"),
                ("arm_command", "arm_command"),
            ],
        ),
        (
            "Intent",
            [
                ("intent_forward_backward", "intent_forward_backward"),
                ("intent_left_right", "intent_left_right"),
                ("intent_up_down", "intent_up_down"),
            ],
        ),
    ]
    st.markdown(
        """
        <style>
        .topic-group { margin-bottom: 0.75rem; }
        .topic-group-title { font-weight: 700; margin-bottom: 0.25rem; }
        .topic-wrap { display:flex; flex-wrap:wrap; gap:0.4rem; }
        .topic-ok { background:#2ecc71; color:#fff; padding:0.2rem 0.5rem; border-radius:0.35rem;
                    font-weight:600; font-size:0.9rem; }
        .topic-bad { background:#e74c3c; color:#fff; padding:0.2rem 0.5rem; border-radius:0.35rem;
                     font-weight:600; font-size:0.9rem; }
        .intent-panels { display:flex; flex-wrap:wrap; gap:0.9rem; margin-top:0.8rem; margin-bottom:0.25rem; }
        .intent-card {
            background:linear-gradient(145deg, #f8fbf8, #eef4f0);
            border:1px solid #d9e3dc;
            border-radius:1rem;
            padding:0.9rem 1rem 1rem;
            min-width:260px;
            box-shadow:0 10px 24px rgba(24, 46, 34, 0.08);
        }
        .intent-card-title { font-weight:700; color:#1d2b23; margin-bottom:0.12rem; }
        .intent-card-caption { font-size:0.82rem; color:#64756b; margin-bottom:0.7rem; }
        .intent-card-state { margin-top:0.7rem; font-weight:700; color:#22352a; text-align:center; }
        .intent-stack {
            display:flex;
            flex-direction:column;
            gap:0.45rem;
            align-items:center;
            justify-content:center;
            min-height:194px;
        }
        .intent-row {
            display:flex;
            gap:0.45rem;
            align-items:center;
            justify-content:center;
            min-height:62px;
        }
        .intent-cell {
            display:flex;
            align-items:center;
            justify-content:center;
            border:1px solid #d3ddd7;
            background:#e8ede9;
            color:#a0aca4;
            box-shadow:inset 0 1px 0 rgba(255,255,255,0.8);
        }
        .intent-arrow { width:62px; height:62px; border-radius:1rem; font-size:1.9rem; line-height:1; }
        .intent-center { width:54px; height:54px; border-radius:999px; margin:auto; font-size:1.1rem; }
        .intent-active-up {
            background:linear-gradient(180deg, #34d399, #0f9f6e);
            border-color:#0d845c;
            color:#fff;
        }
        .intent-active-down {
            background:linear-gradient(180deg, #fb923c, #ea580c);
            border-color:#c94a08;
            color:#fff;
        }
        .intent-active-center {
            background:linear-gradient(180deg, #60a5fa, #2563eb);
            border-color:#1d4ed8;
            color:#fff;
        }
        .intent-active-left {
            background:linear-gradient(180deg, #2dd4bf, #0f766e);
            border-color:#115e59;
            color:#fff;
        }
        .intent-active-right {
            background:linear-gradient(180deg, #fbbf24, #d97706);
            border-color:#b45309;
            color:#fff;
        }
        .intent-muted { opacity:0.78; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    group_blocks = []
    for group_name, topic_rows in topic_groups:
        badges = []
        for key, status_key in topic_rows:
            topic_key = topic_names.get(key, f"/{key}")
            ok = _topic_ok(topic_key, status_key)
            available = bool(topic_available.get(topic_key, False))
            rate_hz = topic_rate.get(key)
            parts = [topic_key]
            if available:
                if rate_hz is not None:
                    parts.append(f"hz={rate_hz:.1f}")
                else:
                    parts.append("hz=--")
            cls = "topic-ok" if ok else "topic-bad"
            badges.append(f"<span class=\"{cls}\">{' | '.join(parts)}</span>")
        group_blocks.append(
            "<div class=\"topic-group\">"
            f"<div class=\"topic-group-title\">{group_name}</div>"
            f"<div class=\"topic-wrap\">{''.join(badges)}</div>"
            "</div>"
        )
    st.markdown("".join(group_blocks), unsafe_allow_html=True)
    fb_topic_name = topic_names.get(
        "intent_forward_backward", "/direction_intent/front_back/label"
    )
    lr_topic_name = topic_names.get("intent_left_right", "/direction_intent/left_right/label")
    ud_topic_name = topic_names.get("intent_up_down", "/direction_intent/up_down/label")
    st.markdown(
        "<div class=\"intent-panels\">"
        + _render_front_back_intent_pad(
            topic_name=fb_topic_name,
            rate_hz=topic_rate.get("intent_forward_backward"),
            available=bool(topic_available.get(fb_topic_name, False)),
            fresh=_topic_ok(fb_topic_name, "intent_forward_backward"),
            label=_parse_intent_label(topic_latest_msg.get("intent_forward_backward")),
        )
        + _render_left_right_intent_pad(
            lr_topic_name=lr_topic_name,
            lr_rate_hz=topic_rate.get("intent_left_right"),
            lr_available=bool(topic_available.get(lr_topic_name, False)),
            lr_fresh=_topic_ok(lr_topic_name, "intent_left_right"),
            lr_label=_parse_intent_label(topic_latest_msg.get("intent_left_right")),
        )
        + _render_up_down_intent_pad(
            topic_name=ud_topic_name,
            rate_hz=topic_rate.get("intent_up_down"),
            available=bool(topic_available.get(ud_topic_name, False)),
            fresh=_topic_ok(
                ud_topic_name,
                "intent_up_down",
            ),
            label=_parse_intent_label(topic_latest_msg.get("intent_up_down")),
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Nodes")
    node_names = snapshot.get("node_names", [])
    if node_names:
        st.code("\n".join(node_names), language="text")
    else:
        st.caption("No ROS nodes currently visible to the dashboard.")


if hasattr(st, "fragment"):
    _render_sidebar = st.fragment(run_every="2s")(_render_sidebar)
    _render_dashboard = st.fragment(run_every="2s")(_render_dashboard)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Go2 Telemetry")
    st.title("Go2 Telemetry Dashboard")

    node = get_ros_node()
    with st.sidebar:
        _render_sidebar(node)
    _render_dashboard(node)


if __name__ == "__main__":
    main()

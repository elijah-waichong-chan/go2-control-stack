#!/usr/bin/env python3

from collections import deque
import os
import threading
import time
import uuid
from typing import Deque, Dict, Tuple

import streamlit as st
import streamlit.components.v1 as components

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from go2_msgs.msg import ArmAngles, LocomotionCmd, LoopStatus, QDq
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from unitree_go.msg import LowCmd, LowState
from std_msgs.msg import Int32
from unitree_arm.msg import ArmString
from telemetry_dashboard import launch_process_manager


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
        self._shutdown_event = threading.Event()

        self._status: Dict[str, Tuple[object, float]] = {}
        self._topic_available: Dict[str, bool] = {}
        self._topic_timestamps: Dict[str, Deque[float]] = {}
        self._topic_latest_msg: Dict[str, str] = {}
        self._node_names: list[str] = []
        self._graph_topic_count = 0
        self._graph_node_count = 0
        self._system_stats: Dict[str, object] = {}
        self._last_cpu_sample: tuple[int, int] | None = None
        self._topic_names = {
            "qdq_est": "/qdq_est",
            "odometry_filtered": "/odometry/filtered",
            "lowstate": "/lowstate",
            "locomotion_cmd": "/locomotion_cmd",
            "lowcmd": "/lowcmd",
            "joint_states": "/joint_states",
            "arm_angles": "/arm_angles",
            "arm_feedback": "/arm_Feedback",
            "intent_forward_backward": "/direction_intent/forward_backward",
            "intent_left_right": "/direction_intent/left_right",
        }

        self.create_subscription(QDq, self._topic_names["qdq_est"], self.on_qdq_est, qos)
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
        self.create_subscription(ArmAngles, self._topic_names["arm_angles"], self.on_arm_angles, qos)
        self.create_subscription(
            ArmString, self._topic_names["arm_feedback"], self.on_arm_feedback, qos
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
            "/status/intent_estimator/forward_backward",
            lambda m: self.on_loop_status("intent_estimator_forward_backward", m),
            status_qos,
        )
        self.create_subscription(
            LoopStatus,
            "/status/intent_estimator/left_right",
            lambda m: self.on_loop_status("intent_estimator_left_right", m),
            status_qos,
        )
        self._graph_thread = threading.Thread(
            target=self._graph_monitor_loop,
            name=f"{node_name}_graph_monitor",
            daemon=True,
        )
        self._graph_thread.start()

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

    def on_arm_angles(self, msg: ArmAngles) -> None:
        self._mark_topic("arm_angles")

    def on_arm_feedback(self, msg: ArmString) -> None:
        self._mark_topic("arm_feedback")

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
                self._graph_topic_count = len(topics)
                for _key, name in self._topic_names.items():
                    self._topic_available[name] = name in available
        except Exception:
            with self._lock:
                self._graph_topic_count = 0
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
                self._graph_node_count = len(nodes)
                self._node_names = sorted(formatted)
        except Exception:
            with self._lock:
                self._graph_node_count = 0
                self._node_names = []

    def _update_system_stats(self) -> None:
        stats: Dict[str, object] = {
            "cpu_usage_percent": None,
            "memory_usage_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
        }

        try:
            with open("/proc/stat", "r", encoding="utf-8") as fh:
                fields = fh.readline().split()
            if fields and fields[0] == "cpu":
                values = [int(v) for v in fields[1:]]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                total = sum(values)
                previous = self._last_cpu_sample
                self._last_cpu_sample = (total, idle)
                if previous is not None:
                    prev_total, prev_idle = previous
                    total_delta = total - prev_total
                    idle_delta = idle - prev_idle
                    if total_delta > 0:
                        stats["cpu_usage_percent"] = 100.0 * (
                            1.0 - (idle_delta / float(total_delta))
                        )
        except OSError:
            if hasattr(os, "getloadavg"):
                try:
                    load1, _load5, _load15 = os.getloadavg()
                    cpu_count = max(os.cpu_count() or 1, 1)
                    stats["cpu_usage_percent"] = 100.0 * (load1 / float(cpu_count))
                except OSError:
                    pass

        try:
            meminfo_kb: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    key, value = line.split(":", 1)
                    meminfo_kb[key] = int(value.strip().split()[0])
            total_kb = meminfo_kb.get("MemTotal")
            available_kb = meminfo_kb.get("MemAvailable")
            if total_kb is not None and available_kb is not None and total_kb > 0:
                used_kb = total_kb - available_kb
                stats["memory_total_gb"] = total_kb / (1024.0 * 1024.0)
                stats["memory_used_gb"] = used_kb / (1024.0 * 1024.0)
                stats["memory_usage_percent"] = 100.0 * (used_kb / float(total_kb))
        except OSError:
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                total_pages = os.sysconf("SC_PHYS_PAGES")
                available_pages = os.sysconf("SC_AVPHYS_PAGES")
                total_bytes = page_size * total_pages
                available_bytes = page_size * available_pages
                if total_bytes > 0:
                    used_bytes = total_bytes - available_bytes
                    stats["memory_total_gb"] = total_bytes / (1024.0**3)
                    stats["memory_used_gb"] = used_bytes / (1024.0**3)
                    stats["memory_usage_percent"] = 100.0 * (
                        used_bytes / float(total_bytes)
                    )
            except (AttributeError, OSError, ValueError):
                pass

        with self._lock:
            self._system_stats = stats

    def _graph_monitor_loop(self) -> None:
        poll_period_s = 1.0 / max(self.graph_poll_hz, 0.1)
        while not self._shutdown_event.wait(poll_period_s):
            self._update_topic_availability()
            self._update_node_list()
            self._update_system_stats()

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
                "graph_topic_count": self._graph_topic_count,
                "graph_node_count": self._graph_node_count,
                "system_stats": dict(self._system_stats),
                "status_timeout_s": self.status_timeout_s,
            }

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        if self._graph_thread.is_alive():
            self._graph_thread.join(timeout=1.0)
        return super().destroy_node()


def get_ros_node() -> TelemetryNode:
    if "ros_node" in st.session_state:
        return st.session_state["ros_node"]

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
    except RuntimeError:
        # rclpy may already be initialized in this process
        pass

    if "ros_node_name" not in st.session_state:
        st.session_state["ros_node_name"] = f"telemetry_dashboard_{os.getpid()}_{uuid.uuid4().hex[:6]}"

    node = TelemetryNode(st.session_state["ros_node_name"])
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    st.session_state["ros_executor"] = executor
    st.session_state["ros_spin_thread"] = thread
    st.session_state["ros_node"] = node
    return node


def _get_fresh_status(status_map: Dict[str, Tuple[object, float]], key: str, now: float, timeout: float):
    if key not in status_map:
        return None
    val, t = status_map[key]
    if (now - t) > timeout:
        return None
    return val


def _get_intent_estimator_status(
    status_map: Dict[str, Tuple[object, float]], now: float, timeout: float
):
    fb = _get_fresh_status(status_map, "intent_estimator_forward_backward", now, timeout)
    lr = _get_fresh_status(status_map, "intent_estimator_left_right", now, timeout)

    if fb is None and lr is None:
        return None

    fb_status = _status_code(fb)
    lr_status = _status_code(lr)

    if fb_status == 1 and lr_status == 1:
        return 1
    if fb_status == 2 and lr_status == 1:
        return 2
    if fb_status == 1 and lr_status == 2:
        return 3
    if fb_status == 2 and lr_status == 2:
        return 4
    if fb_status == 1 and lr_status is None:
        return 5
    if lr_status == 1 and fb_status is None:
        return 6
    if fb_status == 2 and lr_status is None:
        return 7
    if lr_status == 2 and fb_status is None:
        return 8
    return 0


def _status_code(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("status")
        if raw is None:
            return None
        return int(raw)
    return int(value)


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


def _format_percent(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "not available"
    return f"{value_f:.1f}%"


def _format_gb_pair(used_value: object, total_value: object) -> str:
    try:
        used_gb = float(used_value)
        total_gb = float(total_value)
    except (TypeError, ValueError):
        return "not available"
    return f"{used_gb:.2f} / {total_gb:.2f} GB"


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

    if key == "state_estimator":
        estimator_status = _status_code(value)
        if estimator_status is None:
            return "error", "invalid status"
        if estimator_status == 1:
            return "success", "running"
        if estimator_status == 2:
            return "warning", "waiting for standing-init"
        return "error", f"idle ({estimator_status})"

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
            return "warning", "forward/backward waiting"
        if intent_status == 3:
            return "warning", "left/right waiting"
        if intent_status == 4:
            return "warning", "both waiting"
        if intent_status == 5:
            return "warning", "left/right status missing"
        if intent_status == 6:
            return "warning", "forward/backward status missing"
        if intent_status == 7:
            return "warning", "forward/backward waiting"
        if intent_status == 8:
            return "warning", "left/right waiting"
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
        ("state_estimator", "State Estimator"),
        ("rl_controller", "RL Controller"),
        ("intent_estimator", "Intent Estimator"),
        ("arm_parser", "Arm Parser"),
    ]

    status_view = {
        "state_estimator": _get_fresh_status(status_map, "state_estimator", now, timeout),
        "standing_init": _get_fresh_status(status_map, "standing_init", now, timeout),
        "arm_parser": _get_fresh_status(status_map, "arm_parser", now, timeout),
        "rl_controller": _get_fresh_status(status_map, "loco_ctrl", now, timeout),
        "intent_estimator": _get_intent_estimator_status(status_map, now, timeout),
    }
    status_topics = {
        "standing_init": "/status/standing_init",
        "state_estimator": "/status/state_estimator",
        "arm_parser": "/status/arm_parser",
        "rl_controller": "/status/loco_ctrl",
    }

    cols = st.columns(len(labels))
    for idx, (key, label) in enumerate(labels):
        with cols[idx]:
            val = status_view.get(key)
            severity, summary = _module_summary(key, val)
            _render_summary(label, severity, summary)

            with st.expander("Details", expanded=False):
                if key == "intent_estimator":
                    st.caption(
                        "Derived from `/status/intent_estimator/forward_backward` and "
                        "`/status/intent_estimator/left_right`."
                    )
                    if val is None:
                        st.write("Aggregate status: unavailable")
                    else:
                        st.write(f"Aggregate status: `{_status_code(val)}`")

                    fb = _get_fresh_status(
                        status_map, "intent_estimator_forward_backward", now, timeout
                    )
                    lr = _get_fresh_status(
                        status_map, "intent_estimator_left_right", now, timeout
                    )
                    _render_loop_status_details(
                        "/status/intent_estimator/forward_backward",
                        fb,
                    )
                    st.divider()
                    _render_loop_status_details(
                        "/status/intent_estimator/left_right",
                        lr,
                    )
                else:
                    _render_loop_status_details(status_topics[key], val)


def _style_sidebar_buttons(
    locomotion_active: bool,
    autonomy_active: bool,
    foxglove_active: bool,
    rosbag_active: bool,
) -> None:
    start_green = "#2ecc71"
    start_border = "#27ae60"
    stop_red = "#e74c3c"
    stop_border = "#c0392b"
    control_label = (
        "Stop Control Stack" if locomotion_active else "Start Control Stack"
    )
    control_bg = stop_red if locomotion_active else start_green
    control_border = stop_border if locomotion_active else start_border
    autonomy_label = "Stop Autonomy" if autonomy_active else "Start Autonomy"
    autonomy_bg = stop_red if autonomy_active else start_green
    autonomy_border = stop_border if autonomy_active else start_border
    foxglove_label = (
        "Stop Foxglove Bridge"
        if foxglove_active
        else "Start Foxglove Bridge"
    )
    foxglove_bg = stop_red if foxglove_active else start_green
    foxglove_border = stop_border if foxglove_active else start_border
    rosbag_label = (
        "Stop Rosbag Recording"
        if rosbag_active
        else "Start Rosbag Recording"
    )
    rosbag_bg = stop_red if rosbag_active else start_green
    rosbag_border = stop_border if rosbag_active else start_border
    js = f"""
    <script>
    const styleBtn = (label, bg, border) => {{
      const btns = Array.from(window.parent.document.querySelectorAll('button'));
      const btn = btns.find(b => b.innerText.trim() === label);
      if (!btn) return false;
      btn.disabled = false;
      btn.style.setProperty('background-color', bg, 'important');
      btn.style.setProperty('color', '#ffffff', 'important');
      btn.style.setProperty('border', '1px solid ' + border, 'important');
      btn.style.setProperty('opacity', '1.0', 'important');
      btn.style.setProperty('cursor', 'pointer', 'important');
      return true;
    }};

    const apply = () => {{
      let ok = true;
      ok = styleBtn('{control_label}', '{control_bg}', '{control_border}') && ok;
      ok = styleBtn('{autonomy_label}', '{autonomy_bg}', '{autonomy_border}') && ok;
      ok = styleBtn('{foxglove_label}', '{foxglove_bg}', '{foxglove_border}') && ok;
      ok = styleBtn('{rosbag_label}', '{rosbag_bg}', '{rosbag_border}') && ok;
      if (!ok) setTimeout(apply, 100);
    }};
    apply();
    if (!window._dashBtnInterval) {{
      window._dashBtnInterval = setInterval(apply, 500);
    }}
    </script>
    """
    components.html(js, height=0)


def _render_sidebar(node: TelemetryNode) -> None:
    snapshot = node.snapshot()

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
    control_label = (
        "Stop Control Stack" if locomotion_active else "Start Control Stack"
    )
    enable_estimator = st.session_state.get("ctrl_enable_estimator", True)
    enable_wireless_cmd_bridge = st.session_state.get("ctrl_enable_wireless_cmd_bridge", True)
    enable_arm_parser = st.session_state.get("ctrl_enable_arm_parser", True)
    enable_forward_backward_estimator = st.session_state.get(
        "ctrl_enable_forward_backward_estimator", True
    )
    enable_left_right_estimator = st.session_state.get("ctrl_enable_left_right_estimator", True)

    if st.button(control_label, key="toggle_ctrl", use_container_width=True):
        if locomotion_active:
            ok, msg = launch_process_manager.stop_launch("control_stack")
        else:
            ok, msg = launch_process_manager.start_launch(
                "control_stack",
                "locomotion_controller",
                "control_stack.launch.py",
                launch_args={
                    "enable_estimator": str(enable_estimator).lower(),
                    "enable_arm_parser": str(enable_arm_parser).lower(),
                    "enable_wireless_cmd_bridge": str(enable_wireless_cmd_bridge).lower(),
                    "enable_forward_backward_estimator": str(enable_forward_backward_estimator).lower(),
                    "enable_left_right_estimator": str(enable_left_right_estimator).lower(),
                },
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    with st.expander("Control Stack Options", expanded=False):
        st.caption("Defaults start the full stack. Disable the optional pieces you want to skip.")
        enable_estimator = st.checkbox(
            "Estimator",
            value=enable_estimator,
            key="ctrl_enable_estimator",
        )
        enable_wireless_cmd_bridge = st.checkbox(
            "Wireless Cmd Bridge",
            value=enable_wireless_cmd_bridge,
            key="ctrl_enable_wireless_cmd_bridge",
        )
        enable_arm_parser = st.checkbox(
            "Arm Parser",
            value=enable_arm_parser,
            key="ctrl_enable_arm_parser",
        )
        enable_forward_backward_estimator = st.checkbox(
            "Forward/Backward Estimator",
            value=enable_forward_backward_estimator,
            key="ctrl_enable_forward_backward_estimator",
        )
        enable_left_right_estimator = st.checkbox(
            "Left/Right Estimator",
            value=enable_left_right_estimator,
            key="ctrl_enable_left_right_estimator",
        )
    autonomy_label = "Stop Autonomy" if autonomy_active else "Start Autonomy"
    if st.button(autonomy_label, key="toggle_autonomy", use_container_width=True):
        if autonomy_active:
            ok, msg = launch_process_manager.stop_launch("autonomy")
        else:
            ok, msg = launch_process_manager.start_launch(
                "autonomy", "locomotion_controller", "autonomy.launch.py"
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    foxglove_label = (
        "Stop Foxglove Bridge"
        if foxglove_active
        else "Start Foxglove Bridge"
    )
    if st.button(foxglove_label, key="toggle_foxglove", use_container_width=True):
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
    if st.button(rosbag_label, key="toggle_rosbag", use_container_width=True):
        if rosbag_active:
            ok, msg = launch_process_manager.stop_launch("rosbag_recording")
        else:
            ok, msg = launch_process_manager.start_rosbag_recording()
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    _style_sidebar_buttons(
        locomotion_active,
        autonomy_active,
        foxglove_active,
        rosbag_active,
    )


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
                ("odometry_filtered", "odometry_filtered"),
                ("joint_states", "joint_states"),
                ("qdq_est", "qdq_est"),
            ],
        ),
        (
            "Arm",
            [
                ("arm_feedback", "arm_feedback"),
                ("arm_angles", "arm_angles"),
            ],
        ),
        (
            "Intent",
            [
                ("intent_forward_backward", "intent_forward_backward"),
                ("intent_left_right", "intent_left_right"),
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
            latest_msg = topic_latest_msg.get(key)
            rate_hz = topic_rate.get(key)
            parts = [topic_key]
            if available:
                if rate_hz is not None:
                    parts.append(f"hz={rate_hz:.1f}")
                else:
                    parts.append("hz=--")
            if key.startswith("intent_") and latest_msg is not None and ok:
                parts.append(f"msg={latest_msg}")
            elif key.startswith("intent_") and available and ok is not False:
                parts.append("msg=waiting")
            cls = "topic-ok" if ok else "topic-bad"
            badges.append(f"<span class=\"{cls}\">{' | '.join(parts)}</span>")
        group_blocks.append(
            "<div class=\"topic-group\">"
            f"<div class=\"topic-group-title\">{group_name}</div>"
            f"<div class=\"topic-wrap\">{''.join(badges)}</div>"
            "</div>"
        )
    st.markdown("".join(group_blocks), unsafe_allow_html=True)
    st.subheader("Nodes")
    node_names = snapshot.get("node_names", [])
    if node_names:
        st.code("\n".join(node_names), language="text")
    else:
        st.caption("No ROS nodes currently visible to the dashboard.")

    st.subheader("Debug")
    system_stats = snapshot.get("system_stats", {})
    graph_topic_count = snapshot.get("graph_topic_count", 0)
    graph_node_count = snapshot.get("graph_node_count", 0)
    debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)
    with debug_col1:
        st.metric("CPU Usage", _format_percent(system_stats.get("cpu_usage_percent")))
    with debug_col2:
        st.metric("RAM Usage", _format_percent(system_stats.get("memory_usage_percent")))
    with debug_col3:
        st.metric(
            "RAM Used",
            _format_gb_pair(
                system_stats.get("memory_used_gb"),
                system_stats.get("memory_total_gb"),
            ),
        )
    with debug_col4:
        st.metric("ROS Nodes / Topics", f"{graph_node_count} / {graph_topic_count}")


if hasattr(st, "fragment"):
    _render_sidebar = st.fragment(run_every="1s")(_render_sidebar)
    _render_dashboard = st.fragment(run_every="1s")(_render_dashboard)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Go2 Telemetry")
    st.title("Go2 Telemetry Dashboard")

    node = get_ros_node()
    with st.sidebar:
        _render_sidebar(node)
    _render_dashboard(node)


if __name__ == "__main__":
    main()

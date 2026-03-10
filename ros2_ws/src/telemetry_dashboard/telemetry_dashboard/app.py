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

from go2_msgs.msg import ArmAngles, QDq
from unitree_go.msg import LowState
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
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
        self._topic_names = {
            "qdq_est": "/qdq_est",
            "lowstate": "/lowstate",
            "arm_angles": "/arm_angles",
            "intent_forward_backward": "/direction_intent/forward_backward",
            "intent_left_right": "/direction_intent/left_right",
        }

        self.create_subscription(QDq, self._topic_names["qdq_est"], self.on_qdq_est, qos)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, qos)
        self.create_subscription(ArmAngles, self._topic_names["arm_angles"], self.on_arm_angles, qos)
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
            "/status/state_estimator",
            lambda m: self.on_status_value("state_estimator", int(m.data)),
            status_qos,
        )
        self.create_subscription(
            Int32,
            "/status/loco_ctrl",
            lambda m: self.on_status_value("loco_ctrl", int(m.data)),
            status_qos,
        )
        self.create_subscription(
            Int32,
            "/status/arm_parser",
            lambda m: self.on_status_value("arm_parser", int(m.data)),
            status_qos,
        )
        self.create_subscription(Bool, "/status/loco_ctrl/safety_stop",
                                 lambda m: self.on_status("safety_stop", m), status_qos)
        self.create_subscription(
            Int32,
            "/status/standing_init",
            lambda m: self.on_status_value("standing_init", int(m.data)),
            status_qos,
        )
        self.create_subscription(
            Int32,
            "/status/intent_estimator/forward_backward",
            lambda m: self.on_status_value("intent_estimator_forward_backward", int(m.data)),
            status_qos,
        )
        self.create_subscription(
            Int32,
            "/status/intent_estimator/left_right",
            lambda m: self.on_status_value("intent_estimator_left_right", int(m.data)),
            status_qos,
        )
        self._graph_thread = threading.Thread(
            target=self._graph_monitor_loop,
            name=f"{node_name}_graph_monitor",
            daemon=True,
        )
        self._graph_thread.start()

    def on_status(self, name: str, msg: Bool) -> None:
        with self._lock:
            self._status[name] = (bool(msg.data), time.monotonic())

    def on_status_value(self, name: str, value: object) -> None:
        with self._lock:
            self._status[name] = (value, time.monotonic())

    def on_qdq_est(self, msg: QDq) -> None:
        with self._lock:
            t = time.monotonic()
            self._status["qdq_est"] = (True, t)
            self._update_topic_rate("qdq_est", t)

    def on_lowstate(self, msg: LowState) -> None:
        with self._lock:
            t = time.monotonic()
            self._status["lowstate"] = (True, t)
            self._update_topic_rate("lowstate", t)

    def on_arm_angles(self, msg: ArmAngles) -> None:
        with self._lock:
            t = time.monotonic()
            self._status["arm_angles"] = (True, t)
            self._update_topic_rate("arm_angles", t)

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

    def _graph_monitor_loop(self) -> None:
        poll_period_s = 1.0 / max(self.graph_poll_hz, 0.1)
        while not self._shutdown_event.wait(poll_period_s):
            self._update_topic_availability()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = time.monotonic()
            return {
                "status": dict(self._status),
                "topic_available": dict(self._topic_available),
                "topic_rate": self._snapshot_topic_rates_locked(now),
                "topic_latest_msg": dict(self._topic_latest_msg),
                "topic_names": dict(self._topic_names),
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
    st.session_state["svc_emergency_stop"] = node.create_client(Trigger, "/loco_ctrl/emergency_stop")
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

    fb_status = int(fb) if fb is not None else None
    lr_status = int(lr) if lr is not None else None

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


def render_status(snapshot: Dict[str, object], timeout_s: float) -> None:
    now = time.monotonic()
    timeout = float(timeout_s)
    status_map = snapshot["status"]

    labels = [
        ("standing_init", "Standing Init"),
        ("state_estimator", "State Estimator"),
        ("arm_parser", "Arm Parser"),
        ("rl_controller", "RL Controller"),
        ("intent_estimator", "Intent Estimator"),
        ("safety_stop", "Safety Stop"),
    ]

    status_view = {
        "state_estimator": _get_fresh_status(status_map, "state_estimator", now, timeout),
        "standing_init": _get_fresh_status(status_map, "standing_init", now, timeout),
        "arm_parser": _get_fresh_status(status_map, "arm_parser", now, timeout),
        "rl_controller": _get_fresh_status(status_map, "loco_ctrl", now, timeout),
        "intent_estimator": _get_intent_estimator_status(status_map, now, timeout),
        "safety_stop": _get_fresh_status(status_map, "safety_stop", now, timeout),
    }

    cols = st.columns(len(labels))
    for idx, (key, label) in enumerate(labels):
        with cols[idx]:
            val = status_view.get(key)
            if val is None:
                if key == "safety_stop":
                    st.success(f"{label}: OK")
                else:
                    st.error(f"{label}: not running")
                continue
            if key == "safety_stop":
                if val:
                    st.error(f"{label}: TRIGGERED")
                else:
                    st.success(f"{label}: OK")
            else:
                if key == "standing_init":
                    standing_status = int(val)
                    if standing_status == 3:
                        st.success(f"{label}: complete (3)")
                    elif standing_status == 2:
                        st.warning(f"{label}: waiting for /lowstate (2)")
                    elif standing_status == 1:
                        st.info(f"{label}: running sequence (1)")
                    else:
                        st.error(f"{label}: idle ({int(val)})")
                elif key == "arm_parser":
                    arm_parser_status = int(val)
                    if arm_parser_status == 1:
                        st.success(f"{label}: publishing (1)")
                    elif arm_parser_status == 2:
                        st.warning(f"{label}: waiting for /arm_Feedback (2)")
                    else:
                        st.error(f"{label}: idle ({int(val)})")
                elif key == "state_estimator":
                    estimator_status = int(val)
                    if estimator_status == 1:
                        st.success(f"{label}: running (1)")
                    elif estimator_status == 2:
                        st.warning(f"{label}: waiting for standing-init readiness (2)")
                    else:
                        st.error(f"{label}: idle ({int(val)})")
                elif key == "rl_controller":
                    loco_status = int(val)
                    if loco_status == 1:
                        st.success(f"{label}: running (1)")
                    elif loco_status == 2:
                        st.warning(f"{label}: waiting for /lowstate (2)")
                    elif loco_status == 3:
                        st.warning(f"{label}: waiting for standing-init readiness (3)")
                    else:
                        st.error(f"{label}: idle (0)")
                elif key == "intent_estimator":
                    intent_status = int(val)
                    if intent_status == 1:
                        st.success(f"{label}: running (1)")
                    elif intent_status == 2:
                        st.warning(f"{label}: forward/backward waiting for topics (2)")
                    elif intent_status == 3:
                        st.warning(f"{label}: left/right waiting for topics (3)")
                    elif intent_status == 4:
                        st.warning(f"{label}: both nodes waiting for topics (4)")
                    elif intent_status == 5:
                        st.warning(f"{label}: forward/backward running, left/right status missing (5)")
                    elif intent_status == 6:
                        st.warning(f"{label}: left/right running, forward/backward status missing (6)")
                    elif intent_status == 7:
                        st.warning(f"{label}: forward/backward waiting, left/right status missing (7)")
                    elif intent_status == 8:
                        st.warning(f"{label}: left/right waiting, forward/backward status missing (8)")
                    else:
                        st.error(f"{label}: idle ({int(val)})")
                elif bool(val):
                    st.success(f"{label}: running")
                else:
                    st.error(f"{label}: not running")


def _style_sidebar_buttons(
    locomotion_active: bool,
    foxglove_active: bool,
    rosbag_active: bool,
) -> None:
    start_green = "#2ecc71"
    start_border = "#27ae60"
    stop_red = "#e74c3c"
    stop_border = "#c0392b"
    control_label = "Stop Control Stack" if locomotion_active else "Start Control Stack"
    control_bg = stop_red if locomotion_active else start_green
    control_border = stop_border if locomotion_active else start_border
    foxglove_label = "Stop Foxglove Bridge" if foxglove_active else "Start Foxglove Bridge"
    foxglove_bg = stop_red if foxglove_active else start_green
    foxglove_border = stop_border if foxglove_active else start_border
    rosbag_label = "Stop Rosbag Recording" if rosbag_active else "Start Rosbag Recording"
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
      ok = styleBtn('{foxglove_label}', '{foxglove_bg}', '{foxglove_border}') && ok;
      ok = styleBtn('{rosbag_label}', '{rosbag_bg}', '{rosbag_border}') && ok;
      ok = styleBtn('EMERGENCY STOP', '{stop_red}', '{stop_border}') && ok;
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
    locomotion_active = (rl_ctrl_status is not None) or launch_process_manager.is_running("control_stack")
    foxglove_active = launch_process_manager.is_running("foxglove_bridge")
    rosbag_active = launch_process_manager.is_running("rosbag_recording")
    control_label = "Stop Control Stack" if locomotion_active else "Start Control Stack"
    if st.button(control_label, key="toggle_ctrl", use_container_width=True):
        if locomotion_active:
            ok, msg = launch_process_manager.stop_launch("control_stack")
        else:
            ok, msg = launch_process_manager.start_launch(
                "control_stack", "locomotion_controller", "control_stack.launch.py"
            )
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    foxglove_label = "Stop Foxglove Bridge" if foxglove_active else "Start Foxglove Bridge"
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
    rosbag_label = "Stop Rosbag Recording" if rosbag_active else "Start Rosbag Recording"
    if st.button(rosbag_label, key="toggle_rosbag", use_container_width=True):
        if rosbag_active:
            ok, msg = launch_process_manager.stop_launch("rosbag_recording")
        else:
            ok, msg = launch_process_manager.start_rosbag_recording()
        if ok:
            st.info(msg)
        else:
            st.warning(msg)
    if st.button("EMERGENCY STOP", key="emergency_stop", use_container_width=True):
        client = st.session_state.get("svc_emergency_stop")
        if client is None or not client.service_is_ready():
            st.warning("Emergency stop service not available")
        else:
            client.call_async(Trigger.Request())
            st.error("Emergency stop requested")
    _style_sidebar_buttons(locomotion_active, foxglove_active, rosbag_active)

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

    topic_rows = [
        ("lowstate", "lowstate"),
        ("qdq_est", "qdq_est"),
        ("arm_angles", "arm_angles"),
        ("intent_forward_backward", "intent_forward_backward"),
        ("intent_left_right", "intent_left_right"),
    ]
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
    st.markdown(
        """
        <style>
        .topic-wrap { display:flex; flex-wrap:wrap; gap:0.4rem; }
        .topic-ok { background:#2ecc71; color:#fff; padding:0.2rem 0.5rem; border-radius:0.35rem;
                    font-weight:600; font-size:0.9rem; }
        .topic-bad { background:#e74c3c; color:#fff; padding:0.2rem 0.5rem; border-radius:0.35rem;
                     font-weight:600; font-size:0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class=\"topic-wrap\">{''.join(badges)}</div>", unsafe_allow_html=True)


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

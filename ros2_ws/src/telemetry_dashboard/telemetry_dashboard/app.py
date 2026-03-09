#!/usr/bin/env python3

import os
import threading
import time
import uuid
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from go2_msgs.msg import QDq, LocomotionCmd
from sensor_msgs.msg import Joy
from unitree_go.msg import LowState
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
from telemetry_dashboard import launch_process_manager


def quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class TelemetryNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.qdq_topic = self.declare_parameter("qdq_topic", "/qdq").value
        self.qdq_est_topic = self.declare_parameter("qdq_est_topic", "/qdq_est").value
        self.cmd_topic = self.declare_parameter("locomotion_cmd_topic", "/locomotion_cmd").value
        self.history_sec = float(self.declare_parameter("history_sec", 20.0).value)
        self.max_samples = int(self.declare_parameter("max_samples", 6000).value)
        self.status_timeout_s = float(self.declare_parameter("status_timeout_s", 3.0).value)
        self.graph_poll_hz = float(self.declare_parameter("graph_poll_hz", 1.0).value)

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
        self._t0_q = None
        self._t0_cmd = None

        self._t_q: Deque[float] = deque(maxlen=self.max_samples)
        self._q_xyz = [deque(maxlen=self.max_samples) for _ in range(3)]
        self._dq_xyz = [deque(maxlen=self.max_samples) for _ in range(3)]
        self._rpy = [deque(maxlen=self.max_samples) for _ in range(3)]
        self._dq_joints = [deque(maxlen=self.max_samples) for _ in range(12)]

        self._t_q_est: Deque[float] = deque(maxlen=self.max_samples)
        self._q_xyz_est = [deque(maxlen=self.max_samples) for _ in range(3)]
        self._dq_xyz_est = [deque(maxlen=self.max_samples) for _ in range(3)]
        self._rpy_est = [deque(maxlen=self.max_samples) for _ in range(3)]

        self._t_cmd: Deque[float] = deque(maxlen=self.max_samples)
        self._cmd_x = deque(maxlen=self.max_samples)
        self._cmd_y = deque(maxlen=self.max_samples)
        self._cmd_z = deque(maxlen=self.max_samples)
        self._cmd_yaw = deque(maxlen=self.max_samples)
        self._cmd_gait = deque(maxlen=self.max_samples)

        self._status: Dict[str, Tuple[bool, float]] = {}
        self._topic_available: Dict[str, bool] = {}
        self._topic_rate: Dict[str, float] = {}
        self._topic_last_time: Dict[str, float] = {}
        self._topic_latest_msg: Dict[str, str] = {}
        self._topic_names = {
            "qdq": str(self.qdq_topic),
            "qdq_est": str(self.qdq_est_topic),
            "lowstate": "/lowstate",
            "joy": "/joy",
            "intent_forward_backward": "/direction_intent/forward_backward",
            "intent_left_right": "/direction_intent/left_right",
        }

        self.create_subscription(QDq, str(self.qdq_topic), self.on_qdq, qos)
        self.create_subscription(QDq, str(self.qdq_est_topic), self.on_qdq_est, qos)
        self.create_subscription(LocomotionCmd, str(self.cmd_topic), self.on_cmd, qos)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, qos)
        self.create_subscription(Joy, "/joy", self.on_joy, qos)
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

        self.create_subscription(Bool, "/status/inekf/is_running",
                                 lambda m: self.on_status("inekf", m), status_qos)
        self.create_subscription(Bool, "/status/loco_ctrl/is_running",
                                 lambda m: self.on_status("loco_ctrl", m), status_qos)
        self.create_subscription(Bool, "/status/loco_ctrl/safety_stop",
                                 lambda m: self.on_status("safety_stop", m), status_qos)
        self.create_subscription(Bool, "/status/standing_init",
                                 lambda m: self.on_status("standing_init", m), status_qos)
        self.create_subscription(Bool, "/status/intent_estimator/is_running",
                                 lambda m: self.on_status("intent_estimator", m), status_qos)
        self._graph_thread = threading.Thread(
            target=self._graph_monitor_loop,
            name=f"{node_name}_graph_monitor",
            daemon=True,
        )
        self._graph_thread.start()

    def on_status(self, name: str, msg: Bool) -> None:
        with self._lock:
            self._status[name] = (bool(msg.data), time.monotonic())

    def on_qdq(self, msg: QDq) -> None:
        t = time.monotonic()
        with self._lock:
            self._status["qdq"] = (True, t)
            self._update_topic_rate("qdq", t)
            if self._t0_q is None:
                self._t0_q = t
            t_rel = t - self._t0_q
            self._t_q.append(t_rel)
            self._q_xyz[0].append(float(msg.q[0]))
            self._q_xyz[1].append(float(msg.q[1]))
            self._q_xyz[2].append(float(msg.q[2]))
            roll, pitch, yaw = quat_to_rpy(msg.q[3], msg.q[4], msg.q[5], msg.q[6])
            self._rpy[0].append(float(roll))
            self._rpy[1].append(float(pitch))
            self._rpy[2].append(float(yaw))
            self._dq_xyz[0].append(float(msg.dq[0]))
            self._dq_xyz[1].append(float(msg.dq[1]))
            self._dq_xyz[2].append(float(msg.dq[2]))
            for i in range(12):
                self._dq_joints[i].append(float(msg.dq[6 + i]))

    def on_qdq_est(self, msg: QDq) -> None:
        t = time.monotonic()
        with self._lock:
            self._status["qdq_est"] = (True, t)
            self._update_topic_rate("qdq_est", t)
            if self._t0_q is None:
                self._t0_q = t
            t_rel = t - self._t0_q
            self._t_q_est.append(t_rel)
            self._q_xyz_est[0].append(float(msg.q[0]))
            self._q_xyz_est[1].append(float(msg.q[1]))
            self._q_xyz_est[2].append(float(msg.q[2]))
            roll, pitch, yaw = quat_to_rpy(msg.q[3], msg.q[4], msg.q[5], msg.q[6])
            self._rpy_est[0].append(float(roll))
            self._rpy_est[1].append(float(pitch))
            self._rpy_est[2].append(float(yaw))
            self._dq_xyz_est[0].append(float(msg.dq[0]))
            self._dq_xyz_est[1].append(float(msg.dq[1]))
            self._dq_xyz_est[2].append(float(msg.dq[2]))

    def on_cmd(self, msg: LocomotionCmd) -> None:
        t = time.monotonic()
        with self._lock:
            if self._t0_cmd is None:
                self._t0_cmd = t
            t_rel = t - self._t0_cmd
            self._t_cmd.append(t_rel)
            self._cmd_x.append(float(msg.x_vel))
            self._cmd_y.append(float(msg.y_vel))
            self._cmd_z.append(float(msg.z_pos))
            self._cmd_yaw.append(float(msg.yaw_rate))
            self._cmd_gait.append(float(msg.gait_hz))

    def on_lowstate(self, msg: LowState) -> None:
        with self._lock:
            self._status["lowstate"] = (True, time.monotonic())
            self._update_topic_rate("lowstate", time.monotonic())

    def on_joy(self, msg: Joy) -> None:
        with self._lock:
            t = time.monotonic()
            self._status["joy"] = (True, t)
            self._update_topic_rate("joy", t)

    def on_intent(self, key: str, msg: Int32) -> None:
        with self._lock:
            t = time.monotonic()
            self._status[key] = (True, t)
            self._update_topic_rate(key, t)
            self._topic_latest_msg[key] = str(int(msg.data))

    def _update_topic_rate(self, key: str, t: float) -> None:
        last_t = self._topic_last_time.get(key)
        if last_t is not None:
            dt = t - last_t
            if dt > 1e-6:
                inst = 1.0 / dt
                prev = self._topic_rate.get(key, inst)
                # Light smoothing to avoid flicker.
                self._topic_rate[key] = 0.8 * prev + 0.2 * inst
        self._topic_last_time[key] = t

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
            return {
                "t_q": list(self._t_q),
                "q_xyz": [list(buf) for buf in self._q_xyz],
                "dq_xyz": [list(buf) for buf in self._dq_xyz],
                "rpy": [list(buf) for buf in self._rpy],
                "dq_joints": [list(buf) for buf in self._dq_joints],
                "t_q_est": list(self._t_q_est),
                "q_xyz_est": [list(buf) for buf in self._q_xyz_est],
                "dq_xyz_est": [list(buf) for buf in self._dq_xyz_est],
                "rpy_est": [list(buf) for buf in self._rpy_est],
                "t_cmd": list(self._t_cmd),
                "cmd": [
                    list(self._cmd_x),
                    list(self._cmd_y),
                    list(self._cmd_z),
                    list(self._cmd_yaw),
                    list(self._cmd_gait),
                ],
                "status": dict(self._status),
                "topic_available": dict(self._topic_available),
                "topic_rate": dict(self._topic_rate),
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


def _get_fresh_status(status_map: Dict[str, Tuple[bool, float]], key: str, now: float, timeout: float):
    if key not in status_map:
        return None
    val, t = status_map[key]
    if (now - t) > timeout:
        return None
    return bool(val)


def render_status(snapshot: Dict[str, object], timeout_s: float) -> None:
    now = time.monotonic()
    timeout = float(timeout_s)
    status_map = snapshot["status"]

    labels = [
        ("inekf", "State Estimator"),
        ("rl_controller", "RL Controller"),
        ("intent_estimator", "Intent Estimator"),
        ("safety_stop", "Safety Stop"),
    ]

    status_view = {
        "inekf": _get_fresh_status(status_map, "inekf", now, timeout),
        "rl_controller": _get_fresh_status(status_map, "loco_ctrl", now, timeout),
        "intent_estimator": _get_fresh_status(status_map, "intent_estimator", now, timeout),
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
                if val:
                    st.success(f"{label}: running")
                else:
                    if key == "rl_controller":
                        st.warning(f"{label}: awaiting states")
                    elif key == "inekf":
                        st.warning(f"{label}: awaiting robot data")
                    elif key == "intent_estimator":
                        st.error(f"{label}: not running")
                    else:
                        st.error(f"{label}: not running")


def _style_sidebar_buttons(locomotion_active: bool) -> None:
    start_green = "#2ecc71"
    start_border = "#27ae60"
    stop_red = "#e74c3c"
    stop_border = "#c0392b"
    gray_bg = "#e0e0e0"
    gray_border = "#cfcfcf"
    gray_text = "#777777"
    js = f"""
    <script>
    const styleBtn = (label, active, bg, border) => {{
      const btns = Array.from(window.parent.document.querySelectorAll('button'));
      const btn = btns.find(b => b.innerText.trim() === label);
      if (!btn) return false;
      if (active) {{
        btn.disabled = false;
        btn.style.setProperty('background-color', bg, 'important');
        btn.style.setProperty('color', '#ffffff', 'important');
        btn.style.setProperty('border', '1px solid ' + border, 'important');
        btn.style.setProperty('opacity', '1.0', 'important');
        btn.style.setProperty('cursor', 'pointer', 'important');
      }} else {{
        btn.disabled = true;
        btn.style.setProperty('background-color', '{gray_bg}', 'important');
        btn.style.setProperty('color', '{gray_text}', 'important');
        btn.style.setProperty('border', '1px solid {gray_border}', 'important');
        btn.style.setProperty('opacity', '0.9', 'important');
        btn.style.setProperty('cursor', 'not-allowed', 'important');
      }}
      return true;
    }};

    const apply = () => {{
      let ok = true;
      ok = styleBtn('Start Control Stack', {str(not locomotion_active).lower()}, '{start_green}', '{start_border}') && ok;
      ok = styleBtn('Stop Control Stack', {str(locomotion_active).lower()}, '{stop_red}', '{stop_border}') && ok;
      ok = styleBtn('EMERGENCY STOP', true, '{stop_red}', '{stop_border}') && ok;
      if (!ok) setTimeout(apply, 100);
    }};
    apply();
    if (!window._dashBtnInterval) {{
      window._dashBtnInterval = setInterval(apply, 500);
    }}
    </script>
    """
    components.html(js, height=0)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Go2 Telemetry")
    st.title("Go2 Telemetry Dashboard")

    node = get_ros_node()
    snapshot = node.snapshot()

    st.sidebar.header("Settings")
    status_map = snapshot["status"]
    rl_ctrl_status = _get_fresh_status(
        status_map,
        "loco_ctrl",
        time.monotonic(),
        float(snapshot["status_timeout_s"]),
    )
    # Consider control stack active when the RL controller status stream is alive.
    locomotion_active = (rl_ctrl_status is not None) or launch_process_manager.is_running("control_stack")
    col_ctrl_start, col_ctrl_stop = st.sidebar.columns(2)
    if col_ctrl_start.button("Start Control Stack", key="start_ctrl", disabled=locomotion_active):
        ok, msg = launch_process_manager.start_launch("control_stack", "control_stack.launch.py")
        if ok:
            st.sidebar.info(msg)
        else:
            st.sidebar.warning(msg)
    if col_ctrl_stop.button("Stop Control Stack", key="stop_ctrl", disabled=not locomotion_active):
        ok, msg = launch_process_manager.stop_launch("control_stack")
        if ok:
            st.sidebar.info(msg)
        else:
            st.sidebar.warning(msg)
    if st.sidebar.button("EMERGENCY STOP", key="emergency_stop", use_container_width=True):
        client = st.session_state.get("svc_emergency_stop")
        if client is None or not client.service_is_ready():
            st.sidebar.warning("Emergency stop service not available")
        else:
            client.call_async(Trigger.Request())
            st.sidebar.error("Emergency stop requested")
    _style_sidebar_buttons(locomotion_active)

    st.subheader("Modules")
    render_status(snapshot, snapshot["status_timeout_s"])
    st.subheader("Topics")
    topic_available = snapshot.get("topic_available", {})
    topic_latest_msg = snapshot.get("topic_latest_msg", {})
    topic_names = snapshot.get("topic_names", {})
    topic_status_timeout = float(snapshot["status_timeout_s"])
    now = time.monotonic()

    def _topic_ok(topic_key: str, status_key: str) -> bool:
        return _get_fresh_status(snapshot["status"], status_key, now, topic_status_timeout) is True

    topic_rows = [
        ("lowstate", "lowstate"),
        ("qdq_est", "qdq_est"),
        ("intent_forward_backward", "intent_forward_backward"),
        ("intent_left_right", "intent_left_right"),
    ]
    badges = []
    for key, status_key in topic_rows:
        topic_key = topic_names.get(key, f"/{key}")
        ok = _topic_ok(topic_key, status_key)
        available = bool(topic_available.get(topic_key, False))
        latest_msg = topic_latest_msg.get(key)
        parts = [topic_key]
        if latest_msg is not None and ok:
            parts.append(f"msg={latest_msg}")
        elif available:
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


if __name__ == "__main__":
    main()

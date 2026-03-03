#!/usr/bin/env python3

import threading
import time
from collections import deque
from typing import Deque, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # type: ignore

from go2_msgs.msg import QDq
from unitree_go.msg import LowState


class QdqPlotter(Node):
    def __init__(self) -> None:
        super().__init__("qdq_plotter")

        self.declare_parameter("qdq_est_topic", "/qdq_est")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("history_sec", 10.0)
        self.declare_parameter("refresh_hz", 10.0)

        self.qdq_est_topic = str(self.get_parameter("qdq_est_topic").value)
        self.lowstate_topic = str(self.get_parameter("lowstate_topic").value)
        self.history_sec = float(self.get_parameter("history_sec").value)
        self.refresh_hz = float(self.get_parameter("refresh_hz").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_est = self.create_subscription(
            QDq, self.qdq_est_topic, self.on_qdq_est, qos
        )
        self.sub_lowstate = self.create_subscription(
            LowState, self.lowstate_topic, self.on_lowstate, qos
        )

        self._lock = threading.Lock()
        self._t0_q = None
        self._t_q: Deque[float] = deque()
        self._q_xyz: Tuple[Deque[float], Deque[float], Deque[float]] = (
            deque(),
            deque(),
            deque(),
        )
        self._t0_force = None
        self._t_force: Deque[float] = deque()
        self._foot_force: Tuple[Deque[float], Deque[float], Deque[float], Deque[float]] = (
            deque(),
            deque(),
            deque(),
            deque(),
        )
        self._imu_acc: Tuple[Deque[float], Deque[float], Deque[float]] = (
            deque(),
            deque(),
            deque(),
        )

    def _trim_history(
        self, t_buf: Deque[float], series: Tuple[Deque[float], ...]
    ) -> None:
        if self.history_sec <= 0.0:
            return
        while t_buf and (t_buf[-1] - t_buf[0] > self.history_sec):
            t_buf.popleft()
            for buf in series:
                buf.popleft()

    def on_qdq_est(self, msg: QDq) -> None:
        t = float(msg.sim_time)
        with self._lock:
            if self._t0_q is None:
                self._t0_q = t
            t_rel = t - self._t0_q
            self._t_q.append(t_rel)
            self._q_xyz[0].append(float(msg.q[0]))
            self._q_xyz[1].append(float(msg.q[1]))
            self._q_xyz[2].append(float(msg.q[2]))
            self._trim_history(self._t_q, self._q_xyz)

    def on_lowstate(self, msg: LowState) -> None:
        t = time.monotonic()
        with self._lock:
            if self._t0_force is None:
                self._t0_force = t
            t_rel = t - self._t0_force
            self._t_force.append(t_rel)
            for i in range(4):
                self._foot_force[i].append(float(msg.foot_force[i]))
            self._imu_acc[0].append(float(msg.imu_state.accelerometer[0]))
            self._imu_acc[1].append(float(msg.imu_state.accelerometer[1]))
            self._imu_acc[2].append(float(msg.imu_state.accelerometer[2]))
            self._trim_history(self._t_force, self._foot_force + self._imu_acc)

    def run_plot(self) -> None:
        window_sec = 10.0
        backend = str(matplotlib.get_backend()).lower()
        # Treat only known non-window backends as headless.
        non_gui_backends = {
            "agg",
            "cairo",
            "pdf",
            "pgf",
            "ps",
            "svg",
            "template",
            "module://matplotlib_inline.backend_inline",
        }
        if backend in non_gui_backends:
            self.get_logger().error(
                f"Matplotlib backend '{backend}' has no window support; cannot open plot windows. "
                "Run in a desktop session (DISPLAY set) and use a GUI backend like TkAgg/QtAgg."
            )
            return

        fig_qdq, axes_qdq = plt.subplots(2, 1, sharex=True)
        fig_qdq.suptitle("EKF State Estimation")

        line_q_x = axes_qdq[0].plot([], [], label="x-position")[0]
        line_q_y = axes_qdq[0].plot([], [], label="y-position")[0]
        line_q_z = axes_qdq[1].plot([], [], label="z-position")[0]

        axes_qdq[0].set_ylabel("x, y (m)")
        axes_qdq[0].legend(loc="upper right")
        axes_qdq[1].set_ylabel("z (m)")
        axes_qdq[1].set_xlabel("time (s)")
        axes_qdq[1].legend(loc="upper right")
        axes_qdq[0].set_ylim(0.0, 0.5)
        axes_qdq[1].set_ylim(0.0, 0.5)

        def update_qdq(_):
            with self._lock:
                t_q = list(self._t_q)
                qx, qy, qz = map(list, self._q_xyz)

            artists = [line_q_x, line_q_y, line_q_z]
            if not t_q:
                return artists

            line_q_x.set_data(t_q, qx)
            line_q_y.set_data(t_q, qy)
            line_q_z.set_data(t_q, qz)

            t_max = max(t_q)
            t_min = max(t_max - window_sec, 0.0)
            for ax in axes_qdq:
                ax.set_xlim(t_min, max(t_max, t_min + 1e-3))

            return artists

        interval_ms = int(1000.0 / max(self.refresh_hz, 1.0))
        self._anim_qdq = FuncAnimation(fig_qdq, update_qdq, interval=interval_ms)

        fig_force, axes_force = plt.subplots(2, 2, sharex=True)
        fig_force.suptitle("Foot Force")
        foot_names = ["FR", "FL", "RR", "RL"]
        force_lines = []
        for i in range(4):
            ax = axes_force[i // 2][i % 2]
            ax.set_xlabel("time (s)")
            line = ax.plot([], [], label=f"{foot_names[i]} (foot_force[{i}])")[0]
            ax.legend(loc="upper right")
            ax.set_ylim(0.0, 100.0)
            force_lines.append(line)

        def update_force(_):
            with self._lock:
                t_force = list(self._t_force)
                foot_force = [list(buf) for buf in self._foot_force]

            if not t_force:
                return force_lines

            t_max = max(t_force)
            t_min = max(t_max - window_sec, 0.0)
            for i in range(4):
                force_lines[i].set_data(t_force, foot_force[i])
                ax = axes_force[i // 2][i % 2]
                ax.set_xlim(t_min, max(t_max, t_min + 1e-3))
            return force_lines

        self._anim_force = FuncAnimation(fig_force, update_force, interval=interval_ms)

        fig_imu, axes_imu = plt.subplots(3, 1, sharex=True)
        fig_imu.suptitle("IMU Linear Acceleration")
        imu_axis_labels = ["x", "y", "z"]
        imu_lines = []
        for i in range(3):
            ax = axes_imu[i]
            line = ax.plot([], [], label=f"acc_{imu_axis_labels[i]}")[0]
            ax.set_ylabel("m/s^2")
            ax.legend(loc="upper right")
            if i == 2:
                ax.set_xlabel("time (s)")
            imu_lines.append(line)

        def update_imu(_):
            with self._lock:
                t_imu = list(self._t_force)
                imu_acc = [list(buf) for buf in self._imu_acc]

            if not t_imu:
                return imu_lines

            t_max = max(t_imu)
            t_min = max(t_max - window_sec, 0.0)
            for i in range(3):
                imu_lines[i].set_data(t_imu, imu_acc[i])
                ax = axes_imu[i]
                ax.set_xlim(t_min, max(t_max, t_min + 1e-3))
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            return imu_lines

        self._anim_imu = FuncAnimation(fig_imu, update_imu, interval=interval_ms)
        plt.show()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QdqPlotter()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run_plot()
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

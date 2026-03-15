#!/usr/bin/env python3

from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from go2_msgs.msg import LocomotionCmd
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32
from unitree_go.msg import LowCmd, LowState

from locomotion_controller.standup_init import (
    HEAD0,
    HEAD1,
    LOW_LEVEL,
    POS_STOP_F,
    VEL_STOP_F,
    get_crc,
)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _quat_to_rot_wb(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _zeros_for_shape(shape: list[Any]) -> np.ndarray:
    out_shape: list[int] = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            out_shape.append(d)
        else:
            out_shape.append(1)
    return np.zeros(out_shape, dtype=np.float32)


class PolicyControllerNode(Node):
    STATUS_IDLE = 0
    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_LOWSTATE = 2
    STATUS_WAITING_FOR_STANDING_INIT = 3

    def __init__(self):
        super().__init__("policy_controller")

        share_dir = Path(get_package_share_directory("locomotion_controller"))
        default_policy_dir = share_dir / "config" / "policy_dir"

        self.policy_dir = default_policy_dir.expanduser()
        self.deploy_cfg = self._load_deploy_cfg(self.policy_dir / "params" / "deploy.yaml")
        self.lowstate_topic = "/lowstate"
        self.locomotion_cmd_topic = "/locomotion_cmd"
        self.lowcmd_topic = "/lowcmd"

        self.control_hz = 50.0          # Policy loop frequency
        self.control_period = 1.0 / max(self.control_hz, 1.0)

        self.status_hz = 10.0           # Status topic publish rate in Hz.
        self.cmd_timeout_s = 0.5        # Zero commanded velocity if /locomotion_cmd is older than this.
        self.lowstate_timeout_s = 0.1   # Stop running policy if /lowstate is older than this.

        self._load_action_cfg()
        self._load_observation_cfg()
        self._load_command_ranges()

        self.last_raw_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_lowstate: LowState | None = None
        self.last_lowstate_time_ns: int | None = None
        self.last_locomotion_cmd: LocomotionCmd | None = None
        self.last_cmd_time_ns: int | None = None
        self.standing_ready = False
        self.wait_logged = False
        self.lowstate_wait_logged = False
        self.lowstate_stale_logged = False
        self.ready_sent = False
        self.running_logged = False
        self.motor_state_wait_logged = False
        self.status_code = -1

        self.session, self.input_name, self.output_name = self._create_onnx_session(
            self.policy_dir / "exported" / "policy.onnx"
        )
        self.recurrent_inputs: dict[str, np.ndarray] = {}
        for inp in self.session.get_inputs():
            if inp.name != self.input_name:
                self.recurrent_inputs[inp.name] = _zeros_for_shape(list(inp.shape))

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        command_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub_lowcmd = self.create_publisher(LowCmd, self.lowcmd_topic, command_qos)
        self.pub_status = self.create_publisher(Int32, "/status/loco_ctrl", status_qos)
        self._set_status(self.STATUS_IDLE)

        self.sub_lowstate = self.create_subscription(
            LowState, self.lowstate_topic, self.on_lowstate, sensor_qos
        )
        self.sub_cmd = self.create_subscription(
            LocomotionCmd, self.locomotion_cmd_topic, self.on_locomotion_cmd, sensor_qos
        )
        self.sub_standing = self.create_subscription(
            Int32, "/status/standing_init", self.on_standing_status, status_qos
        )

        # ROS callback timer
        self.timer = self.create_timer(self.control_period, self.on_timer)  # Policy loop
        self.status_timer = self.create_timer(1.0 / max(self.status_hz, 1.0), self.on_status_timer) # Status update loop

    def _publish_status(self) -> None:
        self.pub_status.publish(Int32(data=int(self.status_code)))

    def _set_status(self, status_code: int) -> None:
        status_code = int(status_code)
        if status_code == self.status_code:
            return
        self.status_code = status_code
        self._publish_status()

    def on_status_timer(self) -> None:
        self._publish_status()

    def _load_deploy_cfg(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError(f"deploy.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise RuntimeError("deploy.yaml must be a YAML mapping.")
        return cfg

    def _load_action_cfg(self) -> None:
        actions_cfg = self.deploy_cfg.get("actions", {})
        if "JointPositionAction" not in actions_cfg:
            raise RuntimeError("deploy.yaml actions must contain JointPositionAction.")
        cfg = actions_cfg["JointPositionAction"]
        self.action_scale = np.array(cfg.get("scale", []), dtype=np.float32)
        self.action_offset = np.array(cfg.get("offset", []), dtype=np.float32)
        self.action_clip = cfg.get("clip", None)
        self.joint_ids_map = [int(v) for v in self.deploy_cfg["joint_ids_map"]]
        self.default_joint_pos = np.array(self.deploy_cfg["default_joint_pos"], dtype=np.float32)
        self.joint_stiffness = np.array(self.deploy_cfg["stiffness"], dtype=np.float32)
        self.joint_damping = np.array(self.deploy_cfg["damping"], dtype=np.float32)
        self.action_dim = len(self.joint_ids_map)

        if self.action_scale.size == 0:
            self.action_scale = np.ones(self.action_dim, dtype=np.float32)
        if self.action_offset.size == 0:
            self.action_offset = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_scale.size != self.action_dim or self.action_offset.size != self.action_dim:
            raise RuntimeError("Action scale/offset size must match joint_ids_map length.")

    def _load_observation_cfg(self) -> None:
        raw_obs_cfg = self.deploy_cfg.get("observations", {})

        self.obs_terms: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for name, cfg in raw_obs_cfg.items():
            term_cfg = dict(cfg or {})
            scale = term_cfg.get("scale", None)
            clip = term_cfg.get("clip", None)
            history_length = int(term_cfg.get("history_length", 1))
            self.obs_terms[name] = {
                "params": term_cfg.get("params", {}) or {},
                "scale": np.array(scale, dtype=np.float32) if scale is not None else None,
                "clip": clip,
                "history_length": max(1, history_length),
                "buffer": deque(),
            }

    def _load_command_ranges(self) -> None:
        base_vel = self.deploy_cfg.get("commands", {}).get("base_velocity", {})
        ranges = base_vel.get("ranges", {})
        self.cmd_lin_x = tuple(ranges.get("lin_vel_x", [-0.5, 1.0]))
        self.cmd_lin_y = tuple(ranges.get("lin_vel_y", [-0.5, 0.5]))
        self.cmd_ang_z = tuple(ranges.get("ang_vel_z", [-1.0, 1.0]))

    def _create_onnx_session(self, policy_path: Path):
        if not policy_path.exists():
            raise RuntimeError(f"policy.onnx not found: {policy_path}")
        self.get_logger().info(f"Loading ONNX policy from {policy_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess = ort.InferenceSession(
            str(policy_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        return sess, inputs[0].name, outputs[0].name

    def on_lowstate(self, msg: LowState) -> None:
        self.last_lowstate = msg
        self.last_lowstate_time_ns = self.get_clock().now().nanoseconds
        self.lowstate_stale_logged = False

    def on_locomotion_cmd(self, msg: LocomotionCmd) -> None:
        self.last_locomotion_cmd = msg
        self.last_cmd_time_ns = self.get_clock().now().nanoseconds

    def on_standing_status(self, msg: Int32) -> None:
        self.standing_ready = int(msg.data) == 3

    def on_timer(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if not self.standing_ready:
            if not self.wait_logged:
                self.get_logger().info("policy_controller waiting for /status/standing_init...")
                self.wait_logged = True
            self._set_status(self.STATUS_WAITING_FOR_STANDING_INIT)
            return
        if self.last_lowstate is None:
            if not self.lowstate_wait_logged:
                self.get_logger().info("policy_controller waiting for /lowstate...")
                self.lowstate_wait_logged = True
            self._set_status(self.STATUS_WAITING_FOR_LOWSTATE)
            return
        if self.last_lowstate_time_ns is None or (
            self.lowstate_timeout_s > 0.0
            and (now_ns - self.last_lowstate_time_ns) * 1e-9 > self.lowstate_timeout_s
        ):
            if not self.lowstate_stale_logged:
                age_s = (
                    float("inf")
                    if self.last_lowstate_time_ns is None
                    else (now_ns - self.last_lowstate_time_ns) * 1e-9
                )
                self.get_logger().warning(
                    f"policy_controller waiting for fresh /lowstate "
                    f"(age={age_s:.3f}s, timeout={self.lowstate_timeout_s:.3f}s)."
                )
                self.lowstate_stale_logged = True
            self._set_status(self.STATUS_WAITING_FOR_LOWSTATE)
            return
        if len(self.last_lowstate.motor_state) <= max(self.joint_ids_map):
            if not self.motor_state_wait_logged:
                self.get_logger().warning("policy_controller waiting for full /lowstate motor_state.")
                self.motor_state_wait_logged = True
            self._set_status(self.STATUS_WAITING_FOR_LOWSTATE)
            return

        obs = self._build_observation(self.last_lowstate)
        raw_action = self._infer_policy(obs)
        processed_action = self._process_action(raw_action)
        cmd_msg = self._build_lowcmd(processed_action)
        self.pub_lowcmd.publish(cmd_msg)

        self.last_raw_action = raw_action
        if not self.ready_sent:
            if not self.running_logged:
                self.get_logger().info("policy_controller running")
                self.running_logged = True
            self.ready_sent = True
        self._set_status(self.STATUS_RUNNING)

    def _compute_term(self, name: str, lowstate: LowState) -> np.ndarray:
        if name == "base_ang_vel":
            g = lowstate.imu_state.gyroscope
            return np.array([g[0], g[1], g[2]], dtype=np.float32)

        if name == "projected_gravity":
            q = lowstate.imu_state.quaternion
            rot_wb = _quat_to_rot_wb(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            gravity_b = rot_wb.T @ gravity_w
            return gravity_b.astype(np.float32)

        if name == "velocity_commands":
            now_ns = self.get_clock().now().nanoseconds
            stale = (
                self.last_locomotion_cmd is None
                or self.last_cmd_time_ns is None
                or (now_ns - self.last_cmd_time_ns) * 1e-9 > self.cmd_timeout_s
            )
            if stale:
                return np.zeros(3, dtype=np.float32)
            cmd = self.last_locomotion_cmd
            return np.array(
                [
                    _clamp(float(cmd.x_vel), float(self.cmd_lin_x[0]), float(self.cmd_lin_x[1])),
                    _clamp(float(cmd.y_vel), float(self.cmd_lin_y[0]), float(self.cmd_lin_y[1])),
                    _clamp(float(cmd.yaw_rate), float(self.cmd_ang_z[0]), float(self.cmd_ang_z[1])),
                ],
                dtype=np.float32,
            )

        if name == "joint_pos_rel":
            q = np.array(
                [float(lowstate.motor_state[j].q) for j in self.joint_ids_map], dtype=np.float32
            )
            return q - self.default_joint_pos

        if name == "joint_vel_rel":
            return np.array(
                [float(lowstate.motor_state[j].dq) for j in self.joint_ids_map], dtype=np.float32
            )

        if name == "last_action":
            return self.last_raw_action.astype(np.float32)

        raise RuntimeError(f"Unsupported observation term in deploy.yaml: {name}")

    def _apply_term_post(self, term_val: np.ndarray, term_cfg: dict[str, Any]) -> np.ndarray:
        out = term_val.copy()
        clip = term_cfg["clip"]
        scale = term_cfg["scale"]
        if clip is not None:
            out = np.clip(out, float(clip[0]), float(clip[1]))
        if scale is not None:
            out = out * scale
        return out.astype(np.float32)

    def _build_observation(self, lowstate: LowState) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for name, term in self.obs_terms.items():
            val = self._compute_term(name, lowstate)
            val = self._apply_term_post(val, term)
            buff: deque[np.ndarray] = term["buffer"]
            hist = term["history_length"]
            if not buff:
                for _ in range(hist):
                    buff.append(val.copy())
            else:
                buff.append(val.copy())
                while len(buff) > hist:
                    buff.popleft()
            for b in buff:
                pieces.append(b)
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def _infer_policy(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        feeds: dict[str, np.ndarray] = {self.input_name: obs_batch}
        feeds.update(self.recurrent_inputs)
        out_vals = self.session.run(None, feeds)

        # Update recurrent state buffers if they exist.
        for out_meta, out_val in zip(self.session.get_outputs(), out_vals):
            mapped = out_meta.name.replace("_out", "_in")
            if mapped in self.recurrent_inputs:
                self.recurrent_inputs[mapped] = np.asarray(out_val, dtype=np.float32)

        raw_action = np.asarray(out_vals[0], dtype=np.float32).reshape(-1)
        if raw_action.size != self.action_dim:
            self.get_logger().warn(
                f"Policy output dim {raw_action.size} != expected {self.action_dim}; trunc/pad applied."
            )
            fixed = np.zeros(self.action_dim, dtype=np.float32)
            n = min(self.action_dim, raw_action.size)
            fixed[:n] = raw_action[:n]
            raw_action = fixed
        return raw_action

    def _process_action(self, raw_action: np.ndarray) -> np.ndarray:
        out = raw_action * self.action_scale + self.action_offset
        if self.action_clip is not None:
            out = np.clip(out, float(self.action_clip[0]), float(self.action_clip[1]))
        return out.astype(np.float32)

    def _build_lowcmd(self, processed_action: np.ndarray) -> LowCmd:
        msg = LowCmd()
        msg.head[0] = HEAD0
        msg.head[1] = HEAD1
        msg.level_flag = LOW_LEVEL
        msg.gpio = 0

        for i in range(20):
            msg.motor_cmd[i].mode = 0x01
            msg.motor_cmd[i].q = float(POS_STOP_F)
            msg.motor_cmd[i].dq = float(VEL_STOP_F)
            msg.motor_cmd[i].kp = 0.0
            msg.motor_cmd[i].kd = 0.0
            msg.motor_cmd[i].tau = 0.0

        for i, sdk_idx in enumerate(self.joint_ids_map):
            msg.motor_cmd[sdk_idx].mode = 0x01
            msg.motor_cmd[sdk_idx].q = float(processed_action[i])
            msg.motor_cmd[sdk_idx].dq = 0.0
            msg.motor_cmd[sdk_idx].kp = float(self.joint_stiffness[i])
            msg.motor_cmd[sdk_idx].kd = float(self.joint_damping[i])
            msg.motor_cmd[sdk_idx].tau = 0.0

        msg.crc = get_crc(msg)
        return msg


def main():
    rclpy.init()
    node = PolicyControllerNode()
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

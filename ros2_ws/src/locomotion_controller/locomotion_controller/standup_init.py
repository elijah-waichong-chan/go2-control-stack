#!/usr/bin/env python3

import ctypes
import struct
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int32
from unitree_go.msg import LowCmd, LowState


LOW_LEVEL = 0xFF
HEAD0 = 0xFE
HEAD1 = 0xEF
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

DEFAULT_CROUCH_POS = [
    0.0,
    1.36,
    -2.65,
    0.0,
    1.36,
    -2.65,
    0.0,
    1.36,
    -2.65,
    0.0,
    1.36,
    -2.65,
]
DEFAULT_TARGET_POS = [
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
]
# Keep the fallback values identical to the pasted C++ node.
FALLBACK_TARGET_POS = [
    0.0,
    0.67,
    -1.3,
    0.0,
    0.67,
    -1.3,
    0.0,
    0.67,
    -1.3,
    0.0,
    0.67,
    -1.3,
]


class MotorCmdRaw(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint8),
        ("q", ctypes.c_float),
        ("dq", ctypes.c_float),
        ("tau", ctypes.c_float),
        ("kp", ctypes.c_float),
        ("kd", ctypes.c_float),
        ("reserve", ctypes.c_uint32 * 3),
    ]


class BmsCmdRaw(ctypes.Structure):
    _fields_ = [
        ("off", ctypes.c_uint8),
        ("reserve", ctypes.c_uint8 * 3),
    ]


class LowCmdRaw(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint8 * 2),
        ("level_flag", ctypes.c_uint8),
        ("frame_reserve", ctypes.c_uint8),
        ("sn", ctypes.c_uint32 * 2),
        ("version", ctypes.c_uint32 * 2),
        ("bandwidth", ctypes.c_uint16),
        ("motor_cmd", MotorCmdRaw * 20),
        ("bms", BmsCmdRaw),
        ("wireless_remote", ctypes.c_uint8 * 40),
        ("led", ctypes.c_uint8 * 12),
        ("fan", ctypes.c_uint8 * 2),
        ("gpio", ctypes.c_uint8),
        ("reserve", ctypes.c_uint32),
        ("crc", ctypes.c_uint32),
    ]


def crc32_core(words: List[int]) -> int:
    poly = 0x04C11DB7
    crc32 = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc32 & 0x80000000:
                crc32 = ((crc32 << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc32 = (crc32 << 1) & 0xFFFFFFFF
            if data & xbit:
                crc32 ^= poly
            xbit >>= 1
    return crc32 & 0xFFFFFFFF


def get_crc(msg: LowCmd) -> int:
    raw = LowCmdRaw()
    raw.head[:] = list(msg.head)
    raw.level_flag = int(msg.level_flag)
    raw.frame_reserve = int(msg.frame_reserve)
    raw.sn[:] = list(msg.sn)
    raw.version[:] = list(msg.version)
    raw.bandwidth = int(msg.bandwidth)

    for i in range(20):
        src = msg.motor_cmd[i]
        dst = raw.motor_cmd[i]
        dst.mode = int(src.mode)
        dst.q = float(src.q)
        dst.dq = float(src.dq)
        dst.tau = float(src.tau)
        dst.kp = float(src.kp)
        dst.kd = float(src.kd)
        dst.reserve[:] = list(src.reserve)

    raw.bms.off = int(msg.bms_cmd.off)
    raw.bms.reserve[:] = list(msg.bms_cmd.reserve)
    raw.wireless_remote[:] = list(msg.wireless_remote)
    raw.led[:] = list(msg.led)
    raw.fan[:] = list(msg.fan)
    raw.gpio = int(msg.gpio)
    raw.reserve = int(msg.reserve)

    size_without_crc = ctypes.sizeof(LowCmdRaw) - ctypes.sizeof(ctypes.c_uint32)
    payload = ctypes.string_at(ctypes.addressof(raw), size_without_crc)
    words = list(struct.unpack(f"<{len(payload) // 4}I", payload))
    return crc32_core(words)


class StandUpInitNode(Node):
    STATUS_RUNNING = 1
    STATUS_WAITING_FOR_LOWSTATE = 2
    STATUS_COMPLETE = 3

    def __init__(self) -> None:
        super().__init__("stand_up_init")

        self.declare_parameter("kp", 60.0)
        self.declare_parameter("kd", 5.0)
        self.declare_parameter("crouch_time_s", 1.0)
        self.declare_parameter("crouch_hold_s", 0.2)
        self.declare_parameter("ramp_time_s", 2.0)
        self.declare_parameter("start_delay_s", 0.5)
        self.declare_parameter("command_hz", 500.0)
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("exit_after_standing_s", 2.0)
        self.declare_parameter("crouch_pos", DEFAULT_CROUCH_POS)
        self.declare_parameter("target_pos", DEFAULT_TARGET_POS)

        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)
        self.crouch_time_s = max(0.01, float(self.get_parameter("crouch_time_s").value))
        self.crouch_hold_s = max(0.0, float(self.get_parameter("crouch_hold_s").value))
        self.ramp_time_s = max(0.01, float(self.get_parameter("ramp_time_s").value))
        self.start_delay_s = max(0.0, float(self.get_parameter("start_delay_s").value))
        self.command_hz = max(1.0, float(self.get_parameter("command_hz").value))
        self.status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.exit_after_standing_s = max(
            0.0, float(self.get_parameter("exit_after_standing_s").value)
        )

        self.crouch_pos = [float(v) for v in self.get_parameter("crouch_pos").value]
        if len(self.crouch_pos) != 12:
            self.get_logger().warn(
                "crouch_pos must have 12 elements (FR,FL,RR,RL). Using defaults."
            )
            self.crouch_pos = list(DEFAULT_CROUCH_POS)

        self.target_pos = [float(v) for v in self.get_parameter("target_pos").value]
        if len(self.target_pos) != 12:
            self.get_logger().warn(
                "target_pos must have 12 elements (FR,FL,RR,RL). Using defaults."
            )
            self.target_pos = list(FALLBACK_TARGET_POS)

        qos = QoSProfile(
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

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", qos)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.on_lowstate, qos
        )
        self.status_pub = self.create_publisher(Int32, "/status/standing_init", status_qos)
        self.ctrl_status_sub = self.create_subscription(
            Int32, "/status/loco_ctrl", self.on_ctrl_status, status_qos
        )

        self.last_state: Optional[LowState] = None
        self.have_state = False
        self.started = False
        self.start_time = self.get_clock().now()
        self.start_pos = [0.0] * 12
        self.status_sent = False
        self.status_code = self.STATUS_WAITING_FOR_LOWSTATE
        self.ctrl_running = False
        self.standing_done_time_ns: Optional[int] = None
        self.shutdown_requested = False

        self.timer = self.create_timer(1.0 / self.command_hz, self.on_timer)
        self.status_timer = self.create_timer(1.0 / self.status_hz, self.on_status_timer)
        self.get_logger().info("stand_up_init running")

    def _set_status(self, status_code: int) -> None:
        self.status_code = int(status_code)

    def on_status_timer(self) -> None:
        self.status_pub.publish(Int32(data=int(self.status_code)))

    def _request_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.get_logger().info(reason)
        self.timer.cancel()
        self.destroy_node()
        if hasattr(rclpy, "try_shutdown"):
            rclpy.try_shutdown()
        elif rclpy.ok():
            rclpy.shutdown()

    def on_lowstate(self, msg: LowState) -> None:
        self.last_state = msg
        self.have_state = True

    def on_timer(self) -> None:
        if not self.have_state or self.last_state is None:
            self._set_status(self.STATUS_WAITING_FOR_LOWSTATE)
            return

        if not self.started:
            for i in range(12):
                self.start_pos[i] = float(self.last_state.motor_state[i].q)
            self.start_time = self.get_clock().now()
            self.started = True
        if self.status_sent:
            self._set_status(self.STATUS_COMPLETE)
        else:
            self._set_status(self.STATUS_RUNNING)

        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9 - self.start_delay_s
        if t < 0.0:
            return

        t_crouch_end = self.crouch_time_s
        t_hold_end = t_crouch_end + self.crouch_hold_s
        t_stand_end = t_hold_end + self.ramp_time_s

        if t >= t_stand_end and not self.status_sent:
            self._set_status(self.STATUS_COMPLETE)
            self.status_sent = True
            self.standing_done_time_ns = self.get_clock().now().nanoseconds

        if (
            self.status_sent
            and (not self.ctrl_running)
            and self.standing_done_time_ns is not None
            and (self.get_clock().now().nanoseconds - self.standing_done_time_ns) * 1e-9
            > self.exit_after_standing_s
        ):
            self._request_shutdown(
                "standing_init done; RL running status not received in time, stopping stand_up_init."
            )
            return

        cmd = LowCmd()
        cmd.head[0] = HEAD0
        cmd.head[1] = HEAD1
        cmd.level_flag = LOW_LEVEL
        cmd.gpio = 0

        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = float(POS_STOP_F)
            cmd.motor_cmd[i].dq = float(VEL_STOP_F)
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        for i in range(12):
            if t < t_crouch_end:
                a = max(0.0, min(1.0, t / self.crouch_time_s))
                q_des = (1.0 - a) * self.start_pos[i] + a * self.crouch_pos[i]
            elif t < t_hold_end:
                q_des = self.crouch_pos[i]
            elif t < t_stand_end:
                a = max(0.0, min(1.0, (t - t_hold_end) / self.ramp_time_s))
                q_des = (1.0 - a) * self.crouch_pos[i] + a * self.target_pos[i]
            else:
                q_des = self.target_pos[i]

            cmd.motor_cmd[i].q = float(q_des)
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = float(self.kp)
            cmd.motor_cmd[i].kd = float(self.kd)
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].mode = 0x01

        cmd.crc = get_crc(cmd)
        self.lowcmd_pub.publish(cmd)

    def on_ctrl_status(self, msg: Int32) -> None:
        if int(msg.data) != 1 or self.ctrl_running:
            return
        self.ctrl_running = True
        self._request_shutdown("locomotion_controller is running; stopping stand_up_init.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StandUpInitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.shutdown_requested:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

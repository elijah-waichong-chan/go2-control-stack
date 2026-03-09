#!/usr/bin/env python3

import json
import re
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from go2_msgs.msg import ArmAngles
from std_msgs.msg import Int32
from unitree_arm.msg import ArmString


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            txt = value.strip()
            if txt.startswith(("0x", "0X")):
                return int(txt, 16)
            return int(float(txt))
    except (TypeError, ValueError):
        pass
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (TypeError, ValueError):
        return default


class ArmFeedbackParser(Node):
    STATUS_PUBLISHING = 1
    STATUS_WAITING_FOR_FEEDBACK = 2

    def __init__(self) -> None:
        super().__init__("arm_feedback_parser")

        self.declare_parameter("feedback_topic", "/arm_Feedback")
        self.declare_parameter("arm_angles_topic", "/arm_angles")
        self.declare_parameter("status_hz", 10.0)

        feedback_topic = str(self.get_parameter("feedback_topic").value)
        arm_angles_topic = str(self.get_parameter("arm_angles_topic").value)
        status_hz = max(1.0, float(self.get_parameter("status_hz").value))

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_feedback = self.create_subscription(
            ArmString, feedback_topic, self.on_feedback, qos
        )
        self.pub_angles = self.create_publisher(ArmAngles, arm_angles_topic, qos)
        self.pub_status = self.create_publisher(Int32, "/status/arm_parser", status_qos)
        self.status_code = self.STATUS_WAITING_FOR_FEEDBACK
        self.status_timer = self.create_timer(1.0 / status_hz, self.on_status_timer)
        self.get_logger().info(f"arm_feedback_parser running: {feedback_topic} -> {arm_angles_topic}")

    def _set_status(self, status_code: int) -> None:
        self.status_code = int(status_code)

    def on_status_timer(self) -> None:
        self.pub_status.publish(Int32(data=int(self.status_code)))

    def on_feedback(self, msg: ArmString) -> None:
        parsed = self._parse_payload(msg.data)
        if parsed is None:
            return

        funcode = _to_int(parsed.get("funcode"), default=-1)
        if funcode == 3:
            return
        if funcode != 1:
            return

        out = ArmAngles()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "arm_feedback_parser"
        out.seq = _to_int(parsed.get("seq"), default=0) & 0xFFFFFFFF
        out.address = _to_int(parsed.get("address"), default=0) & 0xFF
        out.funcode = funcode & 0xFF
        out.angle_deg = self._extract_angles(parsed)
        self.pub_angles.publish(out)
        self._set_status(self.STATUS_PUBLISHING)

    def _parse_payload(self, text: str) -> Optional[Dict[str, Any]]:
        raw = text.strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return self._parse_kv_payload(raw)

    def _parse_kv_payload(self, raw: str) -> Optional[Dict[str, Any]]:
        # Accept strings like: seq=1,address=1,funcode=1,angle0=...
        pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", raw)
        if not pairs:
            return None

        parsed: Dict[str, Any] = {}
        for key, value in pairs:
            if "." in value or "e" in value.lower():
                parsed[key] = _to_float(value, 0.0)
            else:
                parsed[key] = _to_int(value, 0)
        return parsed

    def _extract_angles(self, parsed: Dict[str, Any]) -> List[float]:
        containers: List[Dict[str, Any]] = [parsed]
        data = parsed.get("data")
        if isinstance(data, dict):
            containers.append(data)

        for obj in containers:
            for array_key in ("angle_deg", "angles", "angle"):
                maybe_array = obj.get(array_key)
                if isinstance(maybe_array, (list, tuple)):
                    return self._normalize_angles(maybe_array)

            for pattern in ("servo{}_data", "angle{}", "joint{}", "servo{}"):
                values = [0.0] * 7
                found = False
                for i in range(7):
                    key = pattern.format(i)
                    if key in obj:
                        found = True
                        values[i] = _to_float(obj[key], 0.0)
                if found:
                    return values

            joint_id = obj.get("id")
            joint_angle = obj.get("angle")
            if joint_id is not None and joint_angle is not None:
                idx = _to_int(joint_id, -1)
                values = [0.0] * 7
                if 0 <= idx < 7:
                    values[idx] = _to_float(joint_angle, 0.0)
                return values

        return [0.0] * 7

    def _normalize_angles(self, values: Any) -> List[float]:
        result = [0.0] * 7
        for i, value in enumerate(values):
            if i >= 7:
                break
            result[i] = _to_float(value, 0.0)
        return result


def main() -> None:
    rclpy.init()
    node = ArmFeedbackParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

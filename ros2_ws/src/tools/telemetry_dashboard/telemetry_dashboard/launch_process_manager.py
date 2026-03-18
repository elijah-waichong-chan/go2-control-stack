#!/usr/bin/env python3

import os
import signal
import subprocess
import threading
import time
from typing import Sequence
from typing import Dict, Tuple


_LOCK = threading.Lock()
_PROCESSES: Dict[str, subprocess.Popen] = {}


def _alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def _cleanup_stale(name: str) -> None:
    proc = _PROCESSES.get(name)
    if proc is not None and not _alive(proc):
        _PROCESSES.pop(name, None)


def _start_process(
    name: str,
    command: Sequence[str],
    success_message: str,
) -> Tuple[bool, str]:
    _cleanup_stale(name)
    if name in _PROCESSES:
        return False, f"{name} already running"

    proc = subprocess.Popen(
        list(command),
        start_new_session=True,
    )
    _PROCESSES[name] = proc
    return True, success_message


def start_launch(
    name: str,
    package: str,
    launch_file: str,
    launch_args: Dict[str, str] | None = None,
) -> Tuple[bool, str]:
    with _LOCK:
        command = ["ros2", "launch", package, launch_file]
        if launch_args:
            for key, value in launch_args.items():
                command.append(f"{key}:={value}")
        return _start_process(
            name,
            command,
            f"started {package} {launch_file}",
        )


def start_node(
    name: str,
    package: str,
    executable: str,
    ros_args: Sequence[str] | None = None,
) -> Tuple[bool, str]:
    with _LOCK:
        command = ["ros2", "run", package, executable]
        if ros_args:
            command.extend(ros_args)
        return _start_process(
            name,
            command,
            f"started {package} {executable}",
        )


def start_rosbag_recording() -> Tuple[bool, str]:
    with _LOCK:
        bag_name = time.strftime("go2_data_%Y%m%d_%H%M%S")
        return _start_process(
            "rosbag_recording",
            [
                "ros2",
                "bag",
                "record",
                "-s",
                "mcap",
                "-o",
                bag_name,
                "/data/push_event",
                "/lowstate",
                "/arm_angles",
                "/joint_states",
                "/robot_description",
                "/tf",
                "/tf_static",
                # "/odometry/filtered",
                # "/locomotion_cmd"
            ],
            f"started rosbag recording to {bag_name}",
        )


def stop_launch(name: str) -> Tuple[bool, str]:
    with _LOCK:
        _cleanup_stale(name)
        proc = _PROCESSES.get(name)
        if proc is None:
            return False, f"{name} not running"

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            _PROCESSES.pop(name, None)
            return True, f"stopped {name}"

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=2.0)

        _PROCESSES.pop(name, None)
        return True, f"stopped {name}"


def is_running(name: str) -> bool:
    with _LOCK:
        _cleanup_stale(name)
        return name in _PROCESSES


def stop_all() -> None:
    for name in (
        "control_stack",
        "autonomy",
        "foxglove_bridge",
        "rosbag_recording",
        "arm_controller",
    ):
        stop_launch(name)

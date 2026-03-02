#!/usr/bin/env python3

import os
import signal
import subprocess
import threading
from typing import Dict, Tuple


_LOCK = threading.Lock()
_PROCESSES: Dict[str, subprocess.Popen] = {}


def _alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def _cleanup_stale(name: str) -> None:
    proc = _PROCESSES.get(name)
    if proc is not None and not _alive(proc):
        _PROCESSES.pop(name, None)


def start_launch(name: str, launch_file: str) -> Tuple[bool, str]:
    with _LOCK:
        _cleanup_stale(name)
        if name in _PROCESSES:
            return False, f"{name} already running"

        proc = subprocess.Popen(
            ["ros2", "launch", "locomotion_controller", launch_file],
            start_new_session=True,
        )
        _PROCESSES[name] = proc
        return True, f"started {launch_file}"


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
    for name in ("control_stack", "mujoco_robot"):
        stop_launch(name)

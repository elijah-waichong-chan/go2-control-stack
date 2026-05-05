#!/usr/bin/env python3

import os
from pathlib import Path
import psutil
import re
import signal
import subprocess
import threading
import time
from typing import Dict, Sequence, Tuple


_LOCK = threading.Lock()
_PROCESSES: Dict[str, subprocess.Popen] = {}
_ARM_CONTROLLER_MODES = {
    "pink_mode_switch": {
        "process_name": "arm_pink_controller_mode_switch",
        "command": [
            "ros2",
            "launch",
            "arm_pink_controller",
            "arm_pink_controller_mode_switch.launch.py",
        ],
        "success_message": (
            "started arm_pink_controller arm_pink_controller_mode_switch.launch.py"
        ),
    },
}
_COORDINATION_MODULE_MODES = {
    "autonomous": {
        "process_name": "coordination_module_autonomous",
        "executable": "autonomous_coordinator",
        "success_message": "started coordination_module autonomous_coordinator",
    },
    "teleop": {
        "process_name": "coordination_module_teleop",
        "executable": "teleop_coordinator",
        "success_message": "started coordination_module teleop_coordinator",
    },
}
ROSBAG_TOPICS: tuple[str, ...] = (
    "/data/push_event",
    "/lowstate",
    "/lowcmd",
    "/arm_task",
    "/d1_pink/current_z",
    "/d1_pink/enabled",
    "/d1_pink/mode",
    "/d1_pink/mode_switch_startup_complete",
    "/d1_pink/startup_complete",
    "/d1_pink/z_ref",
    "/d1_pink/z_velocity",
    "/arm/z_reference",
    "/arm/commanded_angles",
    "/arm/servo_feedback",
    "/arm/servo_command",
    "/direction_intent/front_back/label",
    "/direction_intent/left_right/label",
    "/direction_intent/left_right/scores_raw",
    "/direction_intent/left_right/scores_smoothed",
    "/direction_intent/up_down/label",
    "/direction_intent/up_down/scores_raw",
    "/direction_intent/up_down/scores_smoothed",
    "/locomotion_cmd",
    "/joint_states",
    "/robot_description",
    "/tf",
    "/tf_static",
)
FOXGLOVE_TOPICS: tuple[str, ...] = (
    "/lowstate",
    "/arm_task",
    "/d1_pink/current_z",
    "/d1_pink/enabled",
    "/d1_pink/mode",
    "/d1_pink/mode_switch_startup_complete",
    "/d1_pink/startup_complete",
    "/d1_pink/z_ref",
    "/d1_pink/z_velocity",
    "/locomotion_cmd",
)


def _resolve_repo_root() -> Path:
    override = os.environ.get("GO2_CONTROL_STACK_ROOT")
    if override:
        return Path(override).expanduser()

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "ros2_ws").is_dir() and (candidate / "README.md").is_file():
                return candidate

    return Path.cwd()


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
    cwd: str | os.PathLike[str] | None = None,
    log_path: Path | None = None,
) -> Tuple[bool, str]:
    _cleanup_stale(name)
    if name in _PROCESSES:
        return False, f"{name} already running"

    popen_kwargs = {
        "cwd": os.fspath(cwd) if cwd is not None else None,
        "start_new_session": True,
    }
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log_file:
                proc = subprocess.Popen(
                    list(command),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    **popen_kwargs,
                )
        else:
            proc = subprocess.Popen(
                list(command),
                **popen_kwargs,
            )
    except OSError as exc:
        return False, f"failed to start {name}: {exc}"

    time.sleep(0.3)
    if proc.poll() is not None:
        message = f"{name} exited immediately with code {proc.returncode}"
        if log_path is not None:
            message += f"; see {log_path}"
        return False, message

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


def start_foxglove_bridge(selected_topics: Sequence[str] | None = None) -> Tuple[bool, str]:
    topics = list(selected_topics) if selected_topics is not None else list(FOXGLOVE_TOPICS)
    if not topics:
        return False, "select at least one Foxglove topic"

    topic_whitelist = "[" + ",".join(f'"^{re.escape(topic)}$"' for topic in topics) + "]"
    return start_launch(
        "foxglove_bridge",
        "foxglove_bridge",
        "foxglove_bridge_launch.xml",
        launch_args={"topic_whitelist": topic_whitelist},
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


def start_rosbag_recording(selected_topics: Sequence[str] | None = None) -> Tuple[bool, str]:
    with _LOCK:
        repo_root = _resolve_repo_root()
        bag_name = time.strftime("go2_data_%Y%m%d_%H%M%S")
        bag_dir = repo_root / "bag_data"
        bag_path = bag_dir / bag_name
        bag_dir.mkdir(parents=True, exist_ok=True)
        topics = list(selected_topics) if selected_topics is not None else list(ROSBAG_TOPICS)
        if not topics:
            return False, "select at least one rosbag topic"
        invalid_topics = [topic for topic in topics if topic not in ROSBAG_TOPICS]
        if invalid_topics:
            return False, f"unsupported rosbag topics: {', '.join(invalid_topics)}"
        return _start_process(
            "rosbag_recording",
            [
                "ros2",
                "bag",
                "record",
                "-s",
                "mcap",
                "-o",
                str(bag_path),
                *topics,
            ],
            f"started rosbag recording to {bag_path} ({len(topics)} topics)",
            cwd=repo_root,
            log_path=bag_dir / f"{bag_name}.log",
        )


def start_state_converter_stack() -> Tuple[bool, str]:
    with _LOCK:
        _cleanup_stale("state_converter_stack")
        if "state_converter_stack" in _PROCESSES:
            return False, "state converter stack already running"

        started_names: list[str] = []
        try:
            ok, msg = _start_process(
                "state_converter_stack",
                ["ros2", "launch", "go2_state_converter", "state_converter.launch.py"],
                "started go2_state_converter state_converter.launch.py",
            )
            if not ok:
                return False, msg
            started_names.append("state_converter_stack")
        except Exception as exc:
            for name in reversed(started_names):
                proc = _PROCESSES.get(name)
                if proc is None:
                    continue
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=2.0)
                _PROCESSES.pop(name, None)
            return False, f"failed to start state converter stack: {exc}"

        return True, "started state converter stack"


def start_arm_controller() -> Tuple[bool, str]:
    return start_arm_controller_mode("pink_mode_switch")


def start_arm_controller_mode(mode: str) -> Tuple[bool, str]:
    with _LOCK:
        config = _ARM_CONTROLLER_MODES.get(mode)
        if config is None:
            return False, f"unknown arm controller mode: {mode}"

        process_name = str(config["process_name"])
        for other_config in _ARM_CONTROLLER_MODES.values():
            _cleanup_stale(str(other_config["process_name"]))
        active_names = [
            str(other_config["process_name"])
            for other_config in _ARM_CONTROLLER_MODES.values()
            if str(other_config["process_name"]) in _PROCESSES
        ]
        if active_names:
            return False, f"arm controller already running: {', '.join(active_names)}"

        return _start_process(
            process_name,
            config["command"],
            str(config["success_message"]),
        )


def start_coordination_module_mode(mode: str) -> Tuple[bool, str]:
    with _LOCK:
        config = _COORDINATION_MODULE_MODES.get(mode)
        if config is None:
            return False, f"unknown coordination module mode: {mode}"

        process_name = str(config["process_name"])
        for other_config in _COORDINATION_MODULE_MODES.values():
            _cleanup_stale(str(other_config["process_name"]))
        active_names = [
            str(other_config["process_name"])
            for other_config in _COORDINATION_MODULE_MODES.values()
            if str(other_config["process_name"]) in _PROCESSES
        ]
        if active_names:
            return False, f"coordination module already running: {', '.join(active_names)}"

        return _start_process(
            process_name,
            ["ros2", "run", "coordination_module", str(config["executable"])],
            str(config["success_message"]),
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


def stop_state_converter_stack() -> Tuple[bool, str]:
    results: list[str] = []
    any_running = False
    for name in ("state_converter_stack",):
        ok, msg = stop_launch(name)
        if ok:
            any_running = True
            results.append(msg)

    if not any_running:
        return False, "state converter stack not running"
    return True, "; ".join(results)


def stop_arm_controller() -> Tuple[bool, str]:
    results: list[str] = []
    any_running = False
    for config in _ARM_CONTROLLER_MODES.values():
        ok, msg = stop_launch(str(config["process_name"]))
        if ok:
            any_running = True
            results.append(msg)

    if not any_running:
        return False, "arm controller not running"
    return True, "; ".join(results)


def stop_coordination_module() -> Tuple[bool, str]:
    results: list[str] = []
    any_running = False
    for config in _COORDINATION_MODULE_MODES.values():
        ok, msg = stop_launch(str(config["process_name"]))
        if ok:
            any_running = True
            results.append(msg)

    if not any_running:
        return False, "coordination module not running"
    return True, "; ".join(results)


def get_active_arm_controller_mode() -> str | None:
    with _LOCK:
        for mode, config in _ARM_CONTROLLER_MODES.items():
            process_name = str(config["process_name"])
            _cleanup_stale(process_name)
            if process_name in _PROCESSES:
                return mode
    return None


def is_arm_controller_running() -> bool:
    return get_active_arm_controller_mode() is not None


def get_active_coordination_module_mode() -> str | None:
    with _LOCK:
        for mode, config in _COORDINATION_MODULE_MODES.items():
            process_name = str(config["process_name"])
            _cleanup_stale(process_name)
            if process_name in _PROCESSES:
                return mode
    return None


def is_coordination_module_running() -> bool:
    return get_active_coordination_module_mode() is not None


def _normalize_node_name(name: str) -> str:
    return name.strip().lstrip("/")


def _cmdline_matches_node(cmdline: Sequence[str], node_name: str) -> bool:
    if not cmdline:
        return False

    joined = " ".join(cmdline)
    executable = Path(cmdline[0]).name
    return (
        f"__node:={node_name}" in joined
        or f"__node={node_name}" in joined
        or executable == node_name
    )


def stop_ros_nodes(
    node_names: Sequence[str],
    exclude_pids: Sequence[int] | None = None,
) -> Tuple[bool, str]:
    target_names = {
        normalized
        for normalized in (_normalize_node_name(name) for name in node_names)
        if normalized
    }
    if not target_names:
        return True, "no external ROS nodes were visible"

    excluded = {int(pid) for pid in (exclude_pids or ())}
    matched_processes: list[psutil.Process] = []
    matched_nodes: set[str] = set()

    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid in excluded:
            continue

        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue

        matched = [name for name in target_names if _cmdline_matches_node(cmdline, name)]
        if not matched:
            continue

        matched_processes.append(proc)
        matched_nodes.update(matched)

    if not matched_processes:
        return True, "no matching ROS node processes needed to be terminated"

    for proc in matched_processes:
        try:
            proc.terminate()
        except (psutil.Error, OSError):
            pass

    _gone, alive = psutil.wait_procs(matched_processes, timeout=5.0)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            pass
    psutil.wait_procs(alive, timeout=2.0)

    return (
        True,
        "terminated "
        f"{len(matched_processes)} process(es) for node(s): "
        + ", ".join(f"/{name}" for name in sorted(matched_nodes)),
    )


def is_running(name: str) -> bool:
    with _LOCK:
        _cleanup_stale(name)
        return name in _PROCESSES


def stop_all() -> None:
    for name in (
        "control_stack",
        "front_back_estimator",
        "left_right_estimator",
        "up_down_estimator",
        "icon_lab_d1_ros2",
        "foxglove_bridge",
        "rosbag_recording",
        "state_converter_stack",
        "arm_pink_controller_mode_switch",
    ):
        stop_launch(name)
    for config in _COORDINATION_MODULE_MODES.values():
        stop_launch(str(config["process_name"]))

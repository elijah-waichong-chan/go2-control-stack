from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

try:
    import eigenpy  # noqa: F401
    import pinocchio as pin
    import pink
    from pink import solve_ik
    from pink.tasks import FrameTask, PostureTask
    from qpsolvers import available_solvers
except Exception as exc:  # pragma: no cover - handled at runtime
    IMPORT_ERROR = exc
else:  # pragma: no cover - import side effect only
    IMPORT_ERROR = None


class PinkUnavailableError(RuntimeError):
    """Raised when Pink or one of its runtime dependencies is missing."""


@dataclass(frozen=True)
class PinkSolveResult:
    joint_command_deg: np.ndarray
    current_z_m: float


class D1PinkSolver:
    ARM_JOINT_COUNT = 6
    MODEL_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=float)
    SERVO_LIMITS_DEG = np.array(
        [
            [-163.0, 164.0],
            [-88.0, 101.0],
            [-94.0, 92.0],
            [-160.0, 165.0],
            [-95.0, 110.0],
            [-165.0, 165.0],
        ],
        dtype=float,
    )
    POSITION_COST = np.array([1.0, 1.0, 10.0], dtype=float)
    ORIENTATION_COST = np.array([1.0, 1.0, 1.0], dtype=float)
    POSTURE_COST = 1.0e-3
    CURRENT_POSTURE_COST = 0.0
    def __init__(self, z_ref_min_m: float, z_ref_max_m: float) -> None:
        if IMPORT_ERROR is not None:
            raise PinkUnavailableError(
                "Pink runtime dependencies are missing. Install pin-pink, pinocchio,"
                " qpsolvers, and a QP backend such as quadprog or osqp."
            ) from IMPORT_ERROR

        urdf_path = (
            Path(get_package_share_directory("d1_description")) / "urdf" / "d1_description.urdf"
        )
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.controlled_joint_names = tuple(f"Joint{i}" for i in range(1, self.ARM_JOINT_COUNT + 1))
        self.controlled_joint_idx_q = tuple(
            self.model.joints[self.model.getJointId(name)].idx_q for name in self.controlled_joint_names
        )
        self.model_limits_deg = self._compute_model_limits_deg()
        self._apply_model_limits()
        self.ee_frame_name = "Link6"
        self.q0_deg: np.ndarray | None = None
        self.task_reference_q_deg: np.ndarray | None = None
        self.task_reference_configuration = None
        self.nominal_target_transform = None
        self.z_ref_min_m = float(min(z_ref_min_m, z_ref_max_m))
        self.z_ref_max_m = float(max(z_ref_min_m, z_ref_max_m))
        self.solver_name = self._pick_qp_solver()
        self.ee_task = FrameTask(
            self.ee_frame_name,
            position_cost=self.POSITION_COST.tolist(),
            orientation_cost=self.ORIENTATION_COST,
        )
        self.posture_task = PostureTask(cost=self.POSTURE_COST)
        self.current_posture_task = PostureTask(cost=self.CURRENT_POSTURE_COST)

    def _copy_transform(self, transform):
        return pin.SE3(np.array(transform.rotation, copy=True), np.array(transform.translation, copy=True))

    def _pick_qp_solver(self) -> str:
        for candidate in ("quadprog", "osqp", "proxqp", "clarabel", "cvxopt"):
            if candidate in available_solvers:
                return candidate
        if available_solvers:
            return sorted(available_solvers)[0]
        raise PinkUnavailableError("No QP solver backend is available for Pink.")

    def _compute_model_limits_deg(self) -> np.ndarray:
        model_limits = np.zeros_like(self.SERVO_LIMITS_DEG)
        for index, (servo_lower, servo_upper) in enumerate(self.SERVO_LIMITS_DEG):
            mapped = np.array(
                [servo_lower * self.MODEL_SIGN[index], servo_upper * self.MODEL_SIGN[index]],
                dtype=float,
            )
            model_limits[index] = np.array([np.min(mapped), np.max(mapped)], dtype=float)
        return model_limits

    def _apply_model_limits(self) -> None:
        joint_indices = np.array(self.controlled_joint_idx_q, dtype=int)
        self.model.lowerPositionLimit[joint_indices] = np.deg2rad(self.model_limits_deg[:, 0])
        self.model.upperPositionLimit[joint_indices] = np.deg2rad(self.model_limits_deg[:, 1])

    def clamp_model_q_deg(self, q_deg: np.ndarray) -> np.ndarray:
        q_deg = np.asarray(q_deg, dtype=float).reshape(self.ARM_JOINT_COUNT)
        return np.clip(q_deg, self.model_limits_deg[:, 0], self.model_limits_deg[:, 1])

    def servo_to_model(self, servo_angles_deg: np.ndarray) -> np.ndarray:
        model_q_deg = np.asarray(servo_angles_deg[: self.ARM_JOINT_COUNT], dtype=float) * self.MODEL_SIGN
        return self.clamp_model_q_deg(model_q_deg)

    def model_to_servo(self, model_joint_deg: np.ndarray, current_servo_deg: np.ndarray) -> np.ndarray:
        command = np.asarray(current_servo_deg, dtype=float).copy()
        clamped_model_deg = self.clamp_model_q_deg(model_joint_deg)
        command[: self.ARM_JOINT_COUNT] = clamped_model_deg * self.MODEL_SIGN
        return command

    def _initialize_task_targets_from_q0(self) -> None:
        if self.nominal_target_transform is not None:
            return
        if self.q0_deg is None:
            raise ValueError("q0 must be set before initializing task targets")
        self.task_reference_q_deg = self.q0_deg.copy()
        self.task_reference_configuration = self.make_configuration(self.task_reference_q_deg)
        self.nominal_target_transform = self.task_reference_configuration.get_transform_frame_to_world(
            self.ee_frame_name
        )
        self.posture_task.set_target(self.task_reference_configuration.q.copy())
        self.ee_task.set_target(self._copy_transform(self.nominal_target_transform))

    def set_q0(self, q_deg: np.ndarray) -> float:
        self.q0_deg = self.clamp_model_q_deg(q_deg).copy()
        self._initialize_task_targets_from_q0()
        configuration = self.make_configuration(self.q0_deg)
        current_transform = configuration.get_transform_frame_to_world(self.ee_frame_name)
        return float(current_transform.translation[2])

    def get_current_z(self, q_deg: np.ndarray) -> float:
        return float(self.get_current_position_xyz(q_deg)[2])

    def get_current_position_xyz(self, q_deg: np.ndarray) -> np.ndarray:
        configuration = self.make_configuration(q_deg)
        transform = configuration.get_transform_frame_to_world(self.ee_frame_name)
        return np.array(transform.translation, dtype=float, copy=True)

    def get_current_orientation_axes(self, q_deg: np.ndarray) -> np.ndarray:
        configuration = self.make_configuration(q_deg)
        transform = configuration.get_transform_frame_to_world(self.ee_frame_name)
        return np.array(transform.rotation, dtype=float, copy=True)

    def get_target_orientation_axes(self) -> np.ndarray | None:
        if self.nominal_target_transform is None:
            return None
        return np.array(self.nominal_target_transform.rotation, dtype=float, copy=True)

    def make_configuration(self, q_deg: np.ndarray):
        q_deg = self.clamp_model_q_deg(q_deg)
        q_rad = pin.neutral(self.model)
        q_rad[np.array(self.controlled_joint_idx_q, dtype=int)] = np.deg2rad(
            np.asarray(q_deg, dtype=float)
        )
        return pink.Configuration(self.model, self.model.createData(), q_rad)

    def solve_step(
        self,
        current_q_deg: np.ndarray,
        z_reference_m: float,
        dt: float,
    ) -> PinkSolveResult:
        current_q_deg = self.clamp_model_q_deg(current_q_deg)
        if self.q0_deg is None or self.nominal_target_transform is None:
            self.set_q0(current_q_deg)

        configuration = self.make_configuration(current_q_deg)
        current_transform = configuration.get_transform_frame_to_world(self.ee_frame_name)
        target_transform = self._copy_transform(self.nominal_target_transform)
        target_transform.translation[2] = float(
            np.clip(z_reference_m, self.z_ref_min_m, self.z_ref_max_m)
        )
        self.ee_task.set_target(target_transform)
        self.posture_task.set_target(self.task_reference_configuration.q.copy())
        self.current_posture_task.set_target(configuration.q.copy())
        velocity = solve_ik(
            configuration,
            [self.ee_task, self.posture_task, self.current_posture_task],
            dt,
            solver=self.solver_name,
        )
        configuration.integrate_inplace(velocity, dt)
        joint_command_deg = np.rad2deg(
            configuration.q[np.array(self.controlled_joint_idx_q, dtype=int)]
        )
        joint_command_deg = self.clamp_model_q_deg(joint_command_deg)
        return PinkSolveResult(
            joint_command_deg=np.asarray(joint_command_deg, dtype=float),
            current_z_m=float(current_transform.translation[2]),
        )

    def solve_velocity_step(
        self,
        current_q_deg: np.ndarray,
        z_velocity_mps: float,
        dt: float,
    ) -> PinkSolveResult:
        current_q_deg = self.clamp_model_q_deg(current_q_deg)
        if self.q0_deg is None or self.nominal_target_transform is None:
            self.set_q0(current_q_deg)

        configuration = self.make_configuration(current_q_deg)
        current_transform = configuration.get_transform_frame_to_world(self.ee_frame_name)
        target_transform = self._copy_transform(current_transform)
        target_transform.translation[2] = float(
            np.clip(
                current_transform.translation[2] + z_velocity_mps * dt,
                self.z_ref_min_m,
                self.z_ref_max_m,
            )
        )
        self.ee_task.set_target(target_transform)
        self.posture_task.set_target(self.task_reference_configuration.q.copy())
        self.current_posture_task.set_target(configuration.q.copy())
        velocity = solve_ik(
            configuration,
            [self.ee_task, self.posture_task, self.current_posture_task],
            dt,
            solver=self.solver_name,
        )
        configuration.integrate_inplace(velocity, dt)
        joint_command_deg = np.rad2deg(
            configuration.q[np.array(self.controlled_joint_idx_q, dtype=int)]
        )
        joint_command_deg = self.clamp_model_q_deg(joint_command_deg)
        return PinkSolveResult(
            joint_command_deg=np.asarray(joint_command_deg, dtype=float),
            current_z_m=float(current_transform.translation[2]),
        )

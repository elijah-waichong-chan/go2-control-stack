#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_D1_ARM_JOINT_NAMES = (
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
)
DEFAULT_END_EFFECTOR_FRAME = "Empty_Link6"
DEFAULT_D1_URDF_NAME = "d1_550_description.urdf"
_D1_PACKAGE_NAME = "d1_550_description"
_D1_URDF_ENV_VAR = "D1_ARM_URDF_PATH"


class IKSolveError(RuntimeError):
    """Raised when the solver is configured to fail on non-converged IK."""


def _require_pinocchio():
    try:
        import pinocchio as pin  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Pinocchio is required for D1 IK. "
            "Install the Python package first."
        ) from exc
    return pin


def _resolve_source_tree_urdf_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(
            parent / "src" / _D1_PACKAGE_NAME / "urdf" / DEFAULT_D1_URDF_NAME
        )
        candidates.append(
            parent / _D1_PACKAGE_NAME / "urdf" / DEFAULT_D1_URDF_NAME
        )
    return candidates


def resolve_d1_urdf_path(urdf_path: Optional[str | Path] = None) -> Path:
    """Resolve the D1 URDF from path, ROS install, env var, or source tree."""

    candidates: list[Path] = []
    if urdf_path is not None:
        candidates.append(Path(urdf_path).expanduser())

    env_urdf = os.environ.get(_D1_URDF_ENV_VAR)
    if env_urdf:
        candidates.append(Path(env_urdf).expanduser())

    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory(_D1_PACKAGE_NAME))
            / "urdf"
            / DEFAULT_D1_URDF_NAME
        )
    except Exception:
        pass

    candidates.extend(_resolve_source_tree_urdf_candidates())

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    searched = "\n".join(str(path) for path in seen)
    raise FileNotFoundError(
        f"Unable to locate {DEFAULT_D1_URDF_NAME}. Checked:\n{searched}"
    )


def _to_xyz(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3D position, got shape {arr.shape}")
    return arr


def _to_joint_vector(values: Sequence[float], expected_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (expected_len,):
        raise ValueError(
            f"Expected {expected_len} joint values, got shape {arr.shape}"
        )
    return arr


class D1ArmIKSolver:
    """Position-only damped least-squares IK for the D1 arm."""

    def __init__(
        self,
        urdf_path: Optional[str | Path] = None,
        arm_joint_names: Sequence[str] = DEFAULT_D1_ARM_JOINT_NAMES,
        end_effector_frame: str = DEFAULT_END_EFFECTOR_FRAME,
    ) -> None:
        self._pin = _require_pinocchio()
        self.urdf_path = resolve_d1_urdf_path(urdf_path)
        self.model = self._pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        self.arm_joint_names = tuple(arm_joint_names)
        if not self.arm_joint_names:
            raise ValueError("arm_joint_names must not be empty")

        model_joint_names = set(self.model.names)
        missing_joints = [
            name
            for name in self.arm_joint_names
            if name not in model_joint_names
        ]
        if missing_joints:
            raise ValueError(
                f"Joint names not found in URDF: {missing_joints}"
            )

        model_frame_names = {frame.name for frame in self.model.frames}
        if end_effector_frame not in model_frame_names:
            raise ValueError(f"Frame '{end_effector_frame}' not found in URDF")

        self.end_effector_frame = end_effector_frame
        self._frame_id = self.model.getFrameId(end_effector_frame)
        self._joint_ids = [
            self.model.getJointId(name) for name in self.arm_joint_names
        ]
        self._arm_q_indices = [
            self.model.joints[joint_id].idx_q for joint_id in self._joint_ids
        ]
        self._arm_v_indices = [
            self.model.joints[joint_id].idx_v for joint_id in self._joint_ids
        ]
        self._arm_lower_limits = np.array(
            [
                self.model.lowerPositionLimit[q_index]
                for q_index in self._arm_q_indices
            ],
            dtype=float,
        )
        self._arm_upper_limits = np.array(
            [
                self.model.upperPositionLimit[q_index]
                for q_index in self._arm_q_indices
            ],
            dtype=float,
        )
        self._reference_q = self._pin.neutral(self.model)

        for joint_name, joint_id in zip(self.arm_joint_names, self._joint_ids):
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Joint '{joint_name}' is not a 1-DoF joint and is not "
                    "supported by this solver"
                )

    @property
    def dof(self) -> int:
        return len(self.arm_joint_names)

    def neutral_arm_configuration(self) -> list[float]:
        return self._extract_arm_configuration(self._reference_q).tolist()

    def clamp_to_limits(self, joint_angles: Sequence[float]) -> np.ndarray:
        return np.clip(
            _to_joint_vector(joint_angles, self.dof),
            self._arm_lower_limits,
            self._arm_upper_limits,
        )

    def forward_position(self, joint_angles: Sequence[float]) -> np.ndarray:
        q = self._full_configuration(joint_angles)
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)
        return np.array(
            self.data.oMf[self._frame_id].translation,
            dtype=float,
        ).copy()

    def solve_position_ik(
        self,
        target_pos: Sequence[float],
        start_q: Optional[Sequence[float]] = None,
        *,
        step_size: float = 1.0,
        damping: float = 0.05,
        tol: float = 0.005,
        max_iters: int = 500,
        max_attempts: int = 3,
        max_step_norm: float = 0.05,
        restart_scale: float = 0.6,
        random_seed: Optional[int] = None,
        raise_on_failure: bool = False,
    ) -> list[float]:
        target = _to_xyz(target_pos)
        desired_q = (
            self.clamp_to_limits(start_q)
            if start_q is not None
            else self._extract_arm_configuration(self._reference_q)
        )

        best_q = desired_q.copy()
        best_error_norm = float("inf")
        rng = np.random.default_rng(random_seed)

        for attempt in range(max_attempts):
            for _ in range(max_iters):
                q_full = self._full_configuration(desired_q)
                current_pos = self._forward_position_from_full_q(q_full)
                error = target - current_pos
                error_norm = float(np.linalg.norm(error))

                if error_norm < best_error_norm:
                    best_error_norm = error_norm
                    best_q = desired_q.copy()

                if error_norm < tol:
                    return desired_q.tolist()

                if max_step_norm > 0.0 and error_norm > max_step_norm:
                    error = (error / error_norm) * max_step_norm

                jacobian = self._translation_jacobian(q_full)
                lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(3)
                delta_q = jacobian.T @ np.linalg.solve(lhs, error)
                desired_q = self.clamp_to_limits(
                    desired_q + step_size * delta_q
                )

            if attempt + 1 < max_attempts:
                restart = rng.uniform(-0.5, 0.5, size=self.dof) * restart_scale
                desired_q = self.clamp_to_limits(best_q + restart)

        if raise_on_failure:
            raise IKSolveError(
                "IK did not converge within "
                f"{max_attempts * max_iters} iterations; "
                f"best position error was {best_error_norm:.6f} m"
            )
        return best_q.tolist()

    def _extract_arm_configuration(self, q_full: np.ndarray) -> np.ndarray:
        return np.array(
            [q_full[q_index] for q_index in self._arm_q_indices],
            dtype=float,
        )

    def _full_configuration(self, joint_angles: Sequence[float]) -> np.ndarray:
        arm_q = _to_joint_vector(joint_angles, self.dof)
        q_full = self._reference_q.copy()
        for q_index, angle in zip(self._arm_q_indices, arm_q):
            q_full[q_index] = angle
        return q_full

    def _forward_position_from_full_q(self, q_full: np.ndarray) -> np.ndarray:
        self._pin.forwardKinematics(self.model, self.data, q_full)
        self._pin.updateFramePlacements(self.model, self.data)
        return np.array(
            self.data.oMf[self._frame_id].translation,
            dtype=float,
        ).copy()

    def _translation_jacobian(self, q_full: np.ndarray) -> np.ndarray:
        try:
            self._pin.computeJointJacobians(self.model, self.data, q_full)
            self._pin.updateFramePlacements(self.model, self.data)
            jacobian = self._pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self._frame_id,
                self._local_world_aligned_reference(),
            )
            return np.asarray(jacobian[:3, self._arm_v_indices], dtype=float)
        except Exception:
            return self._numeric_translation_jacobian(q_full)

    def _numeric_translation_jacobian(
        self,
        q_full: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        jacobian = np.zeros((3, self.dof), dtype=float)
        for column, q_index in enumerate(self._arm_q_indices):
            q_plus = q_full.copy()
            q_minus = q_full.copy()
            q_plus[q_index] += eps
            q_minus[q_index] -= eps
            pos_plus = self._forward_position_from_full_q(q_plus)
            pos_minus = self._forward_position_from_full_q(q_minus)
            jacobian[:, column] = (pos_plus - pos_minus) / (2.0 * eps)
        return jacobian

    def _local_world_aligned_reference(self):
        reference_frame = getattr(self._pin, "ReferenceFrame", None)
        if reference_frame is not None and hasattr(
            reference_frame, "LOCAL_WORLD_ALIGNED"
        ):
            return reference_frame.LOCAL_WORLD_ALIGNED
        return self._pin.LOCAL_WORLD_ALIGNED


def solve_d1_position_ik(
    target_pos: Sequence[float],
    start_q: Optional[Sequence[float]] = None,
    *,
    urdf_path: Optional[str | Path] = None,
    arm_joint_names: Sequence[str] = DEFAULT_D1_ARM_JOINT_NAMES,
    end_effector_frame: str = DEFAULT_END_EFFECTOR_FRAME,
    step_size: float = 1.0,
    damping: float = 0.05,
    tol: float = 0.005,
    max_iters: int = 500,
    max_attempts: int = 3,
    max_step_norm: float = 0.05,
    restart_scale: float = 0.6,
    random_seed: Optional[int] = None,
    raise_on_failure: bool = False,
) -> list[float]:
    solver = D1ArmIKSolver(
        urdf_path=urdf_path,
        arm_joint_names=arm_joint_names,
        end_effector_frame=end_effector_frame,
    )
    return solver.solve_position_ik(
        target_pos,
        start_q=start_q,
        step_size=step_size,
        damping=damping,
        tol=tol,
        max_iters=max_iters,
        max_attempts=max_attempts,
        max_step_norm=max_step_norm,
        restart_scale=restart_scale,
        random_seed=random_seed,
        raise_on_failure=raise_on_failure,
    )

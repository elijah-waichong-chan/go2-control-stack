"""Drake-based inverse-kinematics helpers for the D1 arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from ament_index_python.packages import get_package_share_directory
from hq_pcot_msgs.msg import ArmTarget
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve

JointVector = np.ndarray


@dataclass(frozen=True)
class _ModeSettings:
    locked_joint_indices: tuple[int, ...]
    pose_position_cost_diag: tuple[float, float, float]
    fallback_tool_axis_cost: float
    orientation_cost: float
    posture_cost: float


class D1DrakeIKSolver:
    """Full inverse kinematics for the 6-DOF D1 arm using Drake."""

    DESCRIPTION_PACKAGE = "d1_description"
    URDF_RELATIVE_PATH = Path("urdf/d1_description.urdf")

    BASE_FRAME = "base_link"
    END_EFFECTOR_FRAME = "Link6"
    JOINT_NAMES = (
        "Joint1",
        "Joint2",
        "Joint3",
        "Joint4",
        "Joint5",
        "Joint6",
    )
    _ZERO_POSITION = np.zeros(3, dtype=float)
    _ZERO_QUATERNION = np.zeros(4, dtype=float)
    _TOOL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)

    FRONT_BACK_SETTINGS = _ModeSettings(
        locked_joint_indices=(0,),
        pose_position_cost_diag=(1.0, 1.0, 5.0),
        fallback_tool_axis_cost=50.0,
        orientation_cost=20.0,
        posture_cost=5e-4,
    )
    UP_DOWN_SETTINGS = _ModeSettings(
        locked_joint_indices=(0, 3),
        pose_position_cost_diag=(1.0, 1.0, 10.0),
        fallback_tool_axis_cost=3.0,
        orientation_cost=10.0,
        posture_cost=5e-4,
    )

    def __init__(self) -> None:
        package_share = Path(get_package_share_directory(self.DESCRIPTION_PACKAGE))
        self.urdf_path = package_share / self.URDF_RELATIVE_PATH

        self.plant = MultibodyPlant(time_step=0.0)
        Parser(self.plant).AddModels(str(self.urdf_path))
        self.plant.Finalize()

        self.world_frame = self.plant.world_frame()
        self.end_effector_frame = self.plant.GetFrameByName(self.END_EFFECTOR_FRAME)
        self.joint_names = self.JOINT_NAMES
        self.controlled_joint_objects = tuple(
            self.plant.GetJointByName(joint_name) for joint_name in self.joint_names
        )
        self.controlled_joint_idx_q = tuple(
            joint.position_start() for joint in self.controlled_joint_objects
        )

        lower_limits = self.plant.GetPositionLowerLimits()
        upper_limits = self.plant.GetPositionUpperLimits()
        controlled_idx_q = np.array(self.controlled_joint_idx_q, dtype=int)
        self.lower_limits = np.rad2deg(np.asarray(lower_limits[controlled_idx_q], dtype=float))
        self.upper_limits = np.rad2deg(np.asarray(upper_limits[controlled_idx_q], dtype=float))
        self.last_solve_time_ms = 0.0

    def solve_front_back(
        self,
        current_q_deg: JointVector,
        target: ArmTarget,
        q0_deg: JointVector,
    ) -> JointVector:
        """Solve a front-back command with Joint1 held fixed."""
        return self._solve_mode(current_q_deg, target, q0_deg, self.FRONT_BACK_SETTINGS)

    def solve_up_down(
        self,
        current_q_deg: JointVector,
        target: ArmTarget,
        q0_deg: JointVector,
    ) -> JointVector:
        """Solve an up-down command with Joint1 and Joint4 held fixed."""
        return self._solve_mode(current_q_deg, target, q0_deg, self.UP_DOWN_SETTINGS)

    def get_model_q(self, q_deg: JointVector) -> JointVector:
        q_deg = self._validate_joint_vector(q_deg)
        model_q = self.plant.GetDefaultPositions().copy()
        model_q[np.array(self.controlled_joint_idx_q, dtype=int)] = np.deg2rad(q_deg)
        return model_q

    def get_end_effector_pose(self, q_deg: JointVector):
        context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context, self.get_model_q(q_deg))
        return self.plant.CalcRelativeTransform(
            context,
            self.world_frame,
            self.end_effector_frame,
        )

    def _solve_mode(
        self,
        current_q_deg: JointVector,
        target: ArmTarget,
        q0_deg: JointVector,
        settings: _ModeSettings,
    ) -> JointVector:
        current_q_deg = self._validate_joint_vector(current_q_deg)
        q0_deg = self._validate_joint_vector(q0_deg)
        current_model_q = self.get_model_q(current_q_deg)
        nominal_rotation = np.array(
            self.get_end_effector_pose(q0_deg).rotation().matrix(),
            dtype=float,
        )
        ik = InverseKinematics(self.plant, with_joint_limits=True)
        decision_q = ik.q()
        prog = ik.get_mutable_prog()

        self._add_locked_joint_constraints(
            prog,
            decision_q,
            current_model_q,
            settings.locked_joint_indices,
        )

        command_type = int(target.command_type)
        if command_type not in (
            ArmTarget.COMMAND_TYPE_END_EFFECTOR_POSE,
            ArmTarget.COMMAND_TYPE_JOINT_CONFIGURATION,
        ):
            raise ValueError(
                f"Unsupported ArmTarget command_type={command_type}. "
                "Expected end-effector pose or joint configuration."
            )

        if command_type == ArmTarget.COMMAND_TYPE_END_EFFECTOR_POSE:
            self._add_pose_target_costs(ik, target, nominal_rotation, settings)
            self._add_posture_cost(prog, decision_q, current_model_q, settings.posture_cost)
        else:
            self._add_joint_target_costs(
                prog,
                decision_q,
                current_q_deg,
                target,
                settings.locked_joint_indices,
            )

        prog.SetInitialGuess(decision_q, current_model_q)

        start_time = time.perf_counter()
        result = Solve(prog)
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        if not result.is_success():
            raise RuntimeError(
                "Drake IK failed: "
                f"solver={result.get_solver_id().name()} "
                f"solution_result={result.get_solution_result()}"
            )

        solved_model_q = np.asarray(result.GetSolution(decision_q), dtype=float).reshape(-1)
        return np.rad2deg(
            solved_model_q[np.array(self.controlled_joint_idx_q, dtype=int)]
        )

    def _add_pose_target_costs(
        self,
        ik: InverseKinematics,
        target: ArmTarget,
        nominal_rotation: np.ndarray,
        settings: _ModeSettings,
    ) -> None:
        frame_id = target.header.frame_id.strip()
        if frame_id and frame_id != self.BASE_FRAME:
            raise ValueError(
                f"Unsupported ArmTarget frame_id='{frame_id}'. "
                f"Expected '' or '{self.BASE_FRAME}'."
            )

        target_position = np.asarray(target.position_m, dtype=float).reshape(3)
        ik.AddPositionCost(
            self.world_frame,
            target_position,
            self.end_effector_frame,
            self._ZERO_POSITION,
            np.diag(np.asarray(settings.pose_position_cost_diag, dtype=float)),
        )

        orientation_xyzw = np.asarray(target.orientation_xyzw, dtype=float).reshape(4)
        if not np.allclose(orientation_xyzw, self._ZERO_QUATERNION):
            ik.AddOrientationCost(
                self.world_frame,
                RotationMatrix(self._quat_xyzw_to_matrix(orientation_xyzw)),
                self.end_effector_frame,
                RotationMatrix(),
                settings.orientation_cost,
            )
            return

        ik.AddAngleBetweenVectorsCost(
            self.world_frame,
            nominal_rotation[:, 2],
            self.end_effector_frame,
            self._TOOL_AXIS,
            settings.fallback_tool_axis_cost,
        )

    def _add_joint_target_costs(
        self,
        prog,
        decision_q,
        current_q_deg: JointVector,
        target: ArmTarget,
        locked_joint_indices: tuple[int, ...],
    ) -> None:
        target_q_deg = self._validate_joint_vector(target.joint_configuration_deg)
        blended_q_deg = current_q_deg.copy()
        for joint_index in range(len(self.joint_names)):
            if joint_index in locked_joint_indices:
                continue
            blended_q_deg[joint_index] = target_q_deg[joint_index]

        weight = np.zeros(self.plant.num_positions(), dtype=float)
        for joint_index in range(len(self.joint_names)):
            if joint_index in locked_joint_indices:
                continue
            q_index = self.controlled_joint_idx_q[joint_index]
            weight[q_index] = 1.0

        prog.AddQuadraticErrorCost(
            np.diag(weight),
            self.get_model_q(blended_q_deg),
            decision_q,
        )

    def _add_posture_cost(
        self,
        prog,
        decision_q,
        reference_model_q: np.ndarray,
        posture_cost: float,
    ) -> None:
        if posture_cost <= 0.0:
            return
        prog.AddQuadraticErrorCost(
            posture_cost * np.eye(self.plant.num_positions()),
            reference_model_q,
            decision_q,
        )

    def _add_locked_joint_constraints(
        self,
        prog,
        decision_q,
        current_model_q: np.ndarray,
        locked_joint_indices: tuple[int, ...],
    ) -> None:
        for joint_index in locked_joint_indices:
            q_index = self.controlled_joint_idx_q[joint_index]
            prog.AddBoundingBoxConstraint(
                current_model_q[q_index],
                current_model_q[q_index],
                np.array([decision_q[q_index]]),
            )

    def _quat_xyzw_to_matrix(self, orientation_xyzw) -> np.ndarray:
        quat_xyzw = np.asarray(orientation_xyzw, dtype=float).reshape(4)
        norm = np.linalg.norm(quat_xyzw)
        if norm <= 1e-9:
            raise ValueError("ArmTarget orientation quaternion must be non-zero.")
        x, y, z, w = quat_xyzw / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _validate_joint_vector(self, q_deg: JointVector) -> JointVector:
        q_deg = np.asarray(q_deg, dtype=float).reshape(-1)
        if q_deg.shape != (len(self.joint_names),):
            raise ValueError(
                f"Expected {len(self.joint_names)} arm joints, got shape {q_deg.shape}."
            )
        return q_deg

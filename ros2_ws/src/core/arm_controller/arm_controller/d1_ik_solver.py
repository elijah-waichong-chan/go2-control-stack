"""Inverse-kinematics helpers for the D1 arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import casadi as ca
import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

JointVector = np.ndarray
q0 = np.array([-90.0, 80.0, 10.0, 0.0, -90.0, 90.0], dtype=float)


@dataclass(frozen=True)
class _ModeSettings:
    locked_joint_indices: tuple[int, ...]
    tool_axis_cost: float
    x_cost: float
    y_cost: float
    z_cost: float
    posture_cost: float


class D1IKSolver:
    """Pinocchio/CasADi IK scaffold for the 6-DOF D1 arm."""

    FREE_ORIENTATION_JOINT_INDEX = 5
    JOINT_ORIGINS_XYZ = (
        (0.0, 0.0, 0.0533),
        (0.0, 0.028, 0.0563),
        (0.0, 0.2693, 0.0009),
        (0.0577, 0.042, -0.0275),
        (-0.0001, -0.0237, 0.14018),
        (0.0825, -0.0010782, -0.023822),
    )
    JOINT_ORIGINS_RPY = (
        (0.0, 0.0, -3.1416),
        (1.5708, 0.0, -3.1416),
        (0.0, 0.0, 0.0),
        (-1.5708, 0.0, -1.5708),
        (1.5708, -1.5708, 0.0),
        (-1.5708, 0.0, -1.5708),
    )
    JOINT_AXIS_Z_SIGN = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)

    FRONT_BACK_SETTINGS = _ModeSettings(
        locked_joint_indices=(0,),
        tool_axis_cost=50.0,
        x_cost=0.0,
        y_cost=1.0,
        z_cost=5.0,
        posture_cost=5e-4,
    )
    UP_DOWN_SETTINGS = _ModeSettings(
        locked_joint_indices=(0, 3),
        tool_axis_cost=3.0,
        x_cost=1.0,
        y_cost=1.0,
        z_cost=10.0,
        posture_cost=0.0,
    )

    def __init__(self) -> None:
        package_share = Path(get_package_share_directory("d1_description"))
        self.urdf_path = package_share / Path("urdf/d1_description.urdf")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.neutral_q = pin.neutral(self.model)

        self.base_frame = "base_link"
        self.end_effector_frame = "Link6"
        self.joint_names = (
            "Joint1",
            "Joint2",
            "Joint3",
            "Joint4",
            "Joint5",
            "Joint6",
        )
        self.base_joint_name = "Joint1"
        self.joint_name_to_index = {
            name: index for index, name in enumerate(self.joint_names)
        }

        self.end_effector_frame_id = self.model.getFrameId(self.end_effector_frame)
        self.controlled_joint_ids = tuple(
            self.model.getJointId(joint_name) for joint_name in self.joint_names
        )
        self.controlled_joint_idx_q = tuple(
            self.model.joints[joint_id].idx_q for joint_id in self.controlled_joint_ids
        )

        self.lower_limits = np.rad2deg(
            np.array(
                [
                    self.model.lowerPositionLimit[idx_q]
                    for idx_q in self.controlled_joint_idx_q
                ],
                dtype=float,
            )
        )
        self.upper_limits = np.rad2deg(
            np.array(
                [
                    self.model.upperPositionLimit[idx_q]
                    for idx_q in self.controlled_joint_idx_q
                ],
                dtype=float,
            )
        )
        self.last_solve_time_ms = 0.0
        self.q0_deg = q0.copy()
        self.set_q0(self.q0_deg)

    def set_q0(self, q: JointVector) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        self.q0_deg = q.copy()
        self.q0_end_effector_x = self.get_end_effector_x(self.q0_deg)
        self.q0_end_effector_y = self.get_end_effector_y(self.q0_deg)
        self.q0_end_effector_z = self.get_end_effector_z(self.q0_deg)
        self.q0_end_effector_rotation = self.get_end_effector_rotation(self.q0_deg)
        self.q0_joint6_minus_joint5 = (
            self.get_joint_position("Joint6", self.q0_deg)
            - self.get_joint_position("Joint5", self.q0_deg)
        )

        # Backward-compatible aliases for older IPOPT callers.
        self.nominal_q = self.q0_deg.copy()
        self.nominal_end_effector_x = self.q0_end_effector_x
        self.nominal_end_effector_y = self.q0_end_effector_y
        self.nominal_end_effector_z = self.q0_end_effector_z
        self.nominal_end_effector_rotation = self.q0_end_effector_rotation.copy()
        self.nominal_joint6_minus_joint5 = self.q0_joint6_minus_joint5.copy()

    def save_current_configuration_as_q0(self, q: JointVector) -> None:
        """Replace the stored q0 configuration with the provided arm state."""
        self.set_q0(q)

    # Backward-compatible alias for existing IPOPT callers.
    def set_nominal_q(self, q: JointVector) -> None:
        self.set_q0(q)

    # Backward-compatible alias for existing IPOPT callers.
    def save_current_configuration_as_nominal(self, q: JointVector) -> None:
        self.save_current_configuration_as_q0(q)

    def get_model_q(self, q: JointVector) -> JointVector:
        q = np.asarray(q, dtype=float)
        model_q = pin.neutral(self.model)
        model_q[np.array(self.controlled_joint_idx_q, dtype=int)] = np.deg2rad(q)
        return model_q

    def get_end_effector_y(self, q: JointVector) -> float:
        model_q = self.get_model_q(q)
        pin.forwardKinematics(self.model, self.data, model_q)
        pin.updateFramePlacements(self.model, self.data)
        return float(self.data.oMf[self.end_effector_frame_id].translation[1])

    def get_end_effector_x(self, q: JointVector) -> float:
        model_q = self.get_model_q(q)
        pin.forwardKinematics(self.model, self.data, model_q)
        pin.updateFramePlacements(self.model, self.data)
        return float(self.data.oMf[self.end_effector_frame_id].translation[0])

    def get_end_effector_z(self, q: JointVector) -> float:
        model_q = self.get_model_q(q)
        pin.forwardKinematics(self.model, self.data, model_q)
        pin.updateFramePlacements(self.model, self.data)
        return float(self.data.oMf[self.end_effector_frame_id].translation[2])

    def get_end_effector_rotation(self, q: JointVector) -> np.ndarray:
        model_q = self.get_model_q(q)
        pin.forwardKinematics(self.model, self.data, model_q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.end_effector_frame_id].rotation, dtype=float)

    def get_joint_position(self, joint_name: str, q: JointVector) -> JointVector:
        model_q = self.get_model_q(q)
        pin.forwardKinematics(self.model, self.data, model_q)
        joint_id = self.model.getJointId(joint_name)
        return np.array(self.data.oMi[joint_id].translation, dtype=float)

    def get_casadi_end_effector_transform(self, q: ca.MX) -> tuple[ca.MX, ca.MX]:
        q_rad = q * (np.pi / 180.0)
        rotation = ca.DM.eye(3)
        translation = ca.DM.zeros(3, 1)

        for i, xyz in enumerate(self.JOINT_ORIGINS_XYZ):
            joint_translation = ca.DM(xyz).reshape((3, 1))
            joint_rotation = self._casadi_rpy_matrix(*self.JOINT_ORIGINS_RPY[i])
            axis_rotation = self._casadi_rot_z(self.JOINT_AXIS_Z_SIGN[i] * q_rad[i])
            translation = translation + rotation @ joint_translation
            rotation = rotation @ joint_rotation @ axis_rotation

        return rotation, translation

    def get_casadi_joint_positions(self, q: ca.MX) -> tuple[ca.MX, ...]:
        q_rad = q * (np.pi / 180.0)
        rotation = ca.DM.eye(3)
        translation = ca.DM.zeros(3, 1)
        joint_positions = []

        for i, xyz in enumerate(self.JOINT_ORIGINS_XYZ):
            joint_translation = ca.DM(xyz).reshape((3, 1))
            joint_rotation = self._casadi_rpy_matrix(*self.JOINT_ORIGINS_RPY[i])
            axis_rotation = self._casadi_rot_z(self.JOINT_AXIS_Z_SIGN[i] * q_rad[i])
            translation = translation + rotation @ joint_translation
            joint_positions.append(translation)
            rotation = rotation @ joint_rotation @ axis_rotation

        return tuple(joint_positions)

    def get_casadi_end_effector_y(self, q: ca.MX) -> ca.MX:
        _, translation = self.get_casadi_end_effector_transform(q)
        return translation[1]

    def get_casadi_end_effector_x(self, q: ca.MX) -> ca.MX:
        _, translation = self.get_casadi_end_effector_transform(q)
        return translation[0]

    def get_casadi_end_effector_z(self, q: ca.MX) -> ca.MX:
        _, translation = self.get_casadi_end_effector_transform(q)
        return translation[2]

    def get_casadi_end_effector_rotation(self, q: ca.MX) -> ca.MX:
        rotation, _ = self.get_casadi_end_effector_transform(q)
        return rotation

    def get_casadi_joint6_minus_joint5(self, q: ca.MX) -> ca.MX:
        joint_positions = self.get_casadi_joint_positions(q)
        return joint_positions[5] - joint_positions[4]

    def _casadi_rot_x(self, angle: ca.MX) -> ca.MX:
        c = ca.cos(angle)
        s = ca.sin(angle)
        return ca.vertcat(
            ca.hcat([1, 0, 0]),
            ca.hcat([0, c, -s]),
            ca.hcat([0, s, c]),
        )

    def _casadi_rot_y(self, angle: ca.MX) -> ca.MX:
        c = ca.cos(angle)
        s = ca.sin(angle)
        return ca.vertcat(
            ca.hcat([c, 0, s]),
            ca.hcat([0, 1, 0]),
            ca.hcat([-s, 0, c]),
        )

    def _casadi_rot_z(self, angle: ca.MX) -> ca.MX:
        c = ca.cos(angle)
        s = ca.sin(angle)
        return ca.vertcat(
            ca.hcat([c, -s, 0]),
            ca.hcat([s, c, 0]),
            ca.hcat([0, 0, 1]),
        )

    def _casadi_rpy_matrix(self, roll: float, pitch: float, yaw: float) -> ca.MX:
        return self._casadi_rot_z(yaw) @ self._casadi_rot_y(pitch) @ self._casadi_rot_x(roll)

    def solve_front_back(
        self,
        current_q_deg: JointVector,
        q0_deg: JointVector,
    ) -> JointVector:
        return self._solve_mode(
            current_q_deg,
            q0_deg,
            self.FRONT_BACK_SETTINGS,
            x_reference=None,
            y_reference=float(self.get_end_effector_y(q0_deg)),
            z_reference=float(self.get_end_effector_z(q0_deg)),
        )

    def solve_up_down(
        self,
        current_q_deg: JointVector,
        z_reference: float,
        q0_deg: JointVector,
        *,
        x_reference: float | None = None,
        y_reference: float | None = None,
    ) -> JointVector:
        q0_deg = np.asarray(q0_deg, dtype=float).reshape(-1)
        return self._solve_mode(
            current_q_deg,
            q0_deg,
            self.UP_DOWN_SETTINGS,
            x_reference=float(self.get_end_effector_x(q0_deg)) if x_reference is None else x_reference,
            y_reference=float(self.get_end_effector_y(q0_deg)) if y_reference is None else y_reference,
            z_reference=float(z_reference),
        )

    def _solve_mode(
        self,
        current_q_deg: JointVector,
        q0_deg: JointVector,
        settings: _ModeSettings,
        *,
        x_reference: float | None,
        y_reference: float | None,
        z_reference: float | None,
    ) -> JointVector:
        current_q_deg = np.asarray(current_q_deg, dtype=float).reshape(-1)
        q0_deg = np.asarray(q0_deg, dtype=float).reshape(-1)

        opti = ca.Opti()
        q_var = opti.variable(len(self.joint_names))
        q_target = ca.DM(current_q_deg)
        q0_target = ca.DM(q0_deg)
        q0_rotation = ca.DM(self.get_end_effector_rotation(q0_deg))

        tool_axis_error = self.get_casadi_end_effector_rotation(q_var)[:, 2] - q0_rotation[:, 2]

        objective = 0
        if settings.tool_axis_cost > 0.0:
            objective += settings.tool_axis_cost * ca.sumsqr(tool_axis_error)
        if x_reference is not None and settings.x_cost > 0.0:
            x_error = self.get_casadi_end_effector_x(q_var) - float(x_reference)
            objective += settings.x_cost * ca.sumsqr(x_error)
        if y_reference is not None and settings.y_cost > 0.0:
            y_error = self.get_casadi_end_effector_y(q_var) - float(y_reference)
            objective += settings.y_cost * ca.sumsqr(y_error)
        if z_reference is not None and settings.z_cost > 0.0:
            z_error = self.get_casadi_end_effector_z(q_var) - float(z_reference)
            objective += settings.z_cost * ca.sumsqr(z_error)
        if settings.posture_cost > 0.0:
            q_input_error = q_var - q_target
            objective += settings.posture_cost * ca.sumsqr(q_input_error)

        opti.minimize(objective)

        opti.subject_to(opti.bounded(self.lower_limits, q_var, self.upper_limits))
        for joint_index in settings.locked_joint_indices:
            opti.subject_to(q_var[joint_index] == q0_target[joint_index])
        opti.set_initial(q_var, np.clip(current_q_deg, self.lower_limits, self.upper_limits))
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"print_level": 0, "sb": "yes"},
        )

        start_time = time.perf_counter()
        solution = opti.solve()
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        return np.asarray(solution.value(q_var), dtype=float).reshape(-1)

    # Backward-compatible wrapper for existing IPOPT callers.
    def solve_with_fixed_joint1(self, q: JointVector) -> JointVector:
        return self.solve_front_back(q, self.q0_deg)

    # Backward-compatible wrapper for existing IPOPT callers.
    def solve_with_z_reference(
        self,
        q: JointVector,
        z_reference: float,
        *,
        x_reference: float | None = None,
        y_reference: float | None = None,
    ) -> JointVector:
        return self.solve_up_down(
            q,
            z_reference,
            self.q0_deg,
            x_reference=x_reference,
            y_reference=y_reference,
        )


if __name__ == "__main__":
    solver = D1IKSolver()
    q_test = q0.copy()
    q_test[0] = -40.0

    print("=" * 48)
    print("Input")
    print(f"  q_test_deg: {np.array2string(q_test, precision=1)}")
    print(f"  q0_y:       {solver.q0_end_effector_y:.3f}")
    print(f"  q0_z:       {solver.q0_end_effector_z:.3f}")
    print("-" * 48)

    q_solution = solver.solve_front_back(q_test, solver.q0_deg)
    solution_y = solver.get_end_effector_y(q_solution)
    solution_z = solver.get_end_effector_z(q_solution)

    print("Solve")
    print(f"  solve_time_ms: {solver.last_solve_time_ms:.3f}")
    print("-" * 48)

    print("Result")
    print(f"  q_solution_deg: {np.array2string(q_solution, precision=1)}")
    print(
        f"  y: q0={solver.q0_end_effector_y:.3f} "
        f"solution={solution_y:.3f} "
        f"delta={solution_y - solver.q0_end_effector_y:.3f}"
    )
    print(
        f"  z: q0={solver.q0_end_effector_z:.3f} "
        f"solution={solution_z:.3f} "
        f"delta={solution_z - solver.q0_end_effector_z:.3f}"
    )
    print("=" * 48)

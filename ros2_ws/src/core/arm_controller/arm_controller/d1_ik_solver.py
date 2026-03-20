"""Inverse-kinematics helpers for the D1 arm."""

from __future__ import annotations

from pathlib import Path
import time

import casadi as ca
import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

JointVector = np.ndarray
q0 = np.array([-90.0, 80.0, 10.0, 0.0, -90.0, 90.0], dtype=float)


class D1IKSolver:
    """Pinocchio/CasADi IK scaffold for the 6-DOF D1 arm."""

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

    BASE_JOINT_NAME = JOINT_NAMES[0]
    JOINT_NAME_TO_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}
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

    def __init__(self) -> None:
        package_share = Path(get_package_share_directory(self.DESCRIPTION_PACKAGE))
        self.urdf_path = package_share / self.URDF_RELATIVE_PATH

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.neutral_q = pin.neutral(self.model)

        self.base_frame = self.BASE_FRAME
        self.end_effector_frame = self.END_EFFECTOR_FRAME
        self.joint_names = self.JOINT_NAMES
        self.base_joint_name = self.BASE_JOINT_NAME
        self.joint_name_to_index = self.JOINT_NAME_TO_INDEX

        self.end_effector_frame_id = self.model.getFrameId(self.end_effector_frame)
        self.controlled_joint_ids = tuple(
            self.model.getJointId(joint_name) for joint_name in self.joint_names
        )
        self.controlled_joint_idx_q = tuple(
            self.model.joints[joint_id].idx_q for joint_id in self.controlled_joint_ids
        )

        self.lower_limits = np.rad2deg(np.array(
            [self.model.lowerPositionLimit[idx_q] for idx_q in self.controlled_joint_idx_q],
            dtype=float,
        ))
        self.upper_limits = np.rad2deg(np.array(
            [self.model.upperPositionLimit[idx_q] for idx_q in self.controlled_joint_idx_q],
            dtype=float,
        ))
        self.nominal_q = q0.copy()
        self.set_nominal_q(self.nominal_q)
        self.last_solve_time_ms = 0.0

    def set_nominal_q(self, q: JointVector) -> None:
        q = np.asarray(q, dtype=float)
        self.nominal_q = q.copy()
        self.nominal_end_effector_y = self.get_end_effector_y(self.nominal_q)
        self.nominal_end_effector_z = self.get_end_effector_z(self.nominal_q)
        self.nominal_end_effector_rotation = self.get_end_effector_rotation(self.nominal_q)
        self.nominal_joint6_minus_joint5 = (
            self.get_joint_position("Joint6", self.nominal_q)
            - self.get_joint_position("Joint5", self.nominal_q)
        )

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

    def solve_with_fixed_joint1(self, q: JointVector) -> JointVector:
        """Solve for a configuration with Joint1 fixed."""
        q = np.asarray(q, dtype=float)

        opti = ca.Opti()
        q_var = opti.variable(len(self.joint_names))
        q_target = ca.DM(q)
        nominal_rotation = ca.DM(self.nominal_end_effector_rotation)
        rotation_error = self.get_casadi_end_effector_rotation(q_var) - nominal_rotation
        y_error = self.get_casadi_end_effector_y(q_var) - self.nominal_end_effector_y
        z_error = self.get_casadi_end_effector_z(q_var) - self.nominal_end_effector_z
        q_input_error = q_var - q_target
        opti.minimize(
            10 * ca.sumsqr(ca.vec(rotation_error))
            + 5 * ca.sumsqr(z_error)
            + ca.sumsqr(y_error)
            # + 0.001 * ca.sumsqr(q_input_error)
        )

        # Joint Angles
        opti.subject_to(opti.bounded(self.lower_limits, q_var, self.upper_limits))
        # No change to base angle
        opti.subject_to(q_var[0] == q[0])
        opti.set_initial(q_var, np.clip(q, self.lower_limits, self.upper_limits))
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"print_level": 0, "sb": "yes"},
        )

        start_time = time.perf_counter()
        solution = opti.solve()
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        return np.asarray(solution.value(q_var), dtype=float).reshape(-1)


if __name__ == "__main__":
    solver = D1IKSolver()
    q_test = q0.copy()
    q_test[0] = -40.0

    print("=" * 48)
    print("Input")
    print(f"  q_test_deg: {np.array2string(q_test, precision=1)}")
    print(f"  nominal_y:  {solver.nominal_end_effector_y:.3f}")
    print(f"  nominal_z:  {solver.nominal_end_effector_z:.3f}")
    print("-" * 48)

    q_solution = solver.solve_with_fixed_joint1(q_test)
    solution_y = solver.get_end_effector_y(q_solution)
    solution_z = solver.get_end_effector_z(q_solution)

    print("Solve")
    print(f"  solve_time_ms: {solver.last_solve_time_ms:.3f}")
    print("-" * 48)

    print("Result")
    print(f"  q_solution_deg: {np.array2string(q_solution, precision=1)}")
    print(
        f"  y: nominal={solver.nominal_end_effector_y:.3f} "
        f"solution={solution_y:.3f} "
        f"delta={solution_y - solver.nominal_end_effector_y:.3f}"
    )
    print(
        f"  z: nominal={solver.nominal_end_effector_z:.3f} "
        f"solution={solution_z:.3f} "
        f"delta={solution_z - solver.nominal_end_effector_z:.3f}"
    )
    print("=" * 48)

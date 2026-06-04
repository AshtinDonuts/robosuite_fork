import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


#: Joint names as they appear in the arm MJCF (before robot-id prefixing).
_ARM_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


class VX300S(ManipulatorModel):
    """
    Trossen Robotics ViperX 300 6DOF arm (vx300s) with integrated parallel jaw gripper.

    MJCF assets are derived from the ``trossen_vx300s`` entry in
    ``google-deepmind/mujoco_menagerie``; the STL meshes live in
    ``robosuite/models/assets/robots/vx300s/meshes/``.

    Joint layout::

        Index  MJCF joint name   Hardware joint name
        ─────────────────────────────────────────────────
        0      waist             waist
        1      shoulder          shoulder
        2      elbow             elbow
        3      forearm_roll      forearm_roll
        4      wrist_angle       wrist_angle
        5      wrist_rotate      wrist_rotate
        ─────────────────────────────────────────────────

    The gripper finger joints (``left_finger``, ``right_finger``) are driven by
    the ``VX300SGripper`` model and are not included in the 6-DOF arm count.

    Teleoperation: ``python -m robosuite.devices.leaderarm`` scales shoulder and
    elbow before the sim clamp (default gains 1.12 when ``--robot VX300S``).
    Adjust with ``--leaderarm-teleop-scale-shoulder`` / ``--leaderarm-teleop-scale-elbow``, or use
    ``--leaderarm-teleop-scale-pivot`` (6 angles in rad) to scale about a reference pose.

    Args:
        idn (int or str): Robot instance id used for MJCF element prefixing.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        self._use_logical_arm_dof = False
        super().__init__(xml_path_completion("robots/vx300s/robot.xml"), idn=idn)
        self._arms_joints = [self.correct_naming(n) for n in _ARM_JOINT_NAMES]
        self._arms_actuators = [self.correct_naming(n) for n in _ARM_JOINT_NAMES]
        self._use_logical_arm_dof = True

    def update_joints(self):
        super().update_joints()
        self._arms_joints = [self.correct_naming(n) for n in _ARM_JOINT_NAMES]

    def update_actuators(self):
        super().update_actuators()
        self._arms_actuators = [self.correct_naming(n) for n in _ARM_JOINT_NAMES]

    @property
    def joints(self):
        """6-DOF arm joints only; finger joints are exposed via the gripper model."""
        return list(self._arms_joints)

    @property
    def dof(self):
        if getattr(self, "_use_logical_arm_dof", False):
            return len(self._arms_joints)
        return len(self._joints)

    @classmethod
    def split_init_qpos(cls, qpos):
        """
        Map a logical init vector to ``(arm_qpos, finger_qpos_pair)``.

        Accepts:

        * length 6  – arm joints only; gripper uses XML defaults.
        * length 7  – 6 arm joints + 1 symmetric finger opening applied to
          both ``left_finger`` and ``right_finger`` (robosuite-style).
        """
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.size == 6:
            return qpos.copy(), None
        if qpos.size == 7:
            g = float(qpos[6])
            return qpos[:6].copy(), np.array([g, -g], dtype=float)
        raise ValueError(
            f"VX300S init_qpos expects length 6 (arm) or 7 (arm + gripper opening); got {qpos.size}."
        )

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "VX300SGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_vx300s"}

    @property
    def init_qpos(self):
        # Home pose from the upstream keyframe ("home") + open gripper.
        return np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.024], dtype=float)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.45

    @property
    def arm_type(self):
        return "single"

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ViperXAI(ManipulatorModel):
    """
    Trossen Viper XAI (WXAI) 6-DOF arm with integrated parallel gripper.

    MJCF follows ``trossen_arm_mujoco`` ``assets/wxai/wxai_follower.xml``. Copy
    the package's ``meshes/*.stl`` next to that tree into
    ``robosuite/models/assets/robots/viper_xai/meshes/`` (same filenames as in the
    upstream ``wxai_follower`` asset block).

    Args:
        idn (int or str): Robot instance id for MJCF prefixing.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        # During RobotModel.__init__, ``dof`` must match all MJCF joints (8) for set_joint_attribute; afterwards use 6.
        self._use_logical_arm_dof = False
        super().__init__(xml_path_completion("robots/viper_xai/robot.xml"), idn=idn)
        # Arm controller uses 6 DOF; carriage joints are driven by the gripper model.
        self._arms_joints = [self.correct_naming(f"joint_{i}") for i in range(6)]
        self._arms_actuators = [self.correct_naming(f"joint_{i}") for i in range(6)]
        self._use_logical_arm_dof = True

    def update_joints(self):
        super().update_joints()
        self._arms_joints = [self.correct_naming(f"joint_{i}") for i in range(6)]

    def update_actuators(self):
        super().update_actuators()
        self._arms_actuators = [self.correct_naming(f"joint_{i}") for i in range(6)]

    @property
    def joints(self):
        """Serial arm only (6); carriage slides are exposed via the gripper model."""
        return list(self._arms_joints)

    @property
    def dof(self):
        if getattr(self, "_use_logical_arm_dof", False):
            return len(self._arms_joints)
        return len(self._joints)

    @classmethod
    def split_init_qpos(cls, qpos):
        """
        Map a logical init vector to (arm_qpos, gripper_carriage_qpos).

        Accepts length 6 (arm only; use gripper XML defaults) or 7 (6 arm + 1 symmetric opening for both carriages, robosuite-style).
        """
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.size == 6:
            return qpos.copy(), None
        if qpos.size == 7:
            g = float(qpos[6])
            return qpos[:6].copy(), np.array([g, g], dtype=float)
        raise ValueError(
            f"ViperXAI init_qpos expects length 6 (arm) or 7 (arm + gripper opening); got {qpos.size}."
        )

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "ViperXAIGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_viperxai"}

    @property
    def init_qpos(self):
        # Logical default: 6 arm + 1 gripper opening (applied to both carriages after split).
        return np.concatenate([np.zeros(6, dtype=float), np.zeros(1, dtype=float)])

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

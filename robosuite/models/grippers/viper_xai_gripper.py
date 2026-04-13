"""
Integrated parallel jaw gripper for the Trossen Viper / WXAI arm.

The finger mechanism lives in the arm MJCF; this gripper only adds the
robosuite eef frame, sensors, and a position actuator on ``left_carriage_joint``.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ViperXAIGripper(GripperModel):
    """
    Args:
        idn (int or str): Passed as ``"{robot_id}_{arm}"`` by robosuite (e.g. ``0_right``).
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/viper_xai_gripper.xml"), idn=idn)
        # Robot joints use prefix ``robot{idn}_`` (see RobotModel.naming_prefix); gripper idn is ``{idn}_{arm}``.
        robot_id = str(self.idn).split("_")[0]
        for act in self.actuator:
            if act.get("joint") == "left_carriage_joint":
                act.set("joint", f"robot{robot_id}_left_carriage_joint")
                break
        self.current_action = np.zeros(self.dof)

    @property
    def joints(self):
        """Carriage slides live on the arm MJCF; both must be initialized for the equality constraint."""
        robot_id = str(self.idn).split("_")[0]
        p = f"robot{robot_id}_"
        return [f"{p}left_carriage_joint", f"{p}right_carriage_joint"]

    def exclude_from_prefixing(self, inp):
        # Joint lives on the arm subtree; keep unprefixed here, then fix in __init__.
        return inp == "left_carriage_joint"

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0])

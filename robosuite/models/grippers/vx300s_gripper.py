"""
Integrated parallel jaw gripper for the Trossen ViperX 300 (vx300s) arm.

The finger joints live in the arm MJCF; this gripper model only contributes
the robosuite EEF frame, F/T sensors, and a position actuator that drives
``left_finger`` (``right_finger`` is coupled via the MJCF equality constraint).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class VX300SGripper(GripperModel):
    """
    Gripper model for the Trossen ViperX 300 (vx300s) robot.

    The finger joints (``left_finger``, ``right_finger``) are defined in the arm
    MJCF and prefixed with ``robot{N}_`` at load time.  This class fixes the
    actuator joint reference to use the correct prefixed name so that robosuite
    can drive the gripper.

    Args:
        idn (int or str): Passed as ``"{robot_id}_{arm}"`` by robosuite (e.g. ``"0_right"``).
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/vx300s_gripper.xml"), idn=idn)
        # Rewrite the actuator's joint reference to the prefixed name that lives
        # on the arm MJCF after robosuite applies robot-id prefixing.
        robot_id = str(self.idn).split("_")[0]
        for act in self.actuator:
            if act.get("joint") == "left_finger":
                act.set("joint", f"robot{robot_id}_left_finger")
                break
        self.current_action = np.zeros(self.dof)

    @property
    def joints(self):
        """Both finger joints (required so the equality constraint initialises correctly)."""
        robot_id = str(self.idn).split("_")[0]
        p = f"robot{robot_id}_"
        return [f"{p}left_finger", f"{p}right_finger"]

    def exclude_from_prefixing(self, inp):
        # The joint lives on the arm subtree; skip prefixing here and fix in __init__.
        return inp == "left_finger"

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0]) * self.speed * np.sign(action),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def init_qpos(self):
        # Both finger joints initialised to open position (0.057 → mapped to 0.0 in normalised space).
        return np.array([0.0, 0.0])

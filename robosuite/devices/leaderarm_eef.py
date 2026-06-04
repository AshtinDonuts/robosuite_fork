"""
EEF-pose teleoperation driver for a physical Trossen leader arm.

This module turns leader-arm joint states into an end-effector pose via a
shadow VX300S MuJoCo model, then maps that pose onto a robosuite follower
robot. The result is robot-agnostic EEF teleop: a physical Trossen arm can
drive any virtual robot that exposes a pose-based arm controller.
"""

import abc
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from pynput.keyboard import Key, Listener

import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.mjcf_utils import xml_path_completion


def _mat_from_quat_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    return T.quat2mat(T.convert_quat(np.asarray(quat_wxyz, dtype=np.float64), to="xyzw"))


class _TrossenShadowKinematics:
    """Small FK-only MuJoCo model for the physical VX300S leader arm."""

    JOINT_NAMES: Tuple[str, ...] = (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    EEF_ROT_OFFSET = _mat_from_quat_wxyz((0.707105, 0.0, 0.707108, 0.0))

    def __init__(self):
        model_path = xml_path_completion("robots/vx300s/robot.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._joint_qpos_addr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
                for joint_name in self.JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    def forward(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(joint_positions, dtype=np.float64).reshape(len(self._joint_qpos_addr))
        self.data.qpos[self._joint_qpos_addr] = qpos
        mujoco.mj_forward(self.model, self.data)
        pos = np.asarray(self.data.site_xpos[self._site_id], dtype=np.float64).copy()
        rot = np.asarray(self.data.site_xmat[self._site_id], dtype=np.float64).reshape(3, 3).copy()
        return pos, rot @ self.EEF_ROT_OFFSET


class LeaderArm:
    """
    Pose-based teleoperation device driven by a physical leader arm.

    The leader arm is calibrated against the follower robot on `start_control()`:
    the current leader EEF pose becomes the neutral input pose and the current
    follower EEF pose becomes the neutral target pose. Subsequent leader motion
    is mapped into the follower base frame and emitted as both absolute and
    delta EEF commands.
    """

    def __init__(
        self,
        env,
        position_scale: float = 1.0,
        position_scale_xyz: Optional[Sequence[float]] = None,
        position_offset_xyz: Optional[Sequence[float]] = None,
        orientation_scale: float = 1.0,
        rotation_offset_rpy: Optional[Sequence[float]] = None,
        calibration_step: float = 0.01,
        calibration_step_rot: float = 0.05,
        leader_joint_scale: Optional[Sequence[float]] = None,
        leader_joint_scale_pivot: Optional[Sequence[float]] = None,
    ):
        self.env = env
        self.position_scale = float(position_scale)
        if position_scale_xyz is None:
            self.position_scale_xyz = np.full(3, self.position_scale, dtype=np.float64)
        else:
            self.position_scale_xyz = np.asarray(position_scale_xyz, dtype=np.float64)
            if self.position_scale_xyz.shape != (3,):
                raise ValueError(f"position_scale_xyz must have shape (3,); got {self.position_scale_xyz.shape}")
        if position_offset_xyz is None:
            self.position_offset_xyz = np.zeros(3, dtype=np.float64)
        else:
            self.position_offset_xyz = np.asarray(position_offset_xyz, dtype=np.float64)
            if self.position_offset_xyz.shape != (3,):
                raise ValueError(f"position_offset_xyz must have shape (3,); got {self.position_offset_xyz.shape}")
        self.orientation_scale = float(orientation_scale)
        if rotation_offset_rpy is None:
            self._rotation_offset_rpy = np.zeros(3, dtype=np.float64)
        else:
            rpy = np.asarray(rotation_offset_rpy, dtype=np.float64)
            if rpy.shape != (3,):
                raise ValueError(f"rotation_offset_rpy must have shape (3,); got {rpy.shape}")
            self._rotation_offset_rpy = rpy.copy()
        self.rotation_offset_mat = T.euler2mat(self._rotation_offset_rpy)
        self.calibration_step = float(calibration_step)
        self.calibration_step_rot = float(calibration_step_rot)
        self._leader_joint_scale = None if leader_joint_scale is None else np.asarray(leader_joint_scale, dtype=float)
        self._leader_joint_scale_pivot = (
            None if leader_joint_scale_pivot is None else np.asarray(leader_joint_scale_pivot, dtype=float)
        )

        self._reset_state = 0
        self._enabled = False
        self._all_robot_arms: Optional[List[List[str]]] = None
        self.grasp_states: List[List[bool]] = []
        self.active_arm_indices: List[int] = []
        self.active_robot = 0
        self._leader_anchor_pose: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._follower_anchor_pose: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}

        self._display_controls()
        self._listener = Listener(on_release=self._on_release)
        self._listener.start()

    def _on_release(self, key) -> None:
        try:
            if hasattr(key, "char") and key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                return
            if not self.grasp_states:
                return
            # Special (non-char) keys
            if key == Key.space:
                i, j = self.active_robot, self.active_arm_index
                self.grasp_states[i][j] = not self.grasp_states[i][j]
            elif key == Key.up:
                self.position_offset_xyz[2] += self.calibration_step
                self._print_calibration()
            elif key == Key.down:
                self.position_offset_xyz[2] -= self.calibration_step
                self._print_calibration()
            elif key == Key.left:
                self.position_offset_xyz[1] -= self.calibration_step
                self._print_calibration()
            elif key == Key.right:
                self.position_offset_xyz[1] += self.calibration_step
                self._print_calibration()
            # Char keys
            elif key.char == "s":
                arms = self.all_robot_arms[self.active_robot]
                self.active_arm_index = (self.active_arm_index + 1) % len(arms)
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots
            elif key.char == "[":
                self.position_offset_xyz[0] -= self.calibration_step
                self._print_calibration()
            elif key.char == "]":
                self.position_offset_xyz[0] += self.calibration_step
                self._print_calibration()
            elif key.char == "o":
                self._rotation_offset_rpy[1] -= self.calibration_step_rot
                self._update_rotation_offset()
                self._print_calibration()
            elif key.char == "p":
                self._rotation_offset_rpy[1] += self.calibration_step_rot
                self._update_rotation_offset()
                self._print_calibration()
        except AttributeError:
            pass

    @abc.abstractmethod
    def get_leader_joint_positions(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_leader_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def all_robot_arms(self) -> List[List[str]]:
        robots = getattr(self.env, "robots", None)
        assert robots is not None and all(r is not None for r in robots), (
            "Environment has no robots to control. "
            "Initialise the environment and call reset() before using the device."
        )
        if self._all_robot_arms is None:
            self._all_robot_arms = [robot.arms for robot in self.env.robots]
        return self._all_robot_arms

    @property
    def num_robots(self) -> int:
        return len(self.all_robot_arms)

    @property
    def active_arm_index(self) -> int:
        return self.active_arm_indices[self.active_robot]

    @active_arm_index.setter
    def active_arm_index(self, value: int):
        self.active_arm_indices[self.active_robot] = value

    @property
    def active_arm(self) -> str:
        return self.all_robot_arms[self.active_robot][self.active_arm_index]

    @property
    def grasp(self) -> bool:
        return self.grasp_states[self.active_robot][self.active_arm_index]

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print(f"{char}	{info}")

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot env)")
        print_command("↑ / ↓", "online calib: shift virtual EEF +/- z")
        print_command("← / →", "online calib: shift virtual EEF -/+ y")
        print_command("[ / ]", "online calib: shift virtual EEF -/+ x")
        print_command("o / p", "online calib: pitch -/+")
        print("")

    def _update_rotation_offset(self) -> None:
        self.rotation_offset_mat = T.euler2mat(self._rotation_offset_rpy)

    def _print_calibration(self) -> None:
        ox, oy, oz = self.position_offset_xyz
        pitch_deg = np.degrees(self._rotation_offset_rpy[1])
        print(
            f"[calib] pos_offset=({ox:+.4f}, {oy:+.4f}, {oz:+.4f}) m  "
            f"pitch={pitch_deg:+.1f}°",
            flush=True,
        )

    def _reset_internal_state(self):
        self.grasp_states = [[False] * len(arms) for arms in self.all_robot_arms]
        self.active_arm_indices = [0] * self.num_robots
        self.active_robot = 0
        self._leader_anchor_pose.clear()
        self._follower_anchor_pose.clear()

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        self._ensure_calibration(self.active_robot, self.active_arm)

    def _apply_joint_affine_map(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        if self._leader_joint_scale is None and self._leader_joint_scale_pivot is None:
            return qpos
        n = qpos.size
        scale = np.ones(n, dtype=np.float64) if self._leader_joint_scale is None else self._leader_joint_scale
        pivot = np.zeros(n, dtype=np.float64) if self._leader_joint_scale_pivot is None else self._leader_joint_scale_pivot
        if scale.shape != (n,) or pivot.shape != (n,):
            raise ValueError(
                f"leader_joint_scale / leader_joint_scale_pivot must have shape ({n},); "
                f"got scale {scale.shape}, pivot {pivot.shape}"
            )
        return pivot + scale * (qpos - pivot)

    def _get_follower_world_pose(self, robot, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        site_id = robot.eef_site_id[arm]
        pos = np.asarray(robot.sim.data.site_xpos[site_id], dtype=np.float64).copy()
        rot = np.asarray(robot.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        return pos, rot

    def _get_follower_base_pose(self, robot, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        world_pos, world_rot = self._get_follower_world_pose(robot, arm)
        base_pos = np.asarray(robot.sim.data.get_body_xpos(robot.robot_model.root_body), dtype=np.float64)
        base_rot = np.asarray(robot.sim.data.get_body_xmat(robot.robot_model.root_body), dtype=np.float64).reshape(3, 3)
        world_pose = T.make_pose(world_pos, world_rot)
        base_pose = T.make_pose(base_pos, base_rot)
        pose_in_base = T.pose_in_A_to_pose_in_B(world_pose, T.pose_inv(base_pose))
        pos_in_base, quat_in_base = T.mat2pose(pose_in_base)
        return np.asarray(pos_in_base, dtype=np.float64), T.quat2mat(np.asarray(quat_in_base, dtype=np.float64))

    def _ensure_calibration(self, robot_idx: int, arm: str) -> None:
        key = (robot_idx, arm)
        if key in self._leader_anchor_pose:
            return
        robot = self.env.robots[robot_idx]
        self._leader_anchor_pose[key] = self.get_leader_eef_pose()
        self._follower_anchor_pose[key] = self._get_follower_base_pose(robot, arm)

    def _map_leader_pose_to_target(self, robot, arm: str) -> Dict[str, np.ndarray]:
        key = (self.active_robot, arm)
        self._ensure_calibration(*key)
        leader_pos, leader_rot = self.get_leader_eef_pose()
        anchor_leader_pos, anchor_leader_rot = self._leader_anchor_pose[key]
        anchor_follower_pos_base, anchor_follower_rot_base = self._follower_anchor_pose[key]

        delta_pos_leader = self.position_scale_xyz * (leader_pos - anchor_leader_pos)
        rel_rot = leader_rot @ anchor_leader_rot.T
        rel_rotvec = T.quat2axisangle(T.mat2quat(rel_rot))
        rel_rot_scaled = T.quat2mat(T.axisangle2quat(self.orientation_scale * rel_rotvec))

        # position_offset_xyz is applied in follower base frame, after the leader delta
        target_pos_base = anchor_follower_pos_base + delta_pos_leader + self.position_offset_xyz
        # rotation_offset_mat is applied in follower base frame as a constant bias rotation
        target_rot_base = self.rotation_offset_mat @ rel_rot_scaled @ anchor_follower_rot_base

        base_pos = np.asarray(robot.sim.data.get_body_xpos(robot.robot_model.root_body), dtype=np.float64)
        base_rot = np.asarray(robot.sim.data.get_body_xmat(robot.robot_model.root_body), dtype=np.float64).reshape(3, 3)

        return {
            "pos_base": target_pos_base,
            "rot_base": target_rot_base,
            "pos_world": base_pos + base_rot @ target_pos_base,
            "rot_world": base_rot @ target_rot_base,
        }

    def _inverse_scale_delta(self, controller, scaled_delta: np.ndarray) -> np.ndarray:
        scaled_delta = np.asarray(scaled_delta, dtype=np.float64)
        input_min = np.asarray(controller.input_min, dtype=np.float64)
        input_max = np.asarray(controller.input_max, dtype=np.float64)
        output_min = np.asarray(controller.output_min, dtype=np.float64)
        output_max = np.asarray(controller.output_max, dtype=np.float64)
        scale = np.abs(output_max - output_min) / np.abs(input_max - input_min)
        out_center = (output_max + output_min) / 2.0
        in_center = (input_max + input_min) / 2.0
        norm_delta = (scaled_delta - out_center) / scale + in_center
        return np.clip(norm_delta, input_min, input_max)

    def _build_pose_action(self, robot, arm: str) -> Dict[str, np.ndarray]:
        controller = robot.part_controllers[arm]
        target = self._map_leader_pose_to_target(robot, arm)

        if not isinstance(controller, OperationalSpaceController):
            raise ValueError(
                f"leaderarm_eef only supports pose-based arm controllers; "
                f"got {controller.name!r} for arm {arm!r}"
            )

        if controller.input_ref_frame == "base":
            current_pos, current_rot = self._get_follower_base_pose(robot, arm)
            abs_pos = target["pos_base"]
            abs_rot = target["rot_base"]
        elif controller.input_ref_frame == "world":
            current_pos, current_rot = self._get_follower_world_pose(robot, arm)
            abs_pos = target["pos_world"]
            abs_rot = target["rot_world"]
        else:
            raise ValueError(f"Unsupported OSC input_ref_frame: {controller.input_ref_frame!r}")

        scaled_delta = np.concatenate([abs_pos - current_pos, orientation_error(abs_rot, current_rot)])
        norm_delta = self._inverse_scale_delta(controller, scaled_delta)
        abs_action = np.concatenate([abs_pos, T.quat2axisangle(T.mat2quat(abs_rot))])
        return {"abs": abs_action, "delta": norm_delta}

    def get_controller_state(self) -> Dict:
        joint_positions = self.get_leader_joint_positions()
        leader_pos, leader_rot = self.get_leader_eef_pose()
        return dict(
            joint_positions=joint_positions,
            eef_pos=leader_pos,
            eef_rot=leader_rot,
            grasp=self.grasp,
            reset=self._reset_state,
        )

    def input2action(self, goal_update_mode: str = "target") -> Optional[Dict]:
        del goal_update_mode

        state = self.get_controller_state()
        if state["reset"]:
            return None

        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm
        grasp_cmd = 1 if state["grasp"] else -1
        ac_dict: Dict[str, np.ndarray] = {}

        for arm in robot.arms:
            gripper = robot.gripper[arm]
            if arm != active_arm:
                current_pos, current_rot = self._get_follower_base_pose(robot, arm)
                ac_dict[f"{arm}_abs"] = np.concatenate([current_pos, T.quat2axisangle(T.mat2quat(current_rot))])
                ac_dict[f"{arm}_delta"] = np.zeros(6, dtype=np.float64)
                ac_dict[arm] = ac_dict[f"{arm}_abs"].copy()
                ac_dict[f"{arm}_gripper"] = np.zeros(gripper.dof, dtype=np.float64)
                continue

            arm_action = self._build_pose_action(robot, arm)
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[arm] = arm_action["abs"].copy()

            if hasattr(gripper, "grasp_qpos"):
                ac_dict[f"{arm}_gripper"] = gripper.grasp_qpos[grasp_cmd]
            else:
                ac_dict[f"{arm}_gripper"] = np.array([grasp_cmd] * gripper.dof, dtype=np.float64)

        return ac_dict


class ROS2LeaderArm(LeaderArm):
    """
    ROS2-backed leader arm device.

    Generic ROS2 instances expose the raw leader joints but do not know how to
    compute FK unless a subclass overrides `get_leader_eef_pose()`.
    """

    DEFAULT_JOINT_NAMES: List[str] = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    ]

    def __init__(
        self,
        env,
        topic: str = "/joint_states",
        joint_names: Optional[List[str]] = None,
        node_name: str = "robosuite_leader_arm_eef",
        gripper_joint: Optional[str] = None,
        gripper_close_threshold: float = 0.0,
        qos_depth: int = 10,
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise ImportError(
                "rclpy and sensor_msgs are required for ROS2LeaderArm. "
                "Source your ROS 2 workspace and ensure the packages are installed."
            ) from exc

        self._joint_names = joint_names if joint_names is not None else list(self.DEFAULT_JOINT_NAMES)
        self._gripper_joint = gripper_joint
        self._gripper_close_threshold = float(gripper_close_threshold)
        self._positions: Dict[str, float] = {}
        self._lock = threading.Lock()

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)
        self._subscription = self._node.create_subscription(
            JointState,
            topic,
            self._joint_state_callback,
            qos_depth,
        )
        self._spin_thread = threading.Thread(target=self._spin, name="ros2_leader_arm_spin", daemon=True)
        self._spin_thread.start()

    def _joint_state_callback(self, msg) -> None:
        incoming = dict(zip(msg.name, msg.position))
        with self._lock:
            self._positions.update(incoming)
            if self._gripper_joint is not None and self.grasp_states:
                gripper_pos = self._positions.get(self._gripper_joint)
                if gripper_pos is not None:
                    closed = gripper_pos < self._gripper_close_threshold
                    for arm_idx in range(len(self.grasp_states[self.active_robot])):
                        self.grasp_states[self.active_robot][arm_idx] = closed

    def _spin(self) -> None:
        import rclpy

        rclpy.spin(self._node)

    def get_leader_joint_positions(self) -> np.ndarray:
        with self._lock:
            positions = [self._positions.get(name, 0.0) for name in self._joint_names]
        return self._apply_joint_affine_map(np.asarray(positions, dtype=np.float64))

    def get_leader_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Generic ROS2LeaderArm does not know the leader arm kinematics. "
            "Use TrossenArmLeaderArm or override get_leader_eef_pose()."
        )

    @property
    def joint_names(self) -> List[str]:
        return list(self._joint_names)

    @property
    def latest_positions(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._positions)

    def close(self) -> None:
        import rclpy

        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TrossenArmLeaderArm(ROS2LeaderArm):
    """ROS2 leader arm backed by a VX300S shadow FK model."""

    DEFAULT_JOINT_NAMES: List[str] = list(_TrossenShadowKinematics.JOINT_NAMES)

    def __init__(
        self,
        env,
        topic: str = "/joint_states",
        gripper_close_threshold: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("node_name", "robosuite_trossen_leader_eef")
        super().__init__(
            env,
            topic=topic,
            joint_names=list(self.DEFAULT_JOINT_NAMES),
            gripper_joint="left_finger",
            gripper_close_threshold=gripper_close_threshold,
            **kwargs,
        )
        self._shadow_fk = _TrossenShadowKinematics()

    def get_leader_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._shadow_fk.forward(self.get_leader_joint_positions())


def main() -> None:
    import argparse
    import time
    from copy import deepcopy

    parser = argparse.ArgumentParser(
        description="Teleop test: physical Trossen leader arm -> robosuite OSC_POSE / FABRIC_OSC_POSE follower."
    )
    parser.add_argument("--topic", type=str, default="/joint_states", help="sensor_msgs/JointState topic")
    parser.add_argument("--period", type=float, default=0.5, help="Seconds between diagnostic prints")
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robot", type=str, default="UR5e")
    parser.add_argument("--position-scale", type=float, default=1.0, help="Scale applied to leader EEF translation")
    parser.add_argument("--position-scale-x", type=float, default=None, help="Per-axis x scale for leader EEF translation")
    parser.add_argument("--position-scale-y", type=float, default=None, help="Per-axis y scale for leader EEF translation")
    parser.add_argument("--position-scale-z", type=float, default=None, help="Per-axis z scale for leader EEF translation")
    parser.add_argument(
        "--position-offset-x",
        type=float,
        default=0.0,
        help="Additive x offset in follower base frame after scaled leader translation",
    )
    parser.add_argument(
        "--position-offset-y",
        type=float,
        default=0.0,
        help="Additive y offset in follower base frame after scaled leader translation",
    )
    parser.add_argument(
        "--position-offset-z",
        type=float,
        default=0.0,
        help="Additive z offset in follower base frame after scaled leader translation",
    )
    parser.add_argument("--orientation-scale", type=float, default=1.0, help="Scale applied to leader EEF rotation")
    parser.add_argument(
        "--rotation-offset-roll",
        type=float,
        default=0.0,
        help="Constant roll offset (rad) added to target orientation in follower base frame",
    )
    parser.add_argument(
        "--rotation-offset-pitch",
        type=float,
        default=0.0,
        help="Constant pitch offset (rad) added to target orientation in follower base frame",
    )
    parser.add_argument(
        "--rotation-offset-yaw",
        type=float,
        default=0.0,
        help="Constant yaw offset (rad) added to target orientation in follower base frame",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-shoulder",
        type=float,
        default=None,
        help="Optional affine scale for the leader shoulder joint before FK. Default: 1.12 for VX300S, else 1.0.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-elbow",
        type=float,
        default=None,
        help="Optional affine scale for the leader elbow joint before FK. Default: 1.12 for VX300S, else 1.0.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-pivot",
        type=float,
        nargs=6,
        metavar=("w", "sh", "el", "fr", "wa", "wr"),
        default=None,
        help="Optional pivot (rad) for the leader joint affine map before FK.",
    )
    parser.add_argument("--no-render", action="store_true", help="Headless (no on-screen window).")
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=("mjviewer", "mujoco"))
    parser.add_argument("--render-camera", type=str, default="frontview")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--refactor-arm-names", nargs="+", default=["right"])
    parser.add_argument(
        "--arm-input-type",
        choices=("absolute", "delta"),
        default="absolute",
        help="Follower OSC input type. Absolute is recommended for pose-mapped teleop.",
    )
    parser.add_argument("--gripper-close-threshold", type=float, default=0.0)
    parser.add_argument(
        "--calibration-step",
        type=float,
        default=0.01,
        help="Position step size (m) per keypress for online calibration (default 1 cm)",
    )
    parser.add_argument(
        "--calibration-step-rot",
        type=float,
        default=0.05,
        help="Rotation step size (rad) per keypress for online pitch calibration (default ~3°)",
    )
    args = parser.parse_args()

    import robosuite as suite
    from robosuite.controllers import load_part_controller_config
    from robosuite.controllers.composite.composite_controller import WholeBody
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper

    if args.leaderarm_teleop_scale_shoulder is None:
        args.leaderarm_teleop_scale_shoulder = 1.12 if args.robot == "VX300S" else 1.0
    if args.leaderarm_teleop_scale_elbow is None:
        args.leaderarm_teleop_scale_elbow = 1.12 if args.robot == "VX300S" else 1.0

    arm_part_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_config = refactor_composite_controller_config(
        arm_part_config,
        args.robot,
        args.refactor_arm_names,
    )
    for arm in args.refactor_arm_names:
        arm_cfg = controller_config.get("body_parts", {}).get(arm)
        if isinstance(arm_cfg, dict) and arm_cfg.get("type") in {"OSC_POSE", "FABRIC_OSC_POSE"}:
            arm_cfg["input_type"] = args.arm_input_type

    env = suite.make(
        args.environment,
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=not args.no_render,
        has_offscreen_renderer=False,
        renderer=args.renderer,
        render_camera=args.render_camera,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=args.control_freq,
        reward_shaping=True,
    )
    # Match demo_device_control: show grip_site / grip_site_cylinder EE aids each step.
    env = VisualizationWrapper(env, indicator_configs=None)
    env.reset()
    if not args.no_render:
        env.render()

    teleop_scale = np.array(
        [1.0, args.leaderarm_teleop_scale_shoulder, args.leaderarm_teleop_scale_elbow, 1.0, 1.0, 1.0],
        dtype=float,
    )
    position_scale_xyz = np.array(
        [
            args.position_scale if args.position_scale_x is None else args.position_scale_x,
            args.position_scale if args.position_scale_y is None else args.position_scale_y,
            args.position_scale if args.position_scale_z is None else args.position_scale_z,
        ],
        dtype=float,
    )
    position_offset_xyz = np.array([args.position_offset_x, args.position_offset_y, args.position_offset_z], dtype=float)
    rotation_offset_rpy = np.array(
        [args.rotation_offset_roll, args.rotation_offset_pitch, args.rotation_offset_yaw], dtype=float
    )
    teleop_kw: Dict[str, object] = dict(
        env=env,
        topic=args.topic,
        position_scale=args.position_scale,
        position_scale_xyz=position_scale_xyz,
        position_offset_xyz=position_offset_xyz,
        orientation_scale=args.orientation_scale,
        rotation_offset_rpy=rotation_offset_rpy,
        calibration_step=args.calibration_step,
        calibration_step_rot=args.calibration_step_rot,
        leader_joint_scale=teleop_scale,
        gripper_close_threshold=args.gripper_close_threshold,
    )
    if args.leaderarm_teleop_scale_pivot is not None:
        teleop_kw["leader_joint_scale_pivot"] = tuple(args.leaderarm_teleop_scale_pivot)
    device = TrossenArmLeaderArm(**teleop_kw)

    anchor_delay_sec = 5
    print("Simulation started. Hold the leader arm in the desired neutral pose.")
    print("Anchor pose capture countdown:")
    deadline = time.time() + anchor_delay_sec
    last_printed = anchor_delay_sec
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        count = int(remaining) + 1
        if count != last_printed:
            print(f"  {count}...")
            last_printed = count
        if not args.no_render and env.viewer is not None:
            env.viewer.update()
    print("Capturing anchor pose now.")
    device.start_control()
    names = getattr(device, "joint_names", None)

    def _prev_gripper_actions():
        return [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

    all_prev_gripper_actions = _prev_gripper_actions()

    print(
        f"Listening on {args.topic!r}; "
        f"arm_input_type={args.arm_input_type!r}; "
        f"render={'off' if args.no_render else args.renderer}; "
        f"prints every {args.period} s. Press q to reset pose. Ctrl+C to exit.\n"
    )

    try:
        last_print = 0.0
        while True:
            input_ac_dict = device.input2action()
            if input_ac_dict is None:
                print("\nReset requested. Exiting teleop loop.")
                break

            action_dict = deepcopy(input_ac_dict)
            active_robot_model = env.robots[device.active_robot]
            for arm in active_robot_model.arms:
                if isinstance(active_robot_model.composite_controller, WholeBody):
                    controller_input_type = active_robot_model.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot_model.part_controllers[arm].input_type
                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError(f"Unsupported controller input_type: {controller_input_type!r}")

            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)
            ]
            env_action[device.active_robot] = active_robot_model.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            env.step(env_action)
            if not args.no_render:
                env.render()

            now = time.time()
            if now - last_print >= args.period:
                last_print = now
                state = device.get_controller_state()
                jp = state["joint_positions"]
                if names is not None:
                    joint_msg = ", ".join(f"{n}={v:+.3f}" for n, v in zip(names, jp))
                else:
                    joint_msg = np.array2string(jp, precision=3, floatmode="fixed")
                eef_pos = np.array2string(state["eef_pos"], precision=3, floatmode="fixed")
                print(f"[leader joints] {joint_msg}")
                print(f"[leader eef] pos={eef_pos}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        device.close()
        env.close()


if __name__ == "__main__":
    main()

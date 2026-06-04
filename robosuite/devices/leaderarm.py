"""
Driver class for a Leader Arm teleoperation device.

Provides direct joint-to-joint mapping from a physical or simulated leader arm
to a follower robot's joint-space controller.  Intentionally does NOT inherit
from Device because it bypasses OSC / IK control and operates purely in joint
space.

Concrete implementation for Trossen Robotics hardware (via ROS 2)::

    device = TrossenArmLeaderArm(env, topic="/leader_solo/joint_states")
    obs = env.reset()
    device.start_control()
    while True:
        ac_dict = device.input2action()
        if ac_dict is None:
            break
        env.step(robot.create_action_vector(ac_dict))
    device.close()

Trossen ``/leader_solo/joint_states`` layout::

    Index  Joint name        Notes
    ─────────────────────────────────────────────────────
    0      waist
    1      shoulder
    2      elbow
    3      forearm_roll
    4      wrist_angle
    5      wrist_rotate
    6      gripper           raw gripper actuator position
    7      left_finger       mirrored finger state
    8      right_finger      mirrored finger state
    ─────────────────────────────────────────────────────

The 6 arm joints (indices 0–5) are forwarded directly to the JOINT_POSITION
controller.  The gripper open/close state is inferred from the hardware
``gripper`` actuator position published on the JointState topic.
"""

import abc
import threading
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from pynput.keyboard import Key, Listener


class LeaderArm:
    """
    Teleoperation device that maps joint positions from a leader arm directly to
    the follower robot's joints.

    Unlike Device-based controllers (Keyboard, SpaceMouse) which command
    end-effector targets in operational space, LeaderArm sends joint-position or
    joint-delta targets and is designed for use with ``JOINT_POSITION`` (or
    ``JOINT_VELOCITY``) controllers.

    Usage pattern::

        device = MyLeaderArm(env)          # subclass of LeaderArm
        obs = env.reset()
        device.start_control()

        while True:
            ac_dict = device.input2action()
            if ac_dict is None:            # reset triggered
                break
            action = robot.create_action_vector(ac_dict)
            env.step(action)

    Subclass this and override :meth:`get_leader_joint_positions` to connect to
    specific hardware (serial/USB, ROS topic, shadow simulation, …).

    Args:
        env (RobotEnv): The environment containing the follower robot(s).
        joint_sensitivity (float): Scalar multiplier applied to joint deltas.
            Values > 1 amplify motion; values < 1 attenuate it.
        joint_limits_safety_factor (float): Fraction of each joint's usable
            range (0–1).  The leader positions are clamped so they stay within
            ``factor * range / 2`` of the joint-limit centre.  Set to 1.0 to
            use the full range.
        leader_joint_scale (sequence of float, optional): Per-joint gain applied
            before sim limit clamping: ``q' = pivot + scale * (q - pivot)``.
            Length must match arm DOF. ``None`` means all ones.
        leader_joint_scale_pivot (sequence of float, optional): Pivot for the
            affine map above (same length as ``leader_joint_scale``).  ``None``
            means all zeros (pure gain vs zero pose).
        leader_joint_angle_scale (float or sequence of float, optional): Scales
            each leader joint angle about zero before sim mapping:
            ``q'[i] = scale[i] * q[i]`` (``0`` rad always stays ``0``).

            * **Scalar** (``float``): one multiplier broadcast to every arm
              joint — e.g. ``2.0`` turns ``15`` rad into ``30`` rad on all joints.
            * **Tuple / sequence** (length = arm DOF, typically 6 for Trossen):
              one multiplier per joint in controller order
              (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate).
              E.g. ``(1.0, 2.0, 2.0, 1.0, 1.0, 1.0)`` doubles only shoulder
              and elbow magnitude.

            Applied before :attr:`leader_joint_scale` / pivot.  ``None`` disables
            this step (equivalent to all ones).
    """

    def __init__(
        self,
        env,
        joint_sensitivity: float = 1.0,
        joint_limits_safety_factor: float = 0.95,
        leader_joint_scale: Optional[Sequence[float]] = None,
        leader_joint_scale_pivot: Optional[Sequence[float]] = None,
        leader_joint_angle_scale: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.env = env
        self.joint_sensitivity = joint_sensitivity
        self.joint_limits_safety_factor = joint_limits_safety_factor
        self._leader_joint_scale: Optional[np.ndarray] = (
            None if leader_joint_scale is None else np.asarray(leader_joint_scale, dtype=float)
        )
        self._leader_joint_scale_pivot: Optional[np.ndarray] = (
            None if leader_joint_scale_pivot is None else np.asarray(leader_joint_scale_pivot, dtype=float)
        )
        self._leader_joint_angle_scale: Optional[np.ndarray] = (
            None if leader_joint_angle_scale is None else np.asarray(leader_joint_angle_scale, dtype=float)
        )

        self._reset_state: int = 0
        self._enabled: bool = False
        self._all_robot_arms: Optional[List[List[str]]] = None

        # Initialised properly in _reset_internal_state / start_control
        self.grasp_states: List[List[bool]] = []
        self.active_arm_indices: List[int] = []
        self.active_robot: int = 0

        self._display_controls()

        # Keyboard listener handles gripper, reset, and arm/robot switching
        self._listener = Listener(on_release=self._on_release)
        self._listener.start()

    def _on_release(self, key) -> None:
        """q: reset signal; space: toggle grasp; s: next arm; =: next robot."""
        try:
            if hasattr(key, "char") and key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                return
            if not self.grasp_states:
                return
            if key == Key.space:
                i, j = self.active_robot, self.active_arm_index
                self.grasp_states[i][j] = not self.grasp_states[i][j]
            elif key.char == "s":
                arms = self.all_robot_arms[self.active_robot]
                self.active_arm_index = (self.active_arm_index + 1) % len(arms)
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_leader_joint_positions(self) -> np.ndarray:
        """
        Return the current joint positions (radians) of the leader arm as a
        1-D numpy array whose length equals the number of controllable arm
        joints of the active follower arm.

        Override this method in subclasses to interface with physical or
        simulated hardware::

            def get_leader_joint_positions(self):
                return np.array(self.hardware_driver.read_joints())
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Robot / environment helpers
    # ------------------------------------------------------------------

    @property
    def all_robot_arms(self) -> List[List[str]]:
        """Nested list: ``all_robot_arms[robot_idx]`` → list of arm names."""
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot env)")
        print("")

    def _reset_internal_state(self):
        """Reset mutable control state (does not clear the reset signal)."""
        self.grasp_states = [[False] * len(arms) for arms in self.all_robot_arms]
        self.active_arm_indices = [0] * self.num_robots
        self.active_robot = 0

    def start_control(self):
        """
        Initialise internal state and enable command forwarding.

        Call this once after ``env.reset()`` and before the teleoperation loop.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    # ------------------------------------------------------------------
    # State retrieval
    # ------------------------------------------------------------------

    def get_controller_state(self) -> Dict:
        """
        Return the current device state.

        Returns:
            dict:
                * ``joint_positions`` – np.ndarray of leader joint angles (rad)
                * ``grasp``           – bool, True means gripper closed
                * ``reset``           – int, 1 when a user reset was requested
        """
        raw = np.asarray(self.get_leader_joint_positions(), dtype=float)
        return dict(
            joint_positions=self._scale_leader_joint_angles(raw),
            grasp=self.grasp,
            reset=self._reset_state,
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def input2action(self) -> Optional[Dict]:
        """
        Convert the leader arm's joint positions into an action dict for
        ``env.step()``.

        For each arm the returned dict contains:

        * ``{arm}``         – joint targets (same array as ``{arm}_abs``)
        * ``{arm}_abs``     – absolute target joint positions (rad), clamped
        * ``{arm}_delta``   – delta from the follower's current joint positions,
                              scaled by ``joint_sensitivity``
        * ``{arm}_gripper`` – gripper command array

        Inactive arms (not the currently selected arm) receive zero-delta /
        hold-current-position commands so they remain stationary.

        Returns:
            Optional[Dict]: Action dict, or ``None`` if a reset was triggered.
        """
        state = self.get_controller_state()
        if state["reset"]:
            return None

        leader_qpos: np.ndarray = state["joint_positions"]
        grasp: bool = state["grasp"]
        grasp_cmd: int = 1 if grasp else -1

        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        ac_dict: Dict = {}

        for arm in robot.arms:
            controller = robot.part_controllers[arm]
            gripper = robot.gripper[arm]
            n_joints = len(controller.joint_indexes["joints"])
            current_qpos = np.array(robot.sim.data.qpos[controller.qpos_index])

            if arm != active_arm:
                # Hold inactive arms at their current configuration
                ac_dict[f"{arm}_abs"] = current_qpos.copy()
                ac_dict[f"{arm}_delta"] = np.zeros(n_joints)
                ac_dict[arm] = current_qpos.copy()
                ac_dict[f"{arm}_gripper"] = np.zeros(gripper.dof)
                continue

            target_qpos = self._map_leader_to_follower(leader_qpos, robot, arm)
            delta_qpos = (target_qpos - current_qpos) * self.joint_sensitivity

            ac_dict[f"{arm}_abs"] = target_qpos
            ac_dict[f"{arm}_delta"] = delta_qpos
            ac_dict[arm] = target_qpos

            if hasattr(gripper, "grasp_qpos"):
                ac_dict[f"{arm}_gripper"] = gripper.grasp_qpos[grasp_cmd]
            else:
                ac_dict[f"{arm}_gripper"] = np.array([grasp_cmd] * gripper.dof)

        return ac_dict

    # ------------------------------------------------------------------
    # Joint mapping / clamping
    # ------------------------------------------------------------------

    def _scale_leader_joint_angles(self, leader_qpos: np.ndarray) -> np.ndarray:
        """
        Apply configured joint-angle scaling to raw leader positions (rad).

        Order: angle magnitude (``leader_joint_angle_scale``), then affine
        teleop map (``leader_joint_scale`` / ``leader_joint_scale_pivot``).
        """
        target = np.array(leader_qpos, dtype=float)
        n = target.size

        if self._leader_joint_angle_scale is not None:
            angle_scale = self._leader_joint_angle_scale
            if angle_scale.ndim == 0:
                angle_scale = np.full(n, float(angle_scale), dtype=float)
            elif angle_scale.shape != (n,):
                raise ValueError(
                    f"leader_joint_angle_scale must be a scalar or shape ({n},); "
                    f"got {angle_scale.shape}"
                )
            target = angle_scale * target

        if self._leader_joint_scale is not None or self._leader_joint_scale_pivot is not None:
            scale = np.ones(n, dtype=float) if self._leader_joint_scale is None else self._leader_joint_scale
            pivot = np.zeros(n, dtype=float) if self._leader_joint_scale_pivot is None else self._leader_joint_scale_pivot
            if scale.shape != (n,) or pivot.shape != (n,):
                raise ValueError(
                    f"leader_joint_scale / pivot must have shape ({n},); "
                    f"got scale {scale.shape}, pivot {pivot.shape}"
                )
            target = pivot + scale * (target - pivot)

        return target

    def _map_leader_to_follower(
        self, leader_qpos: np.ndarray, robot, arm: str
    ) -> np.ndarray:
        """
        Map raw leader joint positions to follower joint targets.

        The default implementation is a 1-to-1 identity mapping (same number
        of joints assumed) with limit clamping.  Override to add per-joint
        scaling, sign flips, or index remapping when leader and follower
        kinematic chains differ.

        Args:
            leader_qpos: Joint angles from the leader, shape ``(n,)`` in rad.
            robot: The follower robot object.
            arm: Name of the arm being controlled.

        Returns:
            np.ndarray: Target joint positions for the follower, shape ``(n,)``.
        """
        controller = robot.part_controllers[arm]
        target = self._scale_leader_joint_angles(leader_qpos)

        jnt_range = robot.sim.model.jnt_range[controller.joint_indexes["joints"]]
        lo = jnt_range[:, 0]
        hi = jnt_range[:, 1]

        margin = (hi - lo) * (1.0 - self.joint_limits_safety_factor) / 2.0
        target = np.clip(target, lo + margin, hi - margin)

        return target


class ROS2LeaderArm(LeaderArm):
    """
    LeaderArm implementation that reads joint positions from a ROS 2 topic.

    Subscribes to a ``sensor_msgs/msg/JointState`` topic and extracts positions
    for a caller-specified ordered list of joint names.  ROS 2 is spun in a
    background daemon thread so it does not block the teleoperation loop.

    Optionally, a *gripper joint* can be designated so that the hardware finger
    position drives the grasp state automatically (instead of the keyboard
    spacebar toggle).  When ``gripper_joint`` is set, the grasp state is
    updated every time a new message arrives: the gripper is considered
    *closed* whenever the joint position is below ``gripper_close_threshold``.

    Args:
        env: The robosuite environment containing the follower robot(s).
        topic (str): ROS 2 topic name publishing ``JointState`` messages.
        joint_names (List[str]): Ordered list of joint names whose positions
            are returned by :meth:`get_leader_joint_positions`.  Defaults to
            the seven standard Trossen arm joints (waist → gripper), excluding
            the finger pair.
        node_name (str): Name given to the ROS 2 node created internally.
        gripper_joint (str | None): Name of a joint whose position is used to
            infer grasp state.  When ``None`` (default) the keyboard spacebar
            toggle is used instead.
        gripper_close_threshold (float): Position threshold (rad) below which
            ``gripper_joint`` is considered closed.  Ignored when
            ``gripper_joint`` is ``None``.
        qos_depth (int): History depth for the ROS 2 subscription QoS profile.
        **kwargs: Forwarded verbatim to :class:`LeaderArm`.

    Example::

        device = ROS2LeaderArm(
            env,
            topic="/leader/joint_states",
            joint_names=["waist", "shoulder", "elbow",
                         "forearm_roll", "wrist_angle", "wrist_rotate"],
            gripper_joint="gripper",
            gripper_close_threshold=0.0,
        )
        device.start_control()
        while True:
            ac_dict = device.input2action()
            if ac_dict is None:
                break
            env.step(robot.create_action_vector(ac_dict))
        device.close()
    """

    #: Default joint name order for a 7-DOF Trossen arm (gripper included).
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
        node_name: str = "robosuite_leader_arm",
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

        self._joint_names: List[str] = (
            joint_names if joint_names is not None else list(self.DEFAULT_JOINT_NAMES)
        )
        self._gripper_joint: Optional[str] = gripper_joint
        self._gripper_close_threshold: float = gripper_close_threshold

        # Latest positions keyed by joint name; protected by _lock.
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

        self._spin_thread = threading.Thread(
            target=self._spin, name="ros2_leader_arm_spin", daemon=True
        )
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # ROS 2 internals
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg) -> None:
        """Store the latest position for every joint reported in the message."""
        incoming: Dict[str, float] = dict(zip(msg.name, msg.position))

        with self._lock:
            self._positions.update(incoming)

            # Derive grasp state from hardware when a gripper joint is given.
            if self._gripper_joint is not None and self.grasp_states:
                gripper_pos = self._positions.get(self._gripper_joint)
                if gripper_pos is not None:
                    closed = gripper_pos < self._gripper_close_threshold
                    # Write through to all arms of the active robot.
                    for arm_idx in range(len(self.grasp_states[self.active_robot])):
                        self.grasp_states[self.active_robot][arm_idx] = closed

    def _spin(self) -> None:
        import rclpy

        rclpy.spin(self._node)

    # ------------------------------------------------------------------
    # LeaderArm interface
    # ------------------------------------------------------------------

    def get_leader_joint_positions(self) -> np.ndarray:
        """
        Return an array of joint positions (rad) in the order of
        :attr:`joint_names`.

        Joints that have not yet been received default to ``0.0``.

        Returns:
            np.ndarray: Shape ``(len(joint_names),)``, dtype ``float64``.
        """
        with self._lock:
            positions = [self._positions.get(name, 0.0) for name in self._joint_names]
        return np.array(positions, dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def joint_names(self) -> List[str]:
        """Ordered list of joint names read from the ROS 2 topic."""
        return list(self._joint_names)

    @property
    def latest_positions(self) -> Dict[str, float]:
        """Snapshot of the most recent position for every received joint."""
        with self._lock:
            return dict(self._positions)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Destroy the ROS 2 node and shut down the rclpy context."""
        import rclpy

        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TrossenArmLeaderArm(ROS2LeaderArm):
    """
    Concrete ROS 2 leader arm for Trossen Robotics single-arm hardware.

    Reads joint positions from a ``sensor_msgs/msg/JointState`` topic and maps
    them to a robosuite follower robot following the qpos layout described in
    ``TrossenAIStationaryTask.before_step`` (``trossen_arm_mujoco/sim_env.py``):

    .. code-block:: text

        HW joint name    HW msg idx  Sim qpos idx  Role
        ──────────────────────────────────────────────────────────
        waist                 0           0         arm joint
        shoulder              1           1         arm joint
        elbow                 2           2         arm joint
        forearm_roll          3           3         arm joint
        wrist_angle           4           4         arm joint
        wrist_rotate          5           5         arm joint
        gripper               6           –         raw gripper actuator state
        left_finger           7           –         mirrored finger state
        right_finger          8           –         mirrored finger state
        ──────────────────────────────────────────────────────────

    :meth:`get_leader_joint_positions` returns **only the 6 arm joints**
    (waist → wrist_rotate), matching the DOF expected by the
    ``JOINT_POSITION`` controller on the follower.

    Grasp state is derived from the ROS ``gripper`` actuator position.  The
    ``left_finger`` / ``right_finger`` entries are mirrored finger states and
    can sit near zero, so they are not reliable open/close signals.  The
    gripper is treated as **closed** whenever
    ``gripper < gripper_close_threshold``.

    Args:
        env: The robosuite environment containing the follower robot(s).
        topic (str): ROS 2 topic publishing ``sensor_msgs/msg/JointState``.
        gripper_close_threshold (float): ``gripper`` position (rad) below
            which the gripper is treated as closed.  Defaults to ``0.0``.
        leader_joint_angle_scale (float or sequence of float, optional): See
            :class:`LeaderArm` — pass a **scalar** to scale every arm joint the
            same way, or a **6-tuple** with one multiplier per joint
            (waist … wrist_rotate).
        **kwargs: Forwarded verbatim to :class:`ROS2LeaderArm`.

    Example::

        device = TrossenArmLeaderArm(
            env,
            topic="/leader/joint_states",
            gripper_close_threshold=0.0,
        )
        obs = env.reset()
        device.start_control()
        while True:
            ac_dict = device.input2action()
            if ac_dict is None:
                break
            env.step(robot.create_action_vector(ac_dict))
        device.close()
    """

    #: The 6-DOF arm joints forwarded to the JOINT_POSITION controller.
    DEFAULT_JOINT_NAMES: List[str] = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]

    def __init__(
        self,
        env,
        topic: str = "/joint_states",
        gripper_close_threshold: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("node_name", "robosuite_trossen_leader")
        super().__init__(
            env,
            topic=topic,
            # Fix arm joint names; finger joints are outside the controller DOF.
            joint_names=list(self.DEFAULT_JOINT_NAMES),
            # Hardware JointState publishes the real gripper actuator as "gripper";
            # finger entries are mirrored and may hover around zero.
            gripper_joint="gripper",
            gripper_close_threshold=gripper_close_threshold,
            **kwargs,
        )


def main() -> None:
    """
    Subscribe to a ROS 2 ``JointState`` topic, step the **Lift** task (default)
    with **JOINT_POSITION** control, and show a live viewer.

    Run from the repo root::

        python -m robosuite.devices.leaderarm --topic /leader_solo/joint_states

    Arm commands use **absolute** joint goals by default: leader joint angles
    (rad) from ``JointState`` are clamped to sim limits and passed as
    ``{arm}_abs`` to ``JOINT_POSITION``. Use ``--arm-input-type delta`` for
    per-step ``{arm}_delta`` (scaled by :attr:`LeaderArm.joint_sensitivity`).

    Requires ``rclpy``, ``sensor_msgs``, and a running publisher on the topic.
    """
    import argparse
    import time
    from copy import deepcopy

    parser = argparse.ArgumentParser(
        description="Teleop test: ROS leader arm → Lift (or chosen env) with optional viewer."
    )
    parser.add_argument(
        "--impl",
        choices=("trossen", "ros2"),
        default="trossen",
        help="trossen: TrossenArmLeaderArm (gripper → grasp). "
        "ros2: generic ROS2LeaderArm; use --gripper-joint for hardware grasp.",
    )
    parser.add_argument("--topic", type=str, default="/joint_states", help="sensor_msgs/JointState topic")
    parser.add_argument("--period", type=float, default=0.5, help="Seconds between diagnostic prints")
    parser.add_argument("--environment", type=str, default="Lift")
    #####
    parser.add_argument("--robot", type=str, default="VX300S") # UR5e, ViperXAI
    #####
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Headless (no on-screen window). Still steps the simulation.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        choices=("mjviewer", "mujoco"),
        help="On-screen backend (mjviewer = native MuJoCo; mujoco = OpenCV window).",
    )
    parser.add_argument(
        "--render-camera",
        type=str,
        default="frontview",
        help="Camera name for the viewer (Lift default scene camera is often frontview or agentview).",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
        help="Simulation stepping rate (Hz) while teleoperating.",
    )
    parser.add_argument(
        "--refactor-arm-names",
        nargs="+",
        default=["right"],
        help="Arm keys when building the composite JOINT_POSITION config (single-arm: right).",
    )
    parser.add_argument(
        "--arm-input-type",
        choices=("absolute", "delta"),
        default="absolute",
        help="JOINT_POSITION mode: absolute = leader angles (rad) as goals via {arm}_abs; "
        "delta = per-step {arm}_delta (scaled by LeaderArm joint_sensitivity, default 1.0).",
    )
    parser.add_argument(
        "--leaderarm-joint-angle-scale",
        type=float,
        default=1.2,
        metavar="MULT",
        help="Scalar multiplier applied uniformly to all 6 arm joints from JointState: "
        "q_out[i] = MULT * q_in[i] for every i (0 rad stays 0). Example: MULT=2.0 "
        "maps 15 rad → 30 rad on waist, shoulder, elbow, etc. Omit for no scaling (1.0). "
        "Applied before --leaderarm-teleop-scale-shoulder/elbow and --leaderarm-teleop-scale-pivot. "
        "Ignored if --leaderarm-joint-angle-scale-joints is set.",
    )
    parser.add_argument(
        "--leaderarm-joint-angle-scale-joints",
        type=float,
        nargs=6,
        metavar=("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        default=None,
        help="Six separate multipliers (one float per joint, same order as metvars): "
        "q_out[i] = scale[i] * q_in[i]. Use this instead of --leaderarm-joint-angle-scale when "
        "joints need different gains — e.g. '1 2 2 1 1 1' doubles shoulder and elbow "
        "only (15 rad → 30 rad there, unchanged elsewhere). Overrides --leaderarm-joint-angle-scale.",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.10,
        help="Joint position below this (rad) counts as closed (trossen: gripper; ros2: if --gripper-joint set).",
    )
    parser.add_argument(
        "--gripper-joint",
        type=str,
        default=None,
        help="(ros2 only) Joint name for grasp; omit to use keyboard spacebar toggle for grasp.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-shoulder",
        type=float,
        default=None,
        help="Gain on shoulder (arm joint index 1) before sim clamp: q_sim = pivot + scale*(q_leader-pivot). "
        "Default: 1.12 if --robot VX300S, else 1.0.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-elbow",
        type=float,
        default=None,
        help="Gain on elbow (arm joint index 2). Default: same rule as --leaderarm-teleop-scale-shoulder.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-pivot",
        type=float,
        nargs=6,
        metavar=("w", "sh", "el", "fr", "wa", "wr"),
        default=None,
        help="Optional pivot (rad) for all6 arm joints; default pivot0. "
        "Use e.g. hardware home pose so scaling expands motion about that configuration.",
    )
    args = parser.parse_args()

    if args.leaderarm_teleop_scale_shoulder is None:
        args.leaderarm_teleop_scale_shoulder = 1.12 if args.robot == "VX300S" else 1.0
    if args.leaderarm_teleop_scale_elbow is None:
        args.leaderarm_teleop_scale_elbow = 1.12 if args.robot == "VX300S" else 1.0

    import robosuite as suite
    from robosuite.controllers import load_part_controller_config
    from robosuite.controllers.composite.composite_controller import WholeBody
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper

    arm_part_config = load_part_controller_config(default_controller="JOINT_POSITION")
    controller_config = refactor_composite_controller_config(
        arm_part_config,
        args.robot,
        args.refactor_arm_names,
    )
    # JOINT_POSITION defaults to input_type "delta" in the part config; teleop matches --arm-input-type.
    for arm in args.refactor_arm_names:
        arm_cfg = controller_config.get("body_parts", {}).get(arm)
        if isinstance(arm_cfg, dict) and arm_cfg.get("type") == "JOINT_POSITION":
            arm_cfg["input_type"] = args.arm_input_type

    # Match Wipe-style table: closer in +x and higher top (see manipulation/wipe.py DEFAULT_WIPE_CONFIG).
    _lift_table_full_size = (0.5, 0.8, 0.05)
    _lift_table_offset = (0.03, 0.0, 1.0)
    make_kwargs = dict(
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
    if args.environment == "Lift":
        make_kwargs["table_full_size"] = _lift_table_full_size
        make_kwargs["table_offset"] = _lift_table_offset

    env = suite.make(args.environment, **make_kwargs)
    # Match demo_device_control: show grip_site / grip_site_cylinder EE aids each step.
    env = VisualizationWrapper(env, indicator_configs=None)
    env.reset()
    if not args.no_render:
        env.render()

    teleop_scale = np.array(
        [
            1.0,
            args.leaderarm_teleop_scale_shoulder,
            args.leaderarm_teleop_scale_elbow,
            1.0,
            1.0,
            1.0,
        ],
        dtype=float,
    )
    teleop_kw: Dict = {
        "leader_joint_scale": teleop_scale,
    }
    if args.leaderarm_teleop_scale_pivot is not None:
        teleop_kw["leader_joint_scale_pivot"] = tuple(args.leaderarm_teleop_scale_pivot)
    if args.leaderarm_joint_angle_scale_joints is not None:
        teleop_kw["leader_joint_angle_scale"] = tuple(args.leaderarm_joint_angle_scale_joints)
    elif args.leaderarm_joint_angle_scale is not None:
        teleop_kw["leader_joint_angle_scale"] = args.leaderarm_joint_angle_scale

    if args.impl == "trossen":
        device: LeaderArm = TrossenArmLeaderArm(
            env,
            topic=args.topic,
            gripper_close_threshold=args.gripper_close_threshold,
            **teleop_kw,
        )
    else:
        ros2_kw: Dict = dict(env=env, topic=args.topic, **teleop_kw)
        if args.gripper_joint:
            ros2_kw["gripper_joint"] = args.gripper_joint
            ros2_kw["gripper_close_threshold"] = args.gripper_close_threshold
        device = ROS2LeaderArm(**ros2_kw)

    device.start_control()
    names: Optional[List[str]] = getattr(device, "joint_names", None)

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

    _angle_scale_msg = (
        list(args.leaderarm_joint_angle_scale_joints)
        if args.leaderarm_joint_angle_scale_joints is not None
        else args.leaderarm_joint_angle_scale
    )
    print(
        f"Listening on {args.topic!r} ({args.impl}); "
        f"arm_input_type={args.arm_input_type!r}; "
        f"leaderarm_joint_angle_scale={_angle_scale_msg}; "
        f"leaderarm_teleop_scale shoulder={args.leaderarm_teleop_scale_shoulder} "
        f"elbow={args.leaderarm_teleop_scale_elbow}; "
        f"render={'off' if args.no_render else args.renderer}; "
        f"prints every {args.period} s. Press q to reset pose. Ctrl+C to exit.\n"
    )

    dt = 1.0 / float(args.control_freq)
    last_print = time.monotonic()

    try:
        while True:
            loop_t0 = time.monotonic()
            input_ac_dict = device.input2action()
            if input_ac_dict is None:
                env.reset()
                device.start_control()
                all_prev_gripper_actions = _prev_gripper_actions()
                if not args.no_render:
                    env.render()
                continue

            active_robot_model = env.robots[device.active_robot]
            action_dict = deepcopy(input_ac_dict)
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

            now = time.monotonic()
            if now - last_print >= args.period:
                state = device.get_controller_state()
                jp = state["joint_positions"]
                line_parts = []
                if names is not None and len(names) == len(jp):
                    line_parts.append(
                        "joints: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, jp))
                    )
                else:
                    line_parts.append("joints: " + np.array2string(jp, precision=4, separator=", "))
                line_parts.append(f"grasp (closed)={state['grasp']}")
                line_parts.append(f"reset_flag={state['reset']}")
                print(time.strftime("%H:%M:%S"), "|", " | ".join(line_parts), flush=True)
                last_print = now

            elapsed = time.monotonic() - loop_t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        closer = getattr(device, "close", None)
        if closer is not None:
            closer()
        env.close()


if __name__ == "__main__":
    main()

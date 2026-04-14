"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import inspect
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def _device_input2action(device, goal_update_mode):
    """Call device.input2action; LeaderArm omits goal_update_mode (joint-space only)."""
    if "goal_update_mode" in inspect.signature(device.input2action).parameters:
        return device.input2action(goal_update_mode=goal_update_mode)
    return device.input2action()


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = _device_input2action(device, goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # Do not call env.close() here: this function is invoked in a loop; closing would
    # destroy the MuJoCo sim and viewer and break the next episode (and ROS leader arms).


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        help="keyboard | spacemouse | dualsense | mjgui | trossen_leaderarm | ros2_leaderarm",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    parser.add_argument(
        "--table-offset",
        type=float,
        nargs=3,
        metavar=("x", "y", "z"),
        default=None,
        help="(x, y, z) offset for TableArena placement; z sets the tabletop height. "
        "Defaults to (0.03, 0.0, 1.0) for VX300S (matches arm reach), "
        "or the environment default (0.15, 0.0, 0.9) for other robots. "
        "Applies to environments that accept table_offset (e.g. Lift, Stack).",
    )
    parser.add_argument(
        "--table-full-size",
        type=float,
        nargs=3,
        metavar=("x", "y", "z"),
        default=None,
        help="(x, y, z) dimensions of the table surface. "
        "Defaults to (0.5, 0.8, 0.05) for VX300S, or the environment default otherwise.",
    )
    parser.add_argument(
        "--leaderarm-topic",
        type=str,
        default="/joint_states",
        help="ROS 2 JointState topic for trossen_leaderarm / ros2_leaderarm.",
    )
    parser.add_argument(
        "--leaderarm-node-name",
        type=str,
        default=None,
        help="ROS 2 node name (default: implementation-specific).",
    )
    parser.add_argument(
        "--leaderarm-joint-names",
        nargs="*",
        default=None,
        help="Ordered joint names for ros2_leaderarm (default: ROS2LeaderArm built-in list). Ignored for trossen_leaderarm.",
    )
    parser.add_argument(
        "--leaderarm-gripper-joint",
        type=str,
        default=None,
        help="Optional joint name on JointState to infer grasp (ros2_leaderarm). Uses keyboard gripper toggle when omitted.",
    )
    parser.add_argument(
        "--leaderarm-gripper-close-threshold",
        type=float,
        default=0.0,
        help="Grasp closed when gripper joint position is below this (rad), if --leaderarm-gripper-joint is set.",
    )
    parser.add_argument(
        "--leaderarm-joint-sensitivity",
        type=float,
        default=1.0,
        help="Scales joint deltas from the leader arm (trossen_leaderarm / ros2_leaderarm).",
    )
    parser.add_argument(
        "--leaderarm-joint-limits-safety-factor",
        type=float,
        default=0.95,
        help="Clamp leader targets within this fraction of each joint range (0–1).",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-shoulder",
        type=float,
        default=None,
        help="Per-joint gain for shoulder (joint index 1) applied before sim-limit clamping. "
        "Defaults to 1.12 when --robots contains VX300S, else 1.0.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-elbow",
        type=float,
        default=None,
        help="Per-joint gain for elbow (joint index 2) applied before sim-limit clamping. "
        "Defaults to 1.12 when --robots contains VX300S, else 1.0.",
    )
    parser.add_argument(
        "--leaderarm-teleop-scale-pivot",
        type=float,
        nargs=6,
        metavar=("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        default=None,
        help="Optional 6-element pivot (rad) for the per-joint affine scaling map "
        "(q_sim = pivot + scale * (q_leader - pivot)). "
        "Set to the hardware home pose to expand motion relative to that configuration.",
    )
    args = parser.parse_args()

    _is_leaderarm_device = args.device in ("trossen_leaderarm", "ros2_leaderarm")

    # VX300S shoulder/elbow joints have a different mechanical range than the
    # hardware encoder range; a gain of 1.12 compensates for this by default.
    _primary_robot = args.robots[0] if isinstance(args.robots, list) else args.robots
    _is_vx300s = _primary_robot.upper() in ("VX300S",)
    _default_scale = 1.12 if _is_vx300s else 1.0

    # Table geometry defaults: VX300S arm reach is best matched with a higher
    # table (z=1.0) and slight forward shift in x, matching leaderarm.py main().
    if args.table_offset is None:
        args.table_offset = (0.03, 0.0, 1.0) if _is_vx300s else (0.15, 0.0, 0.9)
    if args.table_full_size is None:
        args.table_full_size = (0.5, 0.8, 0.05)
    if args.leaderarm_teleop_scale_shoulder is None:
        args.leaderarm_teleop_scale_shoulder = _default_scale
    if args.leaderarm_teleop_scale_elbow is None:
        args.leaderarm_teleop_scale_elbow = _default_scale

    # Get controller config.
    # LeaderArm requires JOINT_POSITION (a part controller), which cannot be
    # passed directly to load_composite_controller_config.  Instead, build it
    # the same way leaderarm.py does: load the part config then wrap it in a
    # composite config via refactor_composite_controller_config.
    if _is_leaderarm_device and args.controller in (None, "JOINT_POSITION"):
        _arm_part_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
        _arm_part_cfg["input_type"] = "absolute"  # LeaderArm sends absolute joint targets
        controller_config = refactor_composite_controller_config(
            _arm_part_cfg,
            _primary_robot,
            ["right"],  # VX300S (and most single-arm robots) use "right"
        )
    else:
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=_primary_robot,
        )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        table_full_size=tuple(args.table_full_size),
        table_offset=tuple(args.table_offset),
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Match robosuite.devices.leaderarm main(): run one reset before starting the ROS
    # subscriber so JointState callbacks attach to the same sim/robot layout as teleop.
    if args.device in ("trossen_leaderarm", "ros2_leaderarm"):
        env.reset()
        env.render()

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    ##
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    elif args.device in ("trossen_leaderarm", "ros2_leaderarm"):
        # Build the per-joint scale array [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate].
        _teleop_scale = np.array(
            [1.0, args.leaderarm_teleop_scale_shoulder, args.leaderarm_teleop_scale_elbow, 1.0, 1.0, 1.0],
            dtype=float,
        )
        _leaderarm_common_kw = dict(
            env=env,
            topic=args.leaderarm_topic,
            joint_sensitivity=args.leaderarm_joint_sensitivity,
            joint_limits_safety_factor=args.leaderarm_joint_limits_safety_factor,
            gripper_close_threshold=args.leaderarm_gripper_close_threshold,
            leader_joint_scale=_teleop_scale,
        )
        if args.leaderarm_teleop_scale_pivot is not None:
            _leaderarm_common_kw["leader_joint_scale_pivot"] = tuple(args.leaderarm_teleop_scale_pivot)
        if args.leaderarm_node_name is not None:
            _leaderarm_common_kw["node_name"] = args.leaderarm_node_name

        if args.device == "trossen_leaderarm":
            try:
                from robosuite.devices import TrossenArmLeaderArm
            except ImportError as exc:
                raise ImportError(
                    "trossen_leaderarm requires ROS 2 packages (rclpy, sensor_msgs). "
                    "Source your ROS 2 workspace and install dependencies."
                ) from exc
            device = TrossenArmLeaderArm(**_leaderarm_common_kw)
        else:  # ros2_leaderarm
            try:
                from robosuite.devices import ROS2LeaderArm
            except ImportError as exc:
                raise ImportError(
                    "ros2_leaderarm requires ROS 2 packages (rclpy, sensor_msgs). "
                    "Source your ROS 2 workspace and install dependencies."
                ) from exc
            _ros2_kw = dict(**_leaderarm_common_kw, gripper_joint=args.leaderarm_gripper_joint)
            if args.leaderarm_joint_names is not None and len(args.leaderarm_joint_names) > 0:
                _ros2_kw["joint_names"] = list(args.leaderarm_joint_names)
            device = ROS2LeaderArm(**_ros2_kw)
    else:
        raise Exception(
            "Invalid device. Choose keyboard, spacemouse, dualsense, mjgui, "
            "trossen_leaderarm, or ros2_leaderarm. "
            f"Got: {args.device!r}"
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    try:
        while True:
            collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
    finally:
        device_closer = getattr(device, "close", None)
        if device_closer is not None:
            device_closer()
        env.close()

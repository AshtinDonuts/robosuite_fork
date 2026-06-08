"""
Collect human demonstrations and save end-effector state trajectories as pickle files.

This script records trajectories in the same pickle format as
`.../kinova_6feb_pick/ee_state_0.pk` used by pumafabrics:

    {
        "x_pos":         [np.ndarray shape (3,), ...],        # world-frame EE position (m)
        "x_rot":         [np.ndarray shape (3, 3), ...],      # world-frame EE rotation matrix
        "x_dot":         [np.ndarray shape (6,), ...],        # [linear_vel(3), angular_vel(3)] in world frame
        "x_stiffness":   [np.ndarray shape (6, 6), ...],      # Cartesian stiffness matrix, diag [xyz, rpy]
        "x_damping":     [np.ndarray shape (6, 6), ...],      # Cartesian damping matrix, diag [xyz, rpy]
        "delta_t":       np.ndarray shape (T,),               # wall-clock dt between samples (s)
        "gripper_action": [np.ndarray shape (dof,), ...],     # per-step gripper command (+1=close, -1=open)
    }

Notes:
    - The EE pose / twist are taken from MuJoCo site quantities for the robot gripper
      "grip_site" (see robot.eef_site_id).
    - `delta_t` is measured using wall-clock time to match the reference dataset.
"""

import argparse
import datetime
import inspect
import json
import os
import pickle
import threading
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


def _hdf5_attr_str(value):
    """
    HDF5 attributes must use native Python scalars/strings. NumPy scalar/ndarray
    values (including dtype=object) otherwise raise TypeError in h5py.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        el = value.reshape(-1)[0]
        if value.dtype == object:
            return _hdf5_attr_str(el.item())
        return _hdf5_attr_str(el.item())
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _get_eef_state(env, robot_index: int, arm: str):
    """
    Returns (x_pos, x_rot, x_dot) using MuJoCo site quantities for the EE grip site.
    """
    robot = env.robots[robot_index]
    site_id = robot.eef_site_id[arm]

    # site_xpos: (3,), site_xmat: (9,) row-major, site linear/angular velocity: (3,)
    x_pos = env.sim.data.site_xpos[site_id].copy().astype(np.float64, copy=False)
    x_rot = env.sim.data.site_xmat[site_id].reshape(3, 3).copy().astype(np.float64, copy=False)
    # MuJoCo >= 3 exposes site velocities via accessors, not data.site_xvelp/site_xvelr
    try:
        site_name = robot.gripper[arm].important_sites["grip_site"]
    except Exception:
        # Fallback if gripper / important_sites isn't available
        site_name = env.sim.model.site_id2name(site_id)
    x_vel_lin = np.asarray(env.sim.data.get_site_xvelp(site_name), dtype=np.float64)
    x_vel_ang = np.asarray(env.sim.data.get_site_xvelr(site_name), dtype=np.float64)
    x_dot = np.concatenate([x_vel_lin, x_vel_ang]).astype(np.float64, copy=False)
    return x_pos, x_rot, x_dot


def _get_cartesian_impedance(env, robot_index: int, arm: str):
    """
    Returns Cartesian stiffness / damping matrices for OSC-controlled arms.

    robosuite OSC stores diagonal task-space gains as kp / kd vectors ordered
    [x, y, z, rx, ry, rz]. Non-OSC controllers do not expose Cartesian task-space
    impedance, so return NaN matrices to keep the puma_dataset schema stable.
    """
    nan_matrix = np.full((6, 6), np.nan, dtype=np.float64)
    try:
        controller = env.robots[robot_index].part_controllers[arm]
    except Exception:
        return nan_matrix.copy(), nan_matrix.copy()

    controller_name = str(getattr(controller, "name", ""))
    if controller.__class__.__name__ != "OperationalSpaceController" and not controller_name.startswith("OSC_"):
        return nan_matrix.copy(), nan_matrix.copy()

    try:
        kp = np.asarray(controller.kp, dtype=np.float64).reshape(-1)
        kd = np.asarray(controller.kd, dtype=np.float64).reshape(-1)
    except Exception:
        return nan_matrix.copy(), nan_matrix.copy()

    if kp.size < 6 or kd.size < 6:
        return nan_matrix.copy(), nan_matrix.copy()
    return np.diag(kp[:6]).astype(np.float64, copy=False), np.diag(kd[:6]).astype(np.float64, copy=False)


def _device_input2action(device, goal_update_mode):
    """Call device.input2action; LeaderArm omits goal_update_mode (joint-space only)."""
    if "goal_update_mode" in inspect.signature(device.input2action).parameters:
        return device.input2action(goal_update_mode=goal_update_mode)
    return device.input2action()


def _prime_data_collection_from_current_state(env):
    """Make delayed recording start from the current simulator state."""
    if not hasattr(env, "_current_task_instance_state"):
        return
    env.states = []
    env.action_infos = []
    env.successful = False
    env.has_interaction = False
    env.t = 0
    if getattr(env, "use_env_xml_for_reset", False):
        env._current_task_instance_xml = env.env.model.get_xml()
    else:
        env._current_task_instance_xml = env.env.sim.model.get_xml()
    env._current_task_instance_state = np.array(env.env.sim.get_state().flatten())
    print("Data collection primed at current state; subsequent steps will be recorded.", flush=True)


def _flush_data_collection(env):
    """Flush the current DataCollectionWrapper episode before HDF5 consolidation."""
    if not hasattr(env, "_flush") or not getattr(env, "has_interaction", False):
        return
    env._flush()
    env.has_interaction = False


class _RecordingHotkeys:
    """Tracks recording hotkeys and prints enough diagnostics to debug focus issues."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ctrl_down = False
        self._recording = False
        self._stop_requested = False
        self._listener = None
        self._key_cls = None

    def start(self):
        from pynput.keyboard import Key, Listener

        self._key_cls = Key
        print(
            "Recording hotkeys enabled: press Ctrl+D to START recording, "
            "Ctrl+F to STOP and save. If Ctrl combos do not register, press plain 'd' / 'f'.",
            flush=True,
        )
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self._listener.wait()
        print("Recording hotkey listener is running. Focus the sim/viewer or terminal before pressing hotkeys.", flush=True)

    def close(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            print("Recording hotkey listener stopped.", flush=True)

    def reset_episode(self):
        with self._lock:
            self._recording = False
            self._stop_requested = False
            self._ctrl_down = False
        print("Recording state reset: waiting for Ctrl+D (or plain 'd') to start.", flush=True)

    @property
    def recording(self):
        with self._lock:
            return self._recording

    @property
    def stop_requested(self):
        with self._lock:
            return self._stop_requested

    def _describe_key(self, key):
        char = getattr(key, "char", None)
        if char is None:
            return repr(key)
        return f"char={char!r}, key={key!r}"

    def _set_recording_started(self, source):
        if self._recording:
            print(f"Recording already active; ignored {source}.", flush=True)
            return
        self._recording = True
        self._stop_requested = False
        print(f"Recording started ({source}).", flush=True)

    def _set_recording_stopped(self, source):
        if not self._recording:
            print(f"Recording is not active; ignored {source}.", flush=True)
            return
        self._recording = False
        self._stop_requested = True
        print(f"Recording stopped ({source}); ending rollout and saving trajectory.", flush=True)

    def _on_press(self, key):
        print(f"Recording hotkey press seen: {self._describe_key(key)}", flush=True)

        if key in (self._key_cls.ctrl, self._key_cls.ctrl_l, self._key_cls.ctrl_r):
            with self._lock:
                self._ctrl_down = True
            print("Ctrl is down. Press d to start, or f to stop.", flush=True)
            return

        char = getattr(key, "char", None)
        if char is None:
            return

        char = char.lower()
        with self._lock:
            ctrl_combo = self._ctrl_down or char in ("\x04", "\x06")
            if char in ("d", "\x04"):
                self._set_recording_started("Ctrl+D" if ctrl_combo else "fallback d")
            elif char in ("f", "\x06"):
                self._set_recording_stopped("Ctrl+F" if ctrl_combo else "fallback f")

    def _on_release(self, key):
        print(f"Recording hotkey release seen: {self._describe_key(key)}", flush=True)
        if key in (self._key_cls.ctrl, self._key_cls.ctrl_l, self._key_cls.ctrl_r):
            with self._lock:
                self._ctrl_down = False
            print("Ctrl released.", flush=True)


def _run_leaderarm_eef_anchor_delay_countdown(env, delay_sec: float) -> None:
    """
    Pause before `LeaderArm.start_control()` captures leader/follower anchor poses.

    Mirrors ``robosuite.devices.leaderarm_eef`` ``__main__`` so the operator can
    hold the physical leader in a neutral pose while the sim viewer stays live.
    """
    if delay_sec <= 0:
        return
    print("Simulation started. Hold the leader arm in the desired neutral pose.")
    print("Anchor pose capture countdown:")
    deadline = time.time() + delay_sec
    last_printed = delay_sec
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        count = int(remaining) + 1
        if count != last_printed:
            print(f"  {count}...")
            last_printed = count
        viewer = getattr(env, "viewer", None)
        if viewer is not None:
            viewer.update()
        env.render()
    print("Capturing anchor pose now.", flush=True)


def _show_aux_camera_window(env, camera_name, width, height, window_name):
    """Render one offscreen camera into a separate OpenCV window."""
    if camera_name is None:
        return

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("--aux-camera-window requires opencv-python / cv2 to be installed.") from exc

    frame = env.sim.render(height=int(height), width=int(width), camera_name=camera_name)
    frame = frame[::-1, :, ::-1]
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def collect_human_trajectory(
    env,
    device,
    arm,
    max_fr,
    goal_update_mode,
    robot_index: int = 0,
    puma_dataset: bool = False,
    leaderarm_eef_anchor_delay_sec: float = 0.0,
    aux_camera_name: str = None,
    aux_camera_width: int = 320,
    aux_camera_height: int = 240,
    aux_camera_window_name: str = "aux camera",
    recording_hotkeys=None,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    If puma_dataset is True, returns a trajectory dict in the target pickle format.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
        robot_index (int): robot index for EE state logging when ``puma_dataset`` is True.
        puma_dataset (bool): if True, return trajectory dict for pickle export.
        leaderarm_eef_anchor_delay_sec (float): When > 0 and the device has EEF anchoring
            (``trossen_leaderarm_eef``), wait this many seconds after ``env.reset()`` before
            ``start_control()`` captures anchors (same idea as ``leaderarm_eef.py`` ``__main__``).
            Use ``0`` to skip.
    """

    env.reset()
    env.render()
    _show_aux_camera_window(
        env,
        aux_camera_name,
        aux_camera_width,
        aux_camera_height,
        aux_camera_window_name,
    )

    if leaderarm_eef_anchor_delay_sec > 0 and hasattr(device, "_leader_anchor_pose"):
        _run_leaderarm_eef_anchor_delay_countdown(env, leaderarm_eef_anchor_delay_sec)

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

    x_pos, x_rot, x_dot, x_stiffness, x_damping, delta_t, gripper_action = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    prev_t = None
    warned_missing_cartesian_impedance = False
    recording_started = recording_hotkeys is None
    if recording_hotkeys is not None:
        recording_hotkeys.reset_episode()
        print("Move to the desired start pose, then press Ctrl+D to start recording; press Ctrl+F to stop.", flush=True)
    if puma_dataset:
        x_pos, x_rot, x_dot, x_stiffness, x_damping, delta_t, gripper_action = [], [], [], [], [], [], []
        if recording_hotkeys is None:
            prev_t = time.time()

    # Loop until we get a reset from the input, a stop hotkey, or the task completes
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

        hotkey_recording = recording_hotkeys is None or recording_hotkeys.recording
        if hotkey_recording and not recording_started:
            _prime_data_collection_from_current_state(env)
            recording_started = True

        if hotkey_recording:
            env.step(env_action)
        else:
            # Let the operator move to the desired start pose without logging HDF5 states/actions.
            env.env.step(env_action)

        env.render()
        _show_aux_camera_window(
            env,
            aux_camera_name,
            aux_camera_width,
            aux_camera_height,
            aux_camera_window_name,
        )

        if puma_dataset:
            if hotkey_recording:
                # Record EE state after the step (achieved state)
                now_t = time.time()
                if prev_t is None:
                    dt = 0.0
                else:
                    dt = now_t - prev_t
                prev_t = now_t

                xp, xr, xd = _get_eef_state(env, robot_index=robot_index, arm=arm)
                xs, xdamp = _get_cartesian_impedance(env, robot_index=robot_index, arm=arm)
                if np.isnan(xs).all() and not warned_missing_cartesian_impedance:
                    print(
                        "Warning: Cartesian stiffness/damping unavailable for this controller; "
                        "recording NaN matrices in x_stiffness / x_damping.",
                        flush=True,
                    )
                    warned_missing_cartesian_impedance = True
                x_pos.append(xp)
                x_rot.append(xr)
                x_dot.append(xd)
                x_stiffness.append(xs)
                x_damping.append(xdamp)
                delta_t.append(float(dt))
                # Record the gripper command that was just sent (+1=close, -1=open).
                # action_dict[f"{arm}_gripper"] is the latched position command maintained
                # by all_prev_gripper_actions, so it reflects the true commanded state.
                gripper_key = f"{arm}_gripper"
                ga = action_dict.get(gripper_key, np.zeros(env.robots[robot_index].gripper[arm].dof))
                gripper_action.append(np.asarray(ga, dtype=np.float64).copy())

                ## Debug
                # print(f"{xp=}\n")
                # print(f"{xr=}")
                # print(f"{xd=}")

        if recording_hotkeys is not None and recording_hotkeys.stop_requested:
            break

        if recording_started:
            # Also break if we complete the task after recording has started.
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
        else:
            task_completion_hold_count = -1

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # Do not call env.close() here: this function is invoked in a loop; closing would
    # destroy the MuJoCo sim and viewer and break the next episode (and ROS leader arms).
    _flush_data_collection(env)
    if not puma_dataset:
        return None
    return {
        "x_pos": x_pos,
        "x_rot": x_rot,
        "x_dot": x_dot,
        "x_stiffness": x_stiffness,
        "x_damping": x_damping,
        "delta_t": np.asarray(delta_t, dtype=np.float64),
        "gripper_action": gripper_action,  # list of np.ndarray (dof,) per timestep
    }


def _save_ee_state_pickle(traj: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a single hdf5 file.

    The structure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        env_info (attribute) - JSON config used to create the env

        demo_1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
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
            env_name = _hdf5_attr_str(dic["env"])

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
            with open(xml_path, "r") as f_xml:
                xml_str = f_xml.read()
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
    grp.attrs["repository_version"] = _hdf5_attr_str(suite.__version__)
    grp.attrs["env"] = _hdf5_attr_str(env_name)
    grp.attrs["env_info"] = _hdf5_attr_str(env_info)

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deterministic_reset",
        action="store_true",
        help="If set, keep the dirt/object layout fixed across episodes (no re-randomisation on reset).",
    )
    parser.add_argument(
        "--save-dirt-layout",
        action="store_true",
        help="(Wipe only) Save the sampled dirt marker XY layout to dirt_layout.json in the output directory.",
    )
    parser.add_argument(
        "--dirt-layout",
        type=str,
        default=None,
        help="(Wipe only) Path to a dirt_layout.json file. If set, forces that exact dirt marker layout on every reset "
        "(independent of --deterministic_reset).",
    )
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
        "--aux-camera-window",
        type=str,
        default=None,
        help="Optional camera name to show in a separate OpenCV window using offscreen rendering, "
        "e.g. robot0_eye_in_hand. This can be used with --renderer mjviewer and --camera agentview.",
    )
    parser.add_argument(
        "--aux-camera-width",
        type=int,
        default=320,
        help="Width of the optional --aux-camera-window view.",
    )
    parser.add_argument(
        "--aux-camera-height",
        type=int,
        default=240,
        help="Height of the optional --aux-camera-window view.",
    )
    parser.add_argument(
        "--visualize-keypoints",
        action="store_true",
        help="Show keypoint marker sites for environments that expose visualize_keypoints.",
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
        help="keyboard | spacemouse | dualsense | mjgui | trossen_leaderarm | ros2_leaderarm | trossen_leaderarm_eef",
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
        "--eps_name",
        type=str,
        required=True,
        help="Name of the episode folder.",
    )
    parser.add_argument(
        "--puma_dataset",
        action="store_true",
        help="If set, save trajectories as ee_state_*.pk in the pumafabrics-compatible format. "
        "If not set, use the original robosuite HDF5 (demo.hdf5) collection pipeline.",
    )
    parser.add_argument(
        "--out-format",
        type=str,
        default="pkl",
        choices=["pkl"],
        help="Output format. Currently only 'pkl' is supported (ee_state_*.pk).",
    )
    parser.add_argument(
        "--save-only-successful",
        action="store_true",
        help="If set, only saves trajectories that end in task success.",
    )
    parser.add_argument(
        "--robot-index",
        type=int,
        default=0,
        help="Which robot index to record EE state from (default: 0).",
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
        "or (0.15, 0.0, 0.9) for other robots on Lift-style tasks. "
        "Omitted for TableShelfPick / TableCubePick so shelf/cube/object layout matches demo_device_control.",
    )
    parser.add_argument(
        "--table-full-size",
        type=float,
        nargs=3,
        metavar=("x", "y", "z"),
        default=None,
        help="(x, y, z) dimensions of the table surface. "
        "Defaults to (0.5, 0.8, 0.05) for VX300S on Lift-style tasks. "
        "Omitted for TableShelfPick / TableCubePick so layout matches demo_device_control.",
    )
    parser.add_argument(
        "--leaderarm-topic",
        type=str,
        default=None,
        help="ROS 2 JointState topic for trossen_leaderarm / ros2_leaderarm. "
        "Defaults to the device implementation default.",
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
        default=None,
        help="Grasp closed when gripper joint position is below this (rad), if --leaderarm-gripper-joint is set. "
        "Defaults to the device implementation default.",
    )
    parser.add_argument(
        "--leaderarm-joint-sensitivity",
        type=float,
        default=None,
        help="Scales joint deltas from the leader arm (trossen_leaderarm / ros2_leaderarm). "
        "Defaults to the device implementation default.",
    )
    parser.add_argument(
        "--leaderarm-joint-limits-safety-factor",
        type=float,
        default=None,
        help="Clamp leader targets within this fraction of each joint range (0–1). "
        "Defaults to the device implementation default.",
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
    parser.add_argument(
        "--leaderarm-joint-angle-scale",
        type=float,
        default=None,
        metavar="MULT",
        help="Scalar multiplier (MULT) for trossen_leaderarm / ros2_leaderarm: the same "
        "MULT is broadcast to all arm joints, q_out[i] = MULT * q_in[i] (0 rad stays 0). "
        "Example: MULT=2.0 maps 15 rad → 30 rad on every joint. Applied before "
        "--leaderarm-teleop-scale-shoulder/elbow. Ignored if "
        "--leaderarm-joint-angle-scale-joints is set.",
    )
    parser.add_argument(
        "--leaderarm-joint-angle-scale-joints",
        type=float,
        nargs=6,
        metavar=("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        default=None,
        help="Six per-joint multipliers (tuple order: waist, shoulder, elbow, "
        "forearm_roll, wrist_angle, wrist_rotate): q_out[i] = scale[i] * q_in[i]. "
        "Use when joints need different gains — e.g. '1 2 2 1 1 1' doubles only "
        "shoulder and elbow. Overrides --leaderarm-joint-angle-scale.",
    )
    # ── trossen_leaderarm_eef-specific arguments ──────────────────────────────
    parser.add_argument(
        "--leaderarm-eef-position-scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to leader EEF translation (trossen_leaderarm_eef).",
    )
    parser.add_argument(
        "--leaderarm-eef-position-scale-x",
        type=float,
        default=None,
        help="Per-axis x override for leader EEF translation scale (trossen_leaderarm_eef).",
    )
    parser.add_argument(
        "--leaderarm-eef-position-scale-y",
        type=float,
        default=None,
        help="Per-axis y override for leader EEF translation scale (trossen_leaderarm_eef).",
    )
    parser.add_argument(
        "--leaderarm-eef-position-scale-z",
        type=float,
        default=None,
        help="Per-axis z override for leader EEF translation scale (trossen_leaderarm_eef).",
    )
    parser.add_argument(
        "--leaderarm-eef-position-offset",
        type=float,
        nargs=3,
        metavar=("x", "y", "z"),
        default=[0.0, 0.0, 0.0],
        help="Additive (x, y, z) offset (m) in follower base frame after scaled leader "
        "translation (trossen_leaderarm_eef). Default: 0 0 0.",
    )
    parser.add_argument(
        "--leaderarm-eef-orientation-scale",
        type=float,
        default=1.0,
        help="Scale applied to leader EEF rotation (trossen_leaderarm_eef).",
    )
    parser.add_argument(
        "--leaderarm-eef-rotation-offset-rpy",
        type=float,
        nargs=3,
        metavar=("roll", "pitch", "yaw"),
        default=[0.0, 0.0, 0.0],
        help="Constant (roll, pitch, yaw) offset (rad) added to target orientation in "
        "follower base frame (trossen_leaderarm_eef). Default: 0 0 0.",
    )
    parser.add_argument(
        "--leaderarm-eef-calibration-step",
        type=float,
        default=0.01,
        help="Position step size (m) per arrow-key press for online calibration "
        "(trossen_leaderarm_eef). Default: 0.01 m.",
    )
    parser.add_argument(
        "--leaderarm-eef-calibration-step-rot",
        type=float,
        default=0.05,
        help="Rotation step size (rad) per o/p key press for online pitch calibration "
        "(trossen_leaderarm_eef). Default: ~3°.",
    )
    parser.add_argument(
        "--leaderarm-eef-anchor-delay-sec",
        type=float,
        default=10.0, #@USER
        help="After each env reset, wait this many seconds before capturing EEF anchor poses "
        "(trossen_leaderarm_eef only; same behavior as robosuite.devices.leaderarm_eef __main__). "
        "Use 0 to disable.",
    )
    args = parser.parse_args()

    _is_leaderarm_device = args.device in ("trossen_leaderarm", "ros2_leaderarm")
    _is_eef_device = args.device == "trossen_leaderarm_eef"

    # VX300S shoulder/elbow joints have a different mechanical range than the
    # hardware encoder range; a gain of 1.12 compensates for this by default.
    _primary_robot = args.robots[0] if isinstance(args.robots, list) else args.robots
    _is_vx300s = _primary_robot.upper() in ("VX300S",)
    _default_scale = 1.12 if _is_vx300s else 1.0

    # demo_device_control does not override table/shelf/cube/object layout for TableShelfPick / TableCubePick;
    # keep env defaults (table_offset, shelf_pos, object_x/y_range, etc.) unless the user
    # passes --table-offset / --table-full-size explicitly.
    _user_table_offset = args.table_offset
    _user_table_full_size = args.table_full_size
    _use_env_builtin_table_layout = args.environment in {"TableShelfPick", "TableCubePick"}

    # Table geometry defaults for Lift-style tasks: VX300S arm reach is best matched
    # with a higher table (z=1.0) and slight forward shift in x (leaderarm.py main()).
    if _user_table_offset is None and not _use_env_builtin_table_layout:
        _user_table_offset = (0.03, 0.0, 1.0) if _is_vx300s else (0.15, 0.0, 0.9)
    if _user_table_full_size is None and not _use_env_builtin_table_layout:
        _user_table_full_size = (0.5, 0.8, 0.05)
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
    elif _is_eef_device and args.controller in (None, "OSC_POSE"):
        # leaderarm_eef maps leader FK → follower EEF pose; requires a pose-based controller.
        _arm_part_cfg = load_part_controller_config(default_controller="OSC_POSE")
        _arm_part_cfg["input_type"] = "absolute"  # absolute EEF targets are recommended
        controller_config = refactor_composite_controller_config(
            _arm_part_cfg,
            _primary_robot,
            ["right"],
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
    _make_kwargs = dict(
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=args.aux_camera_window is not None,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    if args.visualize_keypoints:
        _make_kwargs["visualize_keypoints"] = True
    if _user_table_offset is not None:
        _make_kwargs["table_offset"] = tuple(_user_table_offset)
    if _user_table_full_size is not None:
        _make_kwargs["table_full_size"] = tuple(_user_table_full_size)
    env = suite.make(**config, **_make_kwargs)
    if args.deterministic_reset:
        env.deterministic_reset = True
    # Optional: force a fixed dirt layout (Wipe only) for cross-run reproducibility.
    if args.dirt_layout is not None:
        with open(args.dirt_layout, "r") as f:
            _layout = json.load(f)
        # Accept either {"xy": [...]} or a raw list.
        _xy = _layout.get("xy", _layout) if isinstance(_layout, dict) else _layout
        env.unwrapped._fixed_dirt_xy = [tuple(map(float, xy)) for xy in _xy]

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    tmp_directory = None
    # Wrap the environment with data collection wrapper (records state_*.npz).
    # Even in --puma_dataset mode, we also record using the original robosuite
    # pipeline so users can get demo.hdf5 alongside ee_state_*.pk.
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Match robosuite.devices.leaderarm main(): run one reset before starting the ROS
    # subscriber so JointState callbacks attach to the same sim/robot layout as teleop.
    if args.device in ("trossen_leaderarm", "ros2_leaderarm", "trossen_leaderarm_eef"):
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
            leader_joint_scale=_teleop_scale,
        )
        if args.leaderarm_topic is not None:
            _leaderarm_common_kw["topic"] = args.leaderarm_topic
        if args.leaderarm_joint_sensitivity is not None:
            _leaderarm_common_kw["joint_sensitivity"] = args.leaderarm_joint_sensitivity
        if args.leaderarm_joint_limits_safety_factor is not None:
            _leaderarm_common_kw["joint_limits_safety_factor"] = args.leaderarm_joint_limits_safety_factor
        if args.leaderarm_gripper_close_threshold is not None:
            _leaderarm_common_kw["gripper_close_threshold"] = args.leaderarm_gripper_close_threshold
        if args.leaderarm_teleop_scale_pivot is not None:
            _leaderarm_common_kw["leader_joint_scale_pivot"] = tuple(args.leaderarm_teleop_scale_pivot)
        if args.leaderarm_joint_angle_scale_joints is not None:
            _leaderarm_common_kw["leader_joint_angle_scale"] = tuple(args.leaderarm_joint_angle_scale_joints)
        elif args.leaderarm_joint_angle_scale is not None:
            _leaderarm_common_kw["leader_joint_angle_scale"] = args.leaderarm_joint_angle_scale
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
    elif args.device == "trossen_leaderarm_eef":
        try:
            from robosuite.devices.leaderarm_eef import TrossenArmLeaderArm as TrossenArmLeaderArmEEF
        except ImportError as exc:
            raise ImportError(
                "trossen_leaderarm_eef requires ROS 2 packages (rclpy, sensor_msgs) and pynput. "
                "Source your ROS 2 workspace and install dependencies."
            ) from exc
        _teleop_scale = np.array(
            [1.0, args.leaderarm_teleop_scale_shoulder, args.leaderarm_teleop_scale_elbow, 1.0, 1.0, 1.0],
            dtype=float,
        )
        _position_scale_xyz = np.array(
            [
                args.leaderarm_eef_position_scale if args.leaderarm_eef_position_scale_x is None else args.leaderarm_eef_position_scale_x,
                args.leaderarm_eef_position_scale if args.leaderarm_eef_position_scale_y is None else args.leaderarm_eef_position_scale_y,
                args.leaderarm_eef_position_scale if args.leaderarm_eef_position_scale_z is None else args.leaderarm_eef_position_scale_z,
            ],
            dtype=float,
        )
        _eef_kw = dict(
            env=env,
            position_scale=args.leaderarm_eef_position_scale,
            position_scale_xyz=_position_scale_xyz,
            position_offset_xyz=np.array(args.leaderarm_eef_position_offset, dtype=float),
            orientation_scale=args.leaderarm_eef_orientation_scale,
            rotation_offset_rpy=np.array(args.leaderarm_eef_rotation_offset_rpy, dtype=float),
            calibration_step=args.leaderarm_eef_calibration_step,
            calibration_step_rot=args.leaderarm_eef_calibration_step_rot,
            leader_joint_scale=_teleop_scale,
        )
        if args.leaderarm_topic is not None:
            _eef_kw["topic"] = args.leaderarm_topic
        if args.leaderarm_gripper_close_threshold is not None:
            _eef_kw["gripper_close_threshold"] = args.leaderarm_gripper_close_threshold
        if args.leaderarm_teleop_scale_pivot is not None:
            _eef_kw["leader_joint_scale_pivot"] = tuple(args.leaderarm_teleop_scale_pivot)
        if args.leaderarm_node_name is not None:
            _eef_kw["node_name"] = args.leaderarm_node_name
        device = TrossenArmLeaderArmEEF(**_eef_kw)
    else:
        raise Exception(
            "Invalid device. Choose keyboard, spacemouse, dualsense, mjgui, "
            "trossen_leaderarm, ros2_leaderarm, or trossen_leaderarm_eef. "
            f"Got: {args.device!r}"
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # make a custom named directory
    new_dir = os.path.join(args.directory, args.eps_name)
    os.makedirs(new_dir, exist_ok=True)

    # Optionally snapshot the initial sampled dirt layout for later reproduction (Wipe only).
    if args.save_dirt_layout:
        try:
            arena = env.unwrapped.model.mujoco_arena
            if arena.__class__.__name__ == "WipeArena":
                xy = []
                for marker in arena.markers:
                    body_id = env.unwrapped.sim.model.body_name2id(marker.root_body)
                    pos = env.unwrapped.sim.model.body_pos[body_id]
                    xy.append([float(pos[0]), float(pos[1])])
                with open(os.path.join(new_dir, "dirt_layout.json"), "w") as f:
                    json.dump({"xy": xy}, f, indent=2)
        except Exception:
            # Best-effort: ignore if environment doesn't expose wipe markers.
            pass

    # Save a small metadata file (keeps parity with previous behavior)
    with open(os.path.join(new_dir, "env_info.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    env_info = json.dumps(config)

    recording_hotkeys = _RecordingHotkeys()
    recording_hotkeys.start()

    # collect demonstrations
    saved_idx = 0  # used for ee_state_{i}.pk in puma_dataset mode
    try:
        while True:
            traj = collect_human_trajectory(
                env,
                device,
                args.arm,
                args.max_fr,
                args.goal_update_mode,
                robot_index=args.robot_index,
                puma_dataset=args.puma_dataset,
                leaderarm_eef_anchor_delay_sec=(
                    float(args.leaderarm_eef_anchor_delay_sec)
                    if args.device == "trossen_leaderarm_eef"
                    else 0.0
                ),
                aux_camera_name=args.aux_camera_window,
                aux_camera_width=args.aux_camera_width,
                aux_camera_height=args.aux_camera_height,
                aux_camera_window_name=args.aux_camera_window or "aux camera",
                recording_hotkeys=recording_hotkeys,
            )

            if args.puma_dataset:
                # Determine success based on current env state after the rollout loop ends
                success = bool(env._check_success())
                if args.save_only_successful and not success:
                    print("Demonstration unsuccessful; not saved (use --save-only-successful off to save all).")
                    continue

                out_path = os.path.join(new_dir, f"ee_state_{saved_idx}.pk")
                _save_ee_state_pickle(traj, out_path)
                print(f"Saved: {out_path}  (T={len(traj['delta_t'])}, success={success})")
                saved_idx += 1

            # Original robosuite pipeline: consolidate the raw npz dumps into demo.hdf5
            assert tmp_directory is not None
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
    finally:
        if args.aux_camera_window is not None:
            try:
                import cv2

                cv2.destroyWindow(args.aux_camera_window)
            except Exception:
                pass
        if recording_hotkeys is not None:
            recording_hotkeys.close()
        device_closer = getattr(device, "close", None)
        if device_closer is not None:
            device_closer()
        env.close()

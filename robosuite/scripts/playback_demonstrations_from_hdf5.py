"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations folder or direct path to .hdf5 file
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random
import re
import sys

import h5py
import numpy as np

import robosuite


def _json_attr_to_dict(attr):
    """Decode an HDF5 attribute that stores a JSON object (robosuite / robomimic)."""
    if isinstance(attr, bytes):
        attr = attr.decode("utf-8")
    return json.loads(attr)


def _make_kwargs_from_data_group(data):
    """
    Build keyword arguments for robosuite.make from the HDF5 ``data`` group.

    - Native robosuite human demos: ``env_info`` (JSON) is a flat dict including
      ``env_name`` and all constructor kwargs.
    - Robomimic datasets: ``env_args`` (JSON) has ``env_name`` and nested
      ``env_kwargs``.
    """
    if "env_info" in data.attrs:
        return _json_attr_to_dict(data.attrs["env_info"])
    if "env_args" in data.attrs:
        env_args = _json_attr_to_dict(data.attrs["env_args"])
        env_name = env_args["env_name"]
        env_kwargs = env_args.get("env_kwargs") or {}
        return {"env_name": env_name, **env_kwargs}
    raise KeyError(
        "HDF5 group 'data' has no 'env_info' (robosuite demos) or 'env_args' (robomimic)."
    )


def _resolve_hdf5_path(demo_path: str) -> str:
    """
    Return path to the demonstrations HDF5.

    Accepts a directory (``demo.hdf5`` or a single ``*.hdf5`` inside), or a direct ``.hdf5`` file.
    """
    demo_path = os.path.expanduser(demo_path)
    if os.path.isfile(demo_path) and demo_path.endswith(".hdf5"):
        return demo_path
    joined = os.path.join(demo_path, "demo.hdf5")
    if os.path.isfile(joined):
        return joined
    if os.path.isdir(demo_path):
        names = sorted(f for f in os.listdir(demo_path) if f.endswith(".hdf5"))
        if len(names) == 1:
            return os.path.join(demo_path, names[0])
    return joined


def _resolve_env_name_for_playback(env_name: str) -> str:
    """
    Robomimic stores names like ``Stack_D1``; robosuite registers the base class ``Stack``.
    If stripping a ``_D<number>`` suffix yields a registered env, use that.
    """
    from robosuite.environments.base import REGISTERED_ENVS
    if env_name in REGISTERED_ENVS:
        return env_name
    m = re.match(r"^(.+)_D\d+$", env_name)
    if m:
        base = m.group(1)
        if base in REGISTERED_ENVS:
            print(
                "[playback_demonstrations_from_hdf5] env_name "
                f"{env_name!r} is not registered; using {base!r} instead."
            )
            return base
    return env_name


def _upgrade_part_controller_config_to_composite(env_info: dict) -> dict:
    """
    Robosuite 1.5+ uses composite controller configs (e.g. type ``BASIC`` with ``body_parts``).
    Robomimic datasets store the older flat ``OSC_POSE`` (etc.) dict, which must be wrapped
    before :func:`robosuite.make` or ``composite_loader`` treats ``OSC_POSE`` as a composite type.
    """
    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            is_part_controller_config,
            refactor_composite_controller_config,
        )
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    except ImportError:
        # robosuite < 1.5 does not have composite controllers; nothing to upgrade.
        return env_info

    cfg = env_info.get("controller_configs")
    if cfg is None or not is_part_controller_config(cfg):
        return env_info
    robots = env_info.get("robots")
    if not robots:
        return env_info
    robot_type = robots[0] if isinstance(robots, (list, tuple)) else robots
    if robot_type not in REGISTERED_ROBOTS:
        return env_info
    arms = REGISTERED_ROBOTS[robot_type].arms
    new_cfg = refactor_composite_controller_config(cfg, robot_type, arms)
    print(
        "[playback_demonstrations_from_hdf5] Upgraded flat controller_configs to composite "
        f"({robot_type}, arms={arms})."
    )
    return {**env_info, "controller_configs": new_cfg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Directory that contains demo.hdf5 or a single *.hdf5 (e.g. robomimic), "
        "or the path to the .hdf5 file directly.",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = _resolve_hdf5_path(demo_path)
    if not os.path.isfile(hdf5_path):
        print(
            "Could not find an HDF5 file.\n"
            f"  Looked for: {hdf5_path!r}\n"
            "  Pass a directory containing demo.hdf5 or exactly one *.hdf5, "
            "or pass the .hdf5 path directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    f = h5py.File(hdf5_path, "r")
    data = f["data"]
    env_info = _make_kwargs_from_data_group(data)
    raw_name = env_info.get("env_name")
    if raw_name is not None:
        env_info = {**env_info, "env_name": _resolve_env_name_for_playback(raw_name)}
    env_info = _upgrade_part_controller_config_to_composite(env_info)

    playback_kwargs = {
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_camera_obs": False,
        "reward_shaping": True,
        "control_freq": 20,
    }
    print("[playback_demonstrations_from_hdf5] Creating environment...")
    env = robosuite.make(**{**env_info, **playback_kwargs})
    print("[playback_demonstrations_from_hdf5] Environment created.")

    # Episode groups contain a `states` dataset (see collect_human_demonstrations.py).
    demos = [k for k in data.keys() if isinstance(data[k], h5py.Group) and "states" in data[k]]
    if not demos:
        f.close()
        print(
            "No playable episodes in {!r}: the HDF5 'data' group has no subgroups with a "
            "'states' dataset.\n"
            "If you used collect_human_demonstrations.py, only successful demonstrations are "
            "saved; unsuccessful runs leave an empty file.".format(hdf5_path),
            file=sys.stderr,
        )
        sys.exit(1)

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            import time
            time.sleep(0.5)

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                import time
                time.sleep(0.02)
                if env.renderer == "mjviewer":
                    env.viewer.update()
                env.render()

    f.close()

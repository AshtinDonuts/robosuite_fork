"""
Launch PlaneFollow with an interactive viewer.

View only (zero actions)::

    python robosuite/demos/demo_plane_follow.py
    python robosuite/demos/demo_plane_follow.py --surface-type curved

Teleoperate (keyboard / SpaceMouse / etc.; no recording)::

    python robosuite/demos/demo_plane_follow.py --device keyboard
    python robosuite/demos/demo_plane_follow.py --device keyboard --surface-type curved
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

MAX_FR = 25


def _make_env(args):
    kwargs = dict(
        robots=args.robots,
        gripper_types="WipingGripper",
        surface_type=args.surface_type,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
    )
    if args.device is not None:
        kwargs.update(
            controller_configs=load_composite_controller_config(
                controller=args.controller,
                robot=args.robots,
            ),
            control_freq=20,
            hard_reset=False,
        )
    env = suite.make("PlaneFollow", **kwargs)
    if args.device is not None:
        env = VisualizationWrapper(env, indicator_configs=None)
    return env


def _make_device(env, args):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
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
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise ValueError(f"Unknown device {args.device!r}; choose keyboard, spacemouse, dualsense, or mjgui.")
    return device


def _run_view_only(env, max_fr):
    env.reset()
    zero = np.zeros(env.action_spec[0].shape)
    while True:
        start = time.time()
        env.step(zero)
        env.render()
        elapsed = time.time() - start
        if (sleep := 1 / max_fr - elapsed) > 0:
            time.sleep(sleep)


def _run_teleop(env, device, max_fr):
    while True:
        env.reset()
        env.render()
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        while True:
            start = time.time()
            active_robot = env.robots[device.active_robot]
            input_ac_dict = device.input2action()
            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError(f"Unsupported controller input type: {controller_input_type}")

            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            env.step(env_action)
            env.render()

            if max_fr is not None:
                elapsed = time.time() - start
                if (sleep := 1 / max_fr - elapsed) > 0:
                    time.sleep(sleep)


def main():
    parser = argparse.ArgumentParser(description="View or teleoperate the PlaneFollow environment.")
    parser.add_argument("--robots", type=str, default="Panda")
    parser.add_argument(
        "--surface-type",
        type=str,
        default="flat",
        choices=("flat", "tilted", "curved", "curved_inverse", "curved_wave"),
        help="Workpiece mesh: flat, tilted, convex/inverse arc (Y), or sinusoidal wave (Y).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("keyboard", "spacemouse", "dualsense", "mjgui"),
        help="Teleop device. Omit for view-only mode (zero actions).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Composite controller (teleop only). Defaults to the robot's BASIC config.",
    )
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="Teleop position gain.")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="Teleop rotation gain.")
    parser.add_argument(
        "--reverse-xy",
        action="store_true",
        help="DualSense only: invert joystick X/Y.",
    )
    parser.add_argument(
        "--max-fr",
        type=int,
        default=MAX_FR,
        help="Viewer frame-rate cap (default 25 for view-only, 20 typical for teleop).",
    )
    args = parser.parse_args()

    env = _make_env(args)
    if args.device is None:
        _run_view_only(env, args.max_fr)
    else:
        device = _make_device(env, args)
        _run_teleop(env, device, args.max_fr)


if __name__ == "__main__":
    main()

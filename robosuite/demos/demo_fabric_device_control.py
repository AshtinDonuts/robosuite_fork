"""Teleoperate a Panda robot with fabric obstacle avoidance.

Drop-in replacement for demo_device_control.py that uses
FabricGuidedOSCController so obstacle avoidance runs on top of every
teleoperation command.

The environment is LiftWithObstacle: a standard Lift task with a fixed
red sphere added to the scene.  The fabric correction keeps every arm
link away from the sphere even when the user drives toward it.

Usage (keyboard):
    conda activate pumafabrics
    cd /home/khw/Projects/robosuite-master
    python -m robosuite.demos.demo_fabric_device_control

Usage (SpaceMouse):
    python -m robosuite.demos.demo_fabric_device_control --device spacemouse

Arguments are a subset of demo_device_control.py:
    --device        keyboard (default) | spacemouse | dualsense | mjgui
    --pos-sensitivity / --rot-sensitivity
    --max_fr        target frame rate (default 20)
"""

import argparse
import time

import numpy as np

from robosuite.wrappers import VisualizationWrapper
from robosuite.demos.demo_fabric_guided_osc import (
    LiftWithObstacle,
    make_controller_config,
    OBSTACLE_POS,
    OBSTACLE_RADIUS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--max_fr", type=int, default=20)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Environment: Lift + fixed red sphere obstacle                        #
    # ------------------------------------------------------------------ #
    env = LiftWithObstacle(
        robots="Panda",
        controller_configs=make_controller_config(),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )
    env = VisualizationWrapper(env, indicator_configs=None)

    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------ #
    # Input device                                                         #
    # ------------------------------------------------------------------ #
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
        )
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise ValueError(f"Unknown device: {args.device}")

    print(f"\nFabric-guided OSC teleoperation")
    print(f"  Obstacle sphere : pos={OBSTACLE_POS}  radius={OBSTACLE_RADIUS} m")
    print(f"  The arm will deflect away from the red sphere automatically.\n")

    # ------------------------------------------------------------------ #
    # Main loop (mirrors demo_device_control.py)                          #
    # ------------------------------------------------------------------ #
    from robosuite.controllers.composite.composite_controller import WholeBody
    from copy import deepcopy

    while True:
        obs = env.reset()
        env.render()

        # --- Wire the live obstacle callback after every reset ---------- #
        # The controller is rebuilt on each reset; re-attach obstacle_fn. #
        arm_ctrl = env.env.robots[0].composite_controller.part_controllers["right"]
        arm_ctrl.obstacle_fn = env.env.get_obstacle_pos_radius
        print(f"Fabrics planner active: {arm_ctrl._fabrics_ready}  "
              f"obstacle_fn wired: {arm_ctrl.obstacle_fn is not None}")

        last_grasp = 0
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
                    controller_input_type = (
                        active_robot.composite_controller.joint_action_policy.input_type
                    )
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError(f"Unknown input type: {controller_input_type}")

            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i])
                for i, robot in enumerate(env.robots)
            ]
            env_action[device.active_robot] = active_robot.create_action_vector(
                action_dict
            )
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = (
                    action_dict[gripper_ac]
                )

            env.step(env_action)
            env.render()

            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)


if __name__ == "__main__":
    main()

"""
    Launch PlaneFollow with an interactive viewer (zero actions).
    python robosuite/demos/demo_plane_follow.py
"""

import argparse
import time

import numpy as np

import robosuite as suite

MAX_FR = 25


def main():
    parser = argparse.ArgumentParser(description="View the PlaneFollow environment.")
    parser.add_argument("--robots", type=str, default="Panda")
    parser.add_argument(
        "--surface-type",
        type=str,
        default="flat",
        choices=("flat", "tilted", "curved"),
        help="Workpiece mesh: flat slab, tilted slab, or curved cap.",
    )
    args = parser.parse_args()

    env = suite.make(
        "PlaneFollow",
        robots=args.robots,
        surface_type=args.surface_type,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
    )
    env.reset()

    zero = np.zeros(env.action_spec[0].shape)
    while True:
        start = time.time()
        env.step(zero)
        env.render()
        elapsed = time.time() - start
        if (sleep := 1 / MAX_FR - elapsed) > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()

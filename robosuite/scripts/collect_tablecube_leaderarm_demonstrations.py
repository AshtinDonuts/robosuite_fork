"""
Launch TableCubePick demonstration collection with VX300S leader-arm teleop defaults.

Extra command-line arguments are forwarded to collect_human_demonstrations.py.
For options that take a value, forwarded arguments can override the defaults here
because argparse keeps the last occurrence.
"""

import os
import subprocess
import sys


DEFAULT_ARGS = [
    "--environment",
    "TableCubePick",
    "--robots",
    "VX300S",
    "--renderer",
    "mjviewer",
    "--camera",
    "frontview",
    "--aux-camera-window",
    "robot0_eye_in_hand",
    "--eps_name",
    "tablecube1",
    "--puma_dataset",
    "--device",
    "trossen_leaderarm",
    "--leaderarm-topic",
    "/leader_solo/joint_states",
    "--visualize-keypoints",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    collector = os.path.join(script_dir, "collect_human_demonstrations.py")
    cmd = [sys.executable, collector, *DEFAULT_ARGS, *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, cwd=repo_root).returncode)


if __name__ == "__main__":
    main()

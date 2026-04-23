"""
Playback puma_dataset-style end-effector pose traces in robosuite.

This script expects an episode directory containing:
  - env_info.json
  - ee_state_*.pk  (pickles with keys: x_pos, x_rot, x_dot, delta_t)

Example:
    $ python demo_playback_puma_dataset.py --episode_dir /path/to/episode
"""

import argparse
import json
import os
import pickle
import time
from glob import glob

import numpy as np

import robosuite as suite
from robosuite.utils.control_utils import orientation_error


def _load_env_info(env_info_path: str) -> dict:
    with open(env_info_path, "r") as f:
        return json.load(f)


def _load_ee_trace(pk_path: str) -> dict:
    with open(pk_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {pk_path}, got {type(data)}")
    for k in ("x_pos", "x_rot", "delta_t"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {pk_path}. Found keys: {sorted(list(data.keys()))}")
    return data


def _iter_episode_steps(episode_dir: str):
    pk_paths = sorted(glob(os.path.join(episode_dir, "ee_state_*.pk")))
    if not pk_paths:
        raise FileNotFoundError(f"No ee_state_*.pk found under {episode_dir}")
    for pk_idx, pk_path in enumerate(pk_paths):
        print(f'Replaying Episdoe: {pk_idx}')
        trace = _load_ee_trace(pk_path)
        x_pos = trace["x_pos"]
        x_rot = trace["x_rot"]
        delta_t = np.asarray(trace["delta_t"])
        n = min(len(x_pos), len(x_rot), len(delta_t))
        for i in range(n):
            yield np.asarray(x_pos[i], dtype=np.float64), np.asarray(x_rot[i], dtype=np.float64), float(delta_t[i])


def _get_osc_output_max(controller_configs: dict) -> np.ndarray:
    """
    Best-effort extraction of OSC output_max for the right arm.
    Falls back to robosuite defaults if missing.
    """
    # robosuite composite "BASIC" schema: controller_configs["body_parts"]["right"]
    try:
        cfg = controller_configs["body_parts"]["right"]
        out_max = np.asarray(cfg.get("output_max", [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]), dtype=np.float64)
        if out_max.shape != (6,):
            raise ValueError
        return out_max
    except Exception:
        return np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float64)


def _build_action(delta_pos: np.ndarray, delta_ori: np.ndarray, out_max6: np.ndarray, action_dim: int) -> np.ndarray:
    """
    Convert desired delta pos / ori (in controller output units) into a normalized action in [-1, 1].
    Assumes symmetric output limits (default robosuite OSC configs are symmetric).
    """
    delta6 = np.concatenate([delta_pos, delta_ori], axis=0)
    delta6 = np.clip(delta6, -out_max6, out_max6)
    act6 = delta6 / out_max6
    act6 = np.clip(act6, -1.0, 1.0)

    if action_dim == 6:
        return act6
    if action_dim == 7:
        # Keep gripper neutral (no open/close command)
        return np.concatenate([act6, np.array([0.0], dtype=np.float64)], axis=0)
    # Generic fallback: pad / trim
    if action_dim > 6:
        pad = np.zeros(action_dim - 6, dtype=np.float64)
        return np.concatenate([act6, pad], axis=0)
    return act6[:action_dim]


def playback_puma_episode(env, episode_dir: str, max_fr: int | None = 20, realtime_from_delta_t: bool = False):
    env.reset()

    # OSC scaling (used to normalize deltas into action range)
    controller_configs = getattr(env, "controller_configs", None)
    out_max6 = _get_osc_output_max(controller_configs or {})

    # Resolve which arm to use (default to right if present)
    robot = env.robots[0]
    arm = "right" if "right" in robot.arms else robot.arms[0]

    for (tpos, trot, dt) in _iter_episode_steps(episode_dir):
        start = time.time()

        cur_pos = np.asarray(robot._hand_pos[arm], dtype=np.float64)
        cur_rot = np.asarray(robot._hand_orn[arm], dtype=np.float64)

        delta_pos = tpos - cur_pos
        delta_ori = orientation_error(trot, cur_rot)

        action = _build_action(delta_pos, delta_ori, out_max6, env.action_dim)
        env.step(action)
        env.render()

        # pacing
        if realtime_from_delta_t:
            sleep_s = max(0.0, dt - (time.time() - start))
            if sleep_s > 0:
                time.sleep(sleep_s)
        elif max_fr is not None and max_fr > 0:
            elapsed = time.time() - start
            diff = 1.0 / float(max_fr) - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True, help="Directory containing env_info.json + ee_state_*.pk")
    parser.add_argument("--max_fr", type=int, default=20, help="Limit playback to this FPS (ignored if --realtime)")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="If set, sleep according to delta_t stored in the puma dataset files",
    )
    parser.add_argument("--render_camera", type=str, default="frontview", help="Camera name for onscreen renderer")
    args = parser.parse_args()

    env_info_path = os.path.join(args.episode_dir, "env_info.json")
    if not os.path.exists(env_info_path):
        raise FileNotFoundError(f"Missing env_info.json at: {env_info_path}")

    env_info = _load_env_info(env_info_path)
    env_name = env_info.get("env_name", "Lift")
    robots = env_info.get("robots", ["Panda"])
    controller_configs = env_info.get("controller_configs", None)

    env = suite.make(
        env_name,
        robots=robots,
        controller_configs=controller_configs,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.render_camera,
        control_freq=20,
    )

    playback_puma_episode(env, args.episode_dir, max_fr=args.max_fr, realtime_from_delta_t=args.realtime)

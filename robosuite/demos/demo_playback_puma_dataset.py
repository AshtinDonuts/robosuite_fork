from __future__ import annotations

"""
Playback puma_dataset-style end-effector pose traces in robosuite.

This script expects an episode directory containing:
  - env_info.json
  - ee_state_*.pk  (pickles with keys: x_pos, x_rot, x_dot, delta_t[, gripper_action])

For each trajectory, a red non-collidable sphere (MuJoCo site) marks the demonstration
endpoint at the final recorded EE position (``x_pos[-1]``, world frame).

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


def _episode_pk_paths(episode_dir: str) -> list[str]:
    pk_paths = sorted(glob(os.path.join(episode_dir, "ee_state_*.pk")))
    if not pk_paths:
        raise FileNotFoundError(f"No ee_state_*.pk found under {episode_dir}")
    return pk_paths


GOAL_MARKER_RGBA = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float64)
GOAL_MARKER_RADIUS = 0.03


def _goal_position_from_trace(trace: dict) -> np.ndarray:
    """Final demonstrated EE position in world frame (demonstration endpoint)."""
    return np.asarray(trace["x_pos"][-1], dtype=np.float64)


class _GoalPositionMarker:
    """Non-collidable red sphere (MuJoCo site) at the trajectory goal."""

    _SITE_SUFFIX = "goal_position_marker"

    def __init__(self, env):
        self.env = env
        self._warned_missing = False

    def _resolve_site_id(self, sim):
        for name in sim.model.site_names:
            if name.endswith(self._SITE_SUFFIX):
                return sim.model.site_name2id(name)
        return None

    def set_goal(self, goal_pos: np.ndarray):
        # env.sim is replaced on hard_reset(); always use the live sim and re-resolve the site.
        sim = self.env.sim
        site_id = self._resolve_site_id(sim)
        if site_id is None:
            if not self._warned_missing:
                print(
                    "Warning: goal_position_marker site not found in the environment model; "
                    "goal marker will not be shown.",
                    flush=True,
                )
                self._warned_missing = True
            return
        goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        sim.model.site_pos[site_id] = goal_pos
        sim.model.site_size[site_id] = GOAL_MARKER_RADIUS
        sim.model.site_rgba[site_id] = GOAL_MARKER_RGBA
        sim.forward()


def _iter_trace_steps(trace: dict):
    x_pos = trace["x_pos"]
    x_rot = trace["x_rot"]
    delta_t = np.asarray(trace["delta_t"])
    gripper_action = trace.get("gripper_action", None)
    n = min(len(x_pos), len(x_rot), len(delta_t))
    for i in range(n):
        ga = np.asarray(gripper_action[i], dtype=np.float64) if gripper_action is not None else None
        yield (
            np.asarray(x_pos[i], dtype=np.float64),
            np.asarray(x_rot[i], dtype=np.float64),
            float(delta_t[i]),
            ga,
        )


def _go_to_homepose(env):
    print("Resetting robot to home pose...", flush=True)
    env.reset()


def _sleep_settle(settle_sec: float, label: str = "starting pose"):
    if settle_sec <= 0:
        return
    print(f"Waiting {settle_sec:.1f}s for robot to reach {label}...", flush=True)
    time.sleep(settle_sec)


def _step_toward_target(
    env,
    robot,
    arm: str,
    tpos: np.ndarray,
    trot: np.ndarray,
    out_max6: np.ndarray,
    ref_frame: str,
    gripper_action: np.ndarray | None,
    max_fr: int | None,
    realtime_from_delta_t: bool,
    dt: float = 0.0,
) -> tuple[float, float]:
    """One OSC step toward target pose. Returns (position error norm, orientation error norm) in world frame."""
    start = time.time()
    site_id = robot.eef_site_id[arm]
    cur_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    cur_rot = np.asarray(env.sim.data.site_xmat[site_id].reshape(3, 3), dtype=np.float64)

    delta_pos = tpos - cur_pos
    delta_ori = orientation_error(trot, cur_rot)
    pos_err = float(np.linalg.norm(delta_pos))
    ori_err = float(np.linalg.norm(delta_ori))

    if ref_frame == "base":
        delta_pos = _world_vec_to_controller_base(robot, arm, delta_pos)
        delta_ori = _world_vec_to_controller_base(robot, arm, delta_ori)

    action = _build_action(delta_pos, delta_ori, out_max6, env.action_dim, gripper_action=gripper_action)
    env.step(action)
    env.render()

    if realtime_from_delta_t:
        sleep_s = max(0.0, dt - (time.time() - start))
        if sleep_s > 0:
            time.sleep(sleep_s)
    elif max_fr is not None and max_fr > 0:
        elapsed = time.time() - start
        diff = 1.0 / float(max_fr) - elapsed
        if diff > 0:
            time.sleep(diff)

    return pos_err, ori_err


def _reach_pose(
    env,
    robot,
    arm: str,
    tpos: np.ndarray,
    trot: np.ndarray,
    out_max6: np.ndarray,
    ref_frame: str,
    max_fr: int | None,
    realtime_from_delta_t: bool,
    pos_tol: float = 0.01,
    ori_tol: float = 0.1,
    max_steps: int = 400,
    gripper_action: np.ndarray | None = None,
):
    for _ in range(max_steps):
        pos_err, ori_err = _step_toward_target(
            env,
            robot,
            arm,
            tpos,
            trot,
            out_max6,
            ref_frame,
            gripper_action,
            max_fr,
            realtime_from_delta_t,
        )
        if pos_err < pos_tol and ori_err < ori_tol:
            break


def _prepare_episode_start(
    env,
    robot,
    arm: str,
    trace: dict,
    out_max6: np.ndarray,
    ref_frame: str,
    max_fr: int | None,
    realtime_from_delta_t: bool,
    start_settle_sec: float,
):
    x_pos = trace["x_pos"]
    x_rot = trace["x_rot"]
    gripper_action = trace.get("gripper_action", None)
    tpos = np.asarray(x_pos[0], dtype=np.float64)
    trot = np.asarray(x_rot[0], dtype=np.float64)
    ga = np.asarray(gripper_action[0], dtype=np.float64) if gripper_action is not None else None

    print("Moving to episode starting pose...", flush=True)
    _reach_pose(
        env,
        robot,
        arm,
        tpos,
        trot,
        out_max6,
        ref_frame,
        max_fr,
        realtime_from_delta_t,
        gripper_action=ga,
    )
    _sleep_settle(start_settle_sec, label="starting pose")


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


def _build_action(
    delta_pos: np.ndarray,
    delta_ori: np.ndarray,
    out_max6: np.ndarray,
    action_dim: int,
    gripper_action: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert desired delta pos / ori (in controller output units) into a normalized action in [-1, 1].
    Assumes symmetric output limits (default robosuite OSC configs are symmetric).

    gripper_action: recorded per-step gripper command from the pickle (+1=close, -1=open).
                    When None (old pickles without the key), gripper stays neutral (0.0).
    """
    delta6 = np.concatenate([delta_pos, delta_ori], axis=0)
    delta6 = np.clip(delta6, -out_max6, out_max6)
    act6 = delta6 / out_max6
    act6 = np.clip(act6, -1.0, 1.0)

    if action_dim == 6:
        return act6

    gripper_dof = action_dim - 6
    if gripper_dof > 0:
        if gripper_action is not None:
            ga = np.asarray(gripper_action, dtype=np.float64).reshape(-1)
            # Pad or trim to exactly gripper_dof elements
            if len(ga) < gripper_dof:
                ga = np.concatenate([ga, np.zeros(gripper_dof - len(ga), dtype=np.float64)])
            else:
                ga = ga[:gripper_dof]
        else:
            ga = np.zeros(gripper_dof, dtype=np.float64)
        return np.concatenate([act6, ga], axis=0)

    return act6[:action_dim]


def _resolve_action_ref_frame(env_info: dict, action_ref_frame: str) -> str:
    """
    Returns "world" or "base".

    - If action_ref_frame == "auto", best-effort infer from env_info controller_configs.
    - Otherwise, returns the user-specified value.
    """
    if action_ref_frame != "auto":
        return action_ref_frame
    try:
        cfg = env_info.get("controller_configs", {}) or {}
        ref = cfg.get("body_parts", {}).get("right", {}).get("input_ref_frame", None)
        if isinstance(ref, str) and ref in ("world", "base"):
            return ref
    except Exception:
        pass
    return "world"


def _world_vec_to_controller_base(robot, arm: str, vec3: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector expressed in world frame to the controller "base" frame.

    Important: For OSC, "base" corresponds to the controller origin frame (the
    `{naming_prefix}{part_name}_center` site used by the composite controller),
    not necessarily the robot root body frame.
    """
    base_ori = None
    try:
        cc = getattr(robot, "composite_controller", None)
        if cc is not None and hasattr(cc, "get_controller_base_pose"):
            _, base_ori = cc.get_controller_base_pose(controller_name=arm)
    except Exception:
        base_ori = None

    if base_ori is None:
        # Fallback: last-resort use robot root body orientation (may be wrong for some robots)
        try:
            base_ori = robot.sim.data.get_body_xmat(robot.robot_model.root_body).reshape((3, 3))
        except Exception:
            base_ori = robot.base_ori

    world_R_base = np.asarray(base_ori, dtype=np.float64).reshape(3, 3)
    return world_R_base.T @ np.asarray(vec3, dtype=np.float64).reshape(3,)


def playback_puma_episode(
    env,
    episode_dir: str,
    env_info: dict,
    max_fr: int | None = 20,
    realtime_from_delta_t: bool = False,
    action_ref_frame: str = "auto",
    start_settle_sec: float = 2.0,
):
    env.reset()

    # OSC scaling (used to normalize deltas into action range)
    controller_configs = getattr(env, "controller_configs", None)
    out_max6 = _get_osc_output_max(controller_configs or {})

    # Resolve which arm to use (default to right if present)
    robot = env.robots[0]
    arm = "right" if "right" in robot.arms else robot.arms[0]

    ref_frame = _resolve_action_ref_frame(env_info, action_ref_frame)

    goal_marker = _GoalPositionMarker(env)

    pk_paths = _episode_pk_paths(episode_dir)
    for ep_idx, pk_path in enumerate(pk_paths):
        trace = _load_ee_trace(pk_path)
        goal_marker.set_goal(_goal_position_from_trace(trace))
        _prepare_episode_start(
            env,
            robot,
            arm,
            trace,
            out_max6,
            ref_frame,
            max_fr,
            realtime_from_delta_t,
            start_settle_sec,
        )

        print(f"Replaying Episode: {ep_idx}")
        for (tpos, trot, dt, gripper_ac) in _iter_trace_steps(trace):
            _step_toward_target(
                env,
                robot,
                arm,
                tpos,
                trot,
                out_max6,
                ref_frame,
                gripper_ac,
                max_fr,
                realtime_from_delta_t,
                dt=dt,
            )

        if ep_idx + 1 < len(pk_paths):
            _go_to_homepose(env)

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
    parser.add_argument(
        "--action_ref_frame",
        type=str,
        default="auto",
        choices=["auto", "world", "base"],
        help="Frame for OSC delta commands. 'auto' uses env_info controller_configs input_ref_frame when available.",
    )
    parser.add_argument("--render_camera", type=str, default="frontview", help="Camera name for onscreen renderer")
    parser.add_argument(
        "--start_settle_sec",
        type=float,
        default=0.1,
        help="Seconds to wait after reaching the episode starting pose before replay begins",
    )
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

    playback_puma_episode(
        env,
        args.episode_dir,
        env_info=env_info,
        max_fr=args.max_fr,
        realtime_from_delta_t=args.realtime,
        action_ref_frame=args.action_ref_frame,
        start_settle_sec=args.start_settle_sec,
    )

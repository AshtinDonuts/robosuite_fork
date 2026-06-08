from __future__ import annotations

"""
Playback puma_dataset-style end-effector pose traces in robosuite.

This script expects an episode directory containing:
  - env_info.json
  - ee_state_*.pk  (pickles with keys: x_pos, x_rot, x_dot, delta_t[, gripper_action, initial_state])
  - demo.hdf5      (optional, exact robosuite state/action replay when available)

Visualization markers (non-collidable cylinder sites, requires the Empty environment):
  - Blue cylinder at x_pos[0] / x_rot[0]  — demonstration start pose
  - Red  cylinder at x_pos[-1] / x_rot[-1] — demonstration end pose
  - Green cylinders every --waypoint_stride steps along the recorded trajectory

Example:
    $ python demo_playback_puma_dataset.py --episode_dir /path/to/episode
"""
import argparse
import copy
import json
import os
import pickle
import time
from glob import glob

import mujoco
import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
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


# Cylinder marker dimensions: [radius_m, half_length_m, unused]
START_GOAL_MARKER_SIZE = np.array([0.02, 0.05, 0.0], dtype=np.float64)
WAYPOINT_MARKER_SIZE = np.array([0.008, 0.03, 0.0], dtype=np.float64)
START_MARKER_RGBA = np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float64)
GOAL_MARKER_RGBA = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float64)
WAYPOINT_MARKER_RGBA = np.array([0.0, 0.75, 0.0, 0.85], dtype=np.float64)

# Must be >= the number of waypoint_marker_N sites pre-allocated in empty.py (currently 60).
_N_WAYPOINT_SITES = 100

# Which EE axis is the "forward" (approach) direction shown as the cylinder axis.
# 0=X, 1=Y, 2=Z. PUMA/PyBullet playback draws trace cylinders along
# the local Z column of x_rot. MuJoCo site_xmat and SciPy matrices share the
# same column-as-local-axis convention, so do not transpose x_rot here.
_EE_FORWARD_AXIS = 2


def _marker_direction(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    direction = rot[:, _EE_FORWARD_AXIS]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / norm


def _marker_segment(pos: np.ndarray, rot: np.ndarray, size3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(pos, dtype=np.float64).reshape(3)
    half_length = float(np.asarray(size3, dtype=np.float64).reshape(-1)[1])
    end = start + 2.0 * half_length * _marker_direction(rot)
    return start, end


def _hide_site_marker(sim, site_id: int) -> None:
    sim.model.site_pos[site_id] = np.array([0.0, 0.0, -100.0], dtype=np.float64)
    hidden = sim.model.site_rgba[site_id].copy()
    hidden[3] = 0.0
    sim.model.site_rgba[site_id] = hidden


def _install_connector_marker_renderer(env) -> None:
    if getattr(env, "_puma_connector_marker_renderer", False):
        return
    env._puma_connector_marker_renderer = True
    env._puma_connector_markers = []

    original_render = env.render

    def render_with_markers(*args, **kwargs):
        viewer_wrapper = getattr(env, "viewer", None)
        viewer = getattr(viewer_wrapper, "viewer", None)
        if viewer is not None and hasattr(viewer, "user_scn"):
            markers = getattr(env, "_puma_connector_markers", [])
            viewer.user_scn.ngeom = 0
            for start, end, radius, rgba in markers:
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.eye(3, dtype=np.float64).reshape(-1),
                    rgba,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    float(radius),
                    np.asarray(start, dtype=np.float64),
                    np.asarray(end, dtype=np.float64),
                )
                viewer.user_scn.ngeom += 1
        return original_render(*args, **kwargs)

    env.render = render_with_markers


def _set_connector_markers(env, markers: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> None:
    _install_connector_marker_renderer(env)
    env._puma_connector_markers = markers


class _PoseMarker:
    """Non-collidable cylinder site showing a full pose (position + orientation)."""

    def __init__(self, env, site_suffix: str, rgba: np.ndarray, label: str, size3: np.ndarray):
        self.env = env
        self.site_suffix = site_suffix
        self.rgba = np.asarray(rgba, dtype=np.float64)
        self.label = label
        self.size3 = np.asarray(size3, dtype=np.float64)
        self._warned_missing = False

    def _resolve_site_id(self, sim):
        for name in sim.model.site_names:
            if name.endswith(self.site_suffix):
                return sim.model.site_name2id(name)
        return None

    def marker_segment(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        start, end = _marker_segment(pos, rot, self.size3)
        return start, end, float(self.size3[0]), self.rgba

    def set_pose(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
        # Hide the preallocated MuJoCo site because site cylinder orientation was unreliable here.
        # The visible cylinder is drawn as a viewer connector between two world points.
        sim = self.env.sim
        site_id = self._resolve_site_id(sim)
        if site_id is not None:
            _hide_site_marker(sim, site_id)
            sim.forward()
        elif not self._warned_missing:
            print(
                f"Warning: {self.site_suffix} site not found in the environment model; "
                f"{self.label} marker will be drawn only in the viewer overlay.",
                flush=True,
            )
            self._warned_missing = True
        return self.marker_segment(pos, rot)


class _WaypointMarkerPool:
    """
    Manages the pre-allocated pool of waypoint_marker_N cylinder sites (defined in empty.py).

    Unused slots are hidden by setting alpha=0.  All model writes are batched before a
    single sim.forward() call for efficiency.
    """

    def __init__(self, env, rgba: np.ndarray, size3: np.ndarray):
        self.env = env
        self.rgba = np.asarray(rgba, dtype=np.float64)
        self.size3 = np.asarray(size3, dtype=np.float64)

    def _site_id(self, sim, idx: int) -> int | None:
        try:
            return sim.model.site_name2id(f"waypoint_marker_{idx}")
        except Exception:
            return None

    def place_waypoints(self, trace: dict, stride: int) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        sim = self.env.sim
        x_pos = trace["x_pos"]
        x_rot = trace["x_rot"]
        n = min(len(x_pos), len(x_rot))
        wp_indices = list(range(0, n, stride))
        markers = []
        slot = 0
        for wi in wp_indices:
            site_id = self._site_id(sim, slot)
            if site_id is None:
                break
            _hide_site_marker(sim, site_id)
            start, end = _marker_segment(
                np.asarray(x_pos[wi], dtype=np.float64),
                np.asarray(x_rot[wi], dtype=np.float64),
                self.size3,
            )
            markers.append((start, end, float(self.size3[0]), self.rgba))
            slot += 1
        # hide remaining pre-allocated slots
        for i in range(slot, _N_WAYPOINT_SITES):
            site_id = self._site_id(sim, i)
            if site_id is None:
                break
            _hide_site_marker(sim, site_id)
        sim.forward()
        return markers


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


def _restore_trace_initial_state(env, trace: dict) -> bool:
    initial_state = trace.get("initial_state", None)
    if initial_state is None:
        return False
    env.sim.set_state_from_flattened(np.asarray(initial_state, dtype=np.float64))
    env.sim.forward()
    return True


def _trace_start_pose_error(env, robot, arm: str, trace: dict) -> float:
    site_id = robot.eef_site_id[arm]
    cur_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    start_pos = np.asarray(trace["x_pos"][0], dtype=np.float64)
    return float(np.linalg.norm(start_pos - cur_pos))


def _rotmat_to_abs_osc_action(tpos: np.ndarray, trot: np.ndarray) -> np.ndarray:
    """OSC absolute mode expects world position (3) + axis-angle orientation (3)."""
    return np.concatenate(
        [
            np.asarray(tpos, dtype=np.float64),
            T.quat2axisangle(T.mat2quat(np.asarray(trot, dtype=np.float64))),
        ]
    )


def _attach_gripper_to_arm_action(arm_action: np.ndarray, action_dim: int, gripper_action: np.ndarray | None) -> np.ndarray:
    if action_dim == arm_action.size:
        return arm_action
    gripper_dof = action_dim - arm_action.size
    if gripper_dof <= 0:
        return arm_action[:action_dim]
    if gripper_action is not None:
        ga = np.asarray(gripper_action, dtype=np.float64).reshape(-1)
        if ga.size < gripper_dof:
            ga = np.concatenate([ga, np.zeros(gripper_dof - ga.size, dtype=np.float64)])
        else:
            ga = ga[:gripper_dof]
    else:
        ga = np.zeros(gripper_dof, dtype=np.float64)
    return np.concatenate([arm_action, ga], axis=0)


def _solve_site_pose_ik(
    sim,
    robot,
    arm: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_seed: np.ndarray | None = None,
    max_iters: int = 50,
    damping: float = 0.05,
    step_size: float = 0.85,
    pos_tol: float = 2e-4,
    ori_tol: float = 0.02,
) -> np.ndarray:
    """
    Damped least-squares IK on the controlled arm dofs for a grip-site pose.

    Uses the current sim joint configuration as the seed so consecutive
    waypoints stay on the same kinematic branch as the recorded trajectory.
    """
    controller = robot.part_controllers[arm]
    qpos_ids = controller.joint_indexes["qpos"]
    qvel_ids = controller.joint_indexes["qvel"]
    joint_ids = controller.joint_indexes["joints"]
    site_id = robot.eef_site_id[arm]

    if q_seed is None:
        q = np.asarray(sim.data.qpos[qpos_ids], dtype=np.float64)
    else:
        q = np.asarray(q_seed, dtype=np.float64)

    lo = sim.model.jnt_range[joint_ids, 0]
    hi = sim.model.jnt_range[joint_ids, 1]
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)

    model = sim.model._model
    data = sim.data._data
    for _ in range(max_iters):
        sim.data.qpos[qpos_ids] = q
        sim.forward()
        pos_err = target_pos - sim.data.site_xpos[site_id]
        ori_err = orientation_error(target_rot, sim.data.site_xmat[site_id].reshape(3, 3))
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
            break

        jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
        jacr = np.zeros((3, sim.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, qvel_ids], jacr[:, qvel_ids]])
        err6 = np.concatenate([pos_err, ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(6), err6)
        q = np.clip(q + step_size * dq, lo, hi)

    return q


def _replay_trace_waypoint_ik(
    env,
    robot,
    arm: str,
    tpos: np.ndarray,
    trot: np.ndarray,
    gripper_action: np.ndarray | None,
    ik_max_iters: int,
    ik_settle_steps: int,
):
    controller = robot.part_controllers[arm]
    qpos_ids = controller.joint_indexes["qpos"]
    q_seed = np.asarray(env.sim.data.qpos[qpos_ids], dtype=np.float64)
    q_target = _solve_site_pose_ik(
        env.sim,
        robot,
        arm,
        tpos,
        trot,
        q_seed=q_seed,
        max_iters=ik_max_iters,
    )
    arm_action = q_target
    action = _attach_gripper_to_arm_action(arm_action, env.action_dim, gripper_action)
    settle_steps = max(1, int(ik_settle_steps))
    for _ in range(settle_steps):
        env.step(action)
        _maybe_render(env)


def _go_to_homepose(env):
    print("Resetting robot to home pose...", flush=True)
    env.reset()


def _episode_hdf5_path(episode_dir: str) -> str:
    return os.path.join(episode_dir, "demo.hdf5")


def _maybe_render(env) -> None:
    if getattr(env, "has_renderer", False):
        env.render()


def _sleep_for_replay_step(start: float, dt: float, max_fr: int | None, realtime_from_delta_t: bool):
    if realtime_from_delta_t:
        sleep_s = max(0.0, dt - (time.time() - start))
        if sleep_s > 0:
            time.sleep(sleep_s)
    elif max_fr is not None and max_fr > 0:
        elapsed = time.time() - start
        diff = 1.0 / float(max_fr) - elapsed
        if diff > 0:
            time.sleep(diff)


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
    ori_action_scale: float = 1.0,
) -> tuple[float, float]:
    """One OSC step toward target pose. Returns (position error norm, orientation error norm) in world frame."""
    start = time.time()
    site_id = robot.eef_site_id[arm]
    cur_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    cur_rot = np.asarray(env.sim.data.site_xmat[site_id].reshape(3, 3), dtype=np.float64)

    pos_err = float(np.linalg.norm(tpos - cur_pos))
    ori_err = float(np.linalg.norm(orientation_error(trot, cur_rot)))

    delta_pos, delta_ori = _pose_error_to_controller_delta(
        robot,
        arm,
        tpos,
        trot,
        cur_pos,
        cur_rot,
        ref_frame,
    )
    delta_ori = delta_ori * float(ori_action_scale)

    action = _build_action(delta_pos, delta_ori, out_max6, env.action_dim, gripper_action=gripper_action)
    env.step(action)
    _maybe_render(env)

    _sleep_for_replay_step(start, dt, max_fr, realtime_from_delta_t)

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
    ori_action_scale: float = 1.0,
    require_orientation: bool = True,
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
            ori_action_scale=ori_action_scale,
        )
        ori_ok = ori_err < ori_tol if require_orientation else True
        if pos_err < pos_tol and ori_ok:
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
    ee_trace_controller: str = "ik",
    ik_max_iters: int = 50,
    ik_settle_steps: int = 1,
    start_pose_tol: float = 0.002,
    replay_ori_action_scale: float = 1.0,
):
    x_pos = trace["x_pos"]
    x_rot = trace["x_rot"]
    gripper_action = trace.get("gripper_action", None)
    tpos = np.asarray(x_pos[0], dtype=np.float64)
    trot = np.asarray(x_rot[0], dtype=np.float64)
    ga = np.asarray(gripper_action[0], dtype=np.float64) if gripper_action is not None else None

    if _trace_start_pose_error(env, robot, arm, trace) <= start_pose_tol:
        print("Already at episode starting pose.", flush=True)
        return

    print("Moving to episode starting pose...", flush=True)
    if ee_trace_controller == "ik":
        _replay_trace_waypoint_ik(
            env,
            robot,
            arm,
            tpos,
            trot,
            ga,
            ik_max_iters=max(ik_max_iters, 80),
            ik_settle_steps=max(ik_settle_steps, 5),
        )
        return

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
        pos_tol=start_pose_tol,
        ori_tol=0.15,
        max_steps=80,
        gripper_action=ga,
        ori_action_scale=replay_ori_action_scale,
        require_orientation=True,
    )


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

    PUMA pickles store achieved EE poses in MuJoCo world frame (site_xpos / site_xmat).
    Playback therefore defaults to world-frame OSC deltas unless the user overrides.
    """
    if action_ref_frame != "auto":
        return action_ref_frame
    return "world"


def _episode_records_puma_ee_trace(env_info: dict) -> bool:
    recorded_files = env_info.get("recorded_files", {})
    if isinstance(recorded_files, dict) and "ee_state_*.pk" in recorded_files:
        return True
    # Backward compatibility: older puma_dataset episodes predate recorded_files metadata.
    return True


def _controller_configs_for_playback(
    env_info: dict,
    robots,
    ref_frame: str,
    ee_trace_controller: str = "ik",
    osc_kp: int = 500,
) -> dict | None:
    """
    Build the controller used to replay puma_dataset EE traces.

    Collection may have used joint teleop, but ee_state_*.pk stores achieved
    Cartesian EE poses. Default playback solves IK and tracks them with
    JOINT_POSITION; OSC_POSE remains available via --ee_trace_controller osc.
    """
    if _episode_records_puma_ee_trace(env_info):
        primary_robot = robots[0] if isinstance(robots, list) else robots
        if ee_trace_controller == "ik":
            arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
            arm_cfg["input_type"] = "absolute"
        else:
            arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
            arm_cfg["input_ref_frame"] = ref_frame
            arm_cfg["input_type"] = "absolute"
            arm_cfg["kp"] = int(osc_kp)
        return refactor_composite_controller_config(arm_cfg, primary_robot, ["right"])

    cfg = copy.deepcopy(env_info.get("controller_configs", None))
    if cfg is None or ref_frame == "base":
        return cfg
    body_parts = cfg.get("body_parts")
    if not isinstance(body_parts, dict):
        return cfg
    for part_cfg in body_parts.values():
        if isinstance(part_cfg, dict) and part_cfg.get("type", "").startswith("OSC"):
            part_cfg["input_ref_frame"] = "world"
    return cfg


def _refresh_controller_origin(robot, arm: str) -> None:
    """Ensure controller origin pose matches the live sim before reading base-frame errors."""
    cc = getattr(robot, "composite_controller", None)
    if cc is not None and hasattr(cc, "update_state"):
        cc.update_state()


def _world_pose_in_controller_origin(
    robot,
    arm: str,
    world_pos: np.ndarray,
    world_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame EE pose in the OSC controller origin frame."""
    _refresh_controller_origin(robot, arm)
    cc = robot.composite_controller
    origin_pos, origin_ori = cc.get_controller_base_pose(controller_name=arm)
    world_pose = T.make_pose(world_pos, world_rot)
    origin_pose = T.make_pose(origin_pos, origin_ori)
    pose_in_origin = T.pose_in_A_to_pose_in_B(world_pose, T.pose_inv(origin_pose))
    pos_in_origin, quat_in_origin = T.mat2pose(pose_in_origin)
    return np.asarray(pos_in_origin, dtype=np.float64), T.quat2mat(quat_in_origin)


def _pose_error_to_controller_delta(
    robot,
    arm: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    current_pos: np.ndarray,
    current_rot: np.ndarray,
    ref_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a world-frame pose error to OSC delta commands in the requested ref frame."""
    if ref_frame == "world":
        delta_pos = target_pos - current_pos
        delta_ori = orientation_error(target_rot, current_rot)
        return delta_pos, delta_ori

    if ref_frame != "base":
        raise ValueError(f"Unsupported ref_frame: {ref_frame!r}")

    target_pos_o, target_rot_o = _world_pose_in_controller_origin(robot, arm, target_pos, target_rot)
    current_pos_o, current_rot_o = _world_pose_in_controller_origin(robot, arm, current_pos, current_rot)
    delta_pos = target_pos_o - current_pos_o
    delta_ori = orientation_error(target_rot_o, current_rot_o)
    return delta_pos, delta_ori


def playback_puma_episode(
    env,
    episode_dir: str,
    env_info: dict,
    max_fr: int | None = 20,
    realtime_from_delta_t: bool = False,
    action_ref_frame: str = "auto",
    ee_trace_controller: str = "ik",
    ik_max_iters: int = 50,
    ik_settle_steps: int = 1,
    waypoint_max_steps: int = 20,
    waypoint_pos_tol: float = 0.002,
    waypoint_ori_tol: float = 0.15,
    replay_ori_action_scale: float = 1.0,
    final_converge_steps: int = 40,
    final_orient_steps: int = 30,
    final_orient_action_scale: float = 0.15,
    waypoint_stride: int = 30,
):
    env.reset()

    # OSC scaling (used only for --ee_trace_controller osc)
    controller_configs = getattr(env, "controller_configs", None)
    out_max6 = _get_osc_output_max(controller_configs or {})

    # Resolve which arm to use (default to right if present)
    robot = env.robots[0]
    arm = "right" if "right" in robot.arms else robot.arms[0]

    ref_frame = _resolve_action_ref_frame(env_info, action_ref_frame)
    if ee_trace_controller == "ik":
        print("EE trace replay: IK -> JOINT_POSITION tracking.", flush=True)
    else:
        print("EE trace replay: OSC_POSE absolute tracking.", flush=True)

    start_marker = _PoseMarker(env, "start_position_marker", START_MARKER_RGBA, "start", START_GOAL_MARKER_SIZE)
    goal_marker = _PoseMarker(env, "goal_position_marker", GOAL_MARKER_RGBA, "goal", START_GOAL_MARKER_SIZE)
    waypoint_pool = _WaypointMarkerPool(env, WAYPOINT_MARKER_RGBA, WAYPOINT_MARKER_SIZE)

    pk_paths = _episode_pk_paths(episode_dir)
    for ep_idx, pk_path in enumerate(pk_paths):
        trace = _load_ee_trace(pk_path)
        restored_initial_state = _restore_trace_initial_state(env, trace)
        if restored_initial_state:
            print("Restored recorded initial simulator state.", flush=True)
            _refresh_controller_origin(robot, arm)
        markers = []
        start_seg = start_marker.set_pose(
            np.asarray(trace["x_pos"][0], dtype=np.float64),
            np.asarray(trace["x_rot"][0], dtype=np.float64),
        )
        goal_seg = goal_marker.set_pose(
            np.asarray(trace["x_pos"][-1], dtype=np.float64),
            np.asarray(trace["x_rot"][-1], dtype=np.float64),
        )
        if start_seg is not None:
            markers.append(start_seg)
        if goal_seg is not None:
            markers.append(goal_seg)
        markers.extend(waypoint_pool.place_waypoints(trace, waypoint_stride))
        _set_connector_markers(env, markers)
        _prepare_episode_start(
            env,
            robot,
            arm,
            trace,
            out_max6,
            ref_frame,
            max_fr,
            realtime_from_delta_t,
            ee_trace_controller=ee_trace_controller,
            ik_max_iters=ik_max_iters,
            ik_settle_steps=ik_settle_steps,
            start_pose_tol=waypoint_pos_tol,
            replay_ori_action_scale=replay_ori_action_scale,
        )

        print(f"Replaying Episode: {ep_idx}")
        for (tpos, trot, dt, gripper_ac) in _iter_trace_steps(trace):
            step_start = time.time()
            if ee_trace_controller == "ik":
                _replay_trace_waypoint_ik(
                    env,
                    robot,
                    arm,
                    tpos,
                    trot,
                    gripper_ac,
                    ik_max_iters=ik_max_iters,
                    ik_settle_steps=ik_settle_steps,
                )
            else:
                for _ in range(waypoint_max_steps):
                    cur_pos = np.asarray(env.sim.data.site_xpos[robot.eef_site_id[arm]], dtype=np.float64)
                    if np.linalg.norm(np.asarray(tpos, dtype=np.float64) - cur_pos) < waypoint_pos_tol:
                        break
                    action = _attach_gripper_to_arm_action(
                        _rotmat_to_abs_osc_action(tpos, trot),
                        env.action_dim,
                        gripper_ac,
                    )
                    env.step(action)
                    _maybe_render(env)
            if realtime_from_delta_t:
                sleep_s = max(0.0, dt - (time.time() - step_start))
                if sleep_s > 0:
                    time.sleep(sleep_s)
            elif max_fr is not None and max_fr > 0:
                elapsed = time.time() - step_start
                diff = 1.0 / float(max_fr) - elapsed
                if diff > 0:
                    time.sleep(diff)

        # Final convergence: lock endpoint position first, then optionally trim orientation.
        last_tpos = np.asarray(trace["x_pos"][-1], dtype=np.float64)
        last_trot = np.asarray(trace["x_rot"][-1], dtype=np.float64)
        last_ga = None
        gripper_action = trace.get("gripper_action", None)
        if gripper_action is not None:
            last_ga = np.asarray(gripper_action[-1], dtype=np.float64)

        if ee_trace_controller == "ik":
            _replay_trace_waypoint_ik(
                env,
                robot,
                arm,
                last_tpos,
                last_trot,
                last_ga,
                ik_max_iters=max(ik_max_iters, 80),
                ik_settle_steps=max(ik_settle_steps, 3),
            )
        else:
            pos_converge_steps = max(1, final_converge_steps // 2)
            _reach_pose(
                env,
                robot,
                arm,
                last_tpos,
                last_trot,
                out_max6,
                ref_frame,
                max_fr,
                realtime_from_delta_t,
                pos_tol=waypoint_pos_tol,
                ori_tol=waypoint_ori_tol,
                max_steps=pos_converge_steps,
                gripper_action=last_ga,
                ori_action_scale=replay_ori_action_scale,
                require_orientation=True,
            )
            if final_orient_steps > 0:
                _reach_pose(
                    env,
                    robot,
                    arm,
                    last_tpos,
                    last_trot,
                    out_max6,
                    ref_frame,
                    max_fr,
                    realtime_from_delta_t,
                    pos_tol=max(waypoint_pos_tol * 2.0, 0.02),
                    ori_tol=0.12,
                    max_steps=final_orient_steps,
                    gripper_action=last_ga,
                    ori_action_scale=final_orient_action_scale,
                    require_orientation=True,
                )

        if ep_idx + 1 < len(pk_paths):
            _go_to_homepose(env)

    env.close()


def _hdf5_demo_keys(data_group) -> list[str]:
    return sorted(
        data_group.keys(),
        key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") and k.split("_")[-1].isdigit() else k,
    )


def _trace_delta_t_for_demo(pk_paths: list[str], demo_index: int) -> np.ndarray | None:
    if demo_index >= len(pk_paths):
        return None
    trace = _load_ee_trace(pk_paths[demo_index])
    return np.asarray(trace.get("delta_t", []), dtype=np.float64)


def playback_hdf5_states_episode(
    env,
    episode_dir: str,
    max_fr: int | None = 20,
    realtime_from_delta_t: bool = False,
):
    """Exact visual playback by setting flattened MuJoCo states from demo.hdf5."""
    import h5py

    hdf5_path = _episode_hdf5_path(episode_dir)
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Missing demo.hdf5 at: {hdf5_path}")

    pk_paths = _episode_pk_paths(episode_dir)
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        for demo_index, demo_key in enumerate(_hdf5_demo_keys(data_group)):
            states = np.asarray(data_group[demo_key]["states"])
            delta_t = _trace_delta_t_for_demo(pk_paths, demo_index)
            print(f"Replaying {demo_key} from demo.hdf5 states")
            env.reset()
            for i, state in enumerate(states):
                start = time.time()
                env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
                env.sim.forward()
                _maybe_render(env)
                dt = float(delta_t[i]) if delta_t is not None and i < len(delta_t) else 0.0
                _sleep_for_replay_step(start, dt, max_fr, realtime_from_delta_t)

    env.close()


def playback_hdf5_actions_episode(
    env,
    episode_dir: str,
    max_fr: int | None = 20,
    realtime_from_delta_t: bool = False,
):
    """
    Replay the original collection-controller actions from demo.hdf5.

    For leader-arm demonstrations this preserves the JOINT_POSITION commands
    produced during collection instead of approximating the path with OSC.
    """
    import h5py

    hdf5_path = _episode_hdf5_path(episode_dir)
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Missing demo.hdf5 at: {hdf5_path}")

    pk_paths = _episode_pk_paths(episode_dir)
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        for demo_index, demo_key in enumerate(_hdf5_demo_keys(data_group)):
            actions = np.asarray(data_group[demo_key]["actions"])
            trace = _load_ee_trace(pk_paths[demo_index]) if demo_index < len(pk_paths) else {}
            if not _restore_trace_initial_state(env, trace):
                raise ValueError(
                    f"{demo_key} action replay needs the matching ee_state_*.pk initial_state. "
                    "Use --replay_source hdf5_states for state-only playback."
                )
            delta_t = np.asarray(trace.get("delta_t", []), dtype=np.float64)
            print(f"Replaying {demo_key} from demo.hdf5 actions")
            for i, action in enumerate(actions):
                start = time.time()
                env.step(np.asarray(action, dtype=np.float64))
                _maybe_render(env)
                dt = float(delta_t[i]) if i < len(delta_t) else 0.0
                _sleep_for_replay_step(start, dt, max_fr, realtime_from_delta_t)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True, help="Directory containing env_info.json + ee_state_*.pk")
    parser.add_argument(
        "--replay_source",
        choices=["auto", "ee_trace", "hdf5_actions", "hdf5_states"],
        default="ee_trace",
        help=(
            "ee_trace (default) tracks recorded x_pos/x_rot with OSC_POSE; "
            "auto prefers demo.hdf5 action replay when available; "
            "hdf5_actions / hdf5_states replay demo.hdf5 directly."
        ),
    )
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
        help="Frame for OSC delta commands. 'auto' defaults to world (PUMA traces are world-frame poses).",
    )
    parser.add_argument("--render_camera", type=str, default="frontview", help="Camera name for onscreen renderer")
    parser.add_argument(
        "--ee_trace_controller",
        choices=["ik", "osc"],
        default="ik",
        help=(
            "How to track recorded x_pos/x_rot during ee_trace replay. "
            "ik (default) solves damped least-squares IK and commands JOINT_POSITION; "
            "osc uses absolute OSC_POSE goals."
        ),
    )
    parser.add_argument(
        "--ik_max_iters",
        type=int,
        default=50,
        help="Max damped-least-squares IK iterations per recorded waypoint (ee_trace_controller=ik)",
    )
    parser.add_argument(
        "--ik_settle_steps",
        type=int,
        default=1,
        help="JOINT_POSITION steps to hold each IK joint target (ee_trace_controller=ik)",
    )
    parser.add_argument(
        "--osc_kp",
        type=int,
        default=500,
        help="OSC position gain used when --ee_trace_controller osc",
    )
    parser.add_argument(
        "--waypoint_max_steps",
        type=int,
        default=20,
        help="Max OSC steps to spend on each recorded waypoint (ee_trace_controller=osc only)",
    )
    parser.add_argument(
        "--waypoint_pos_tol",
        type=float,
        default=0.002,
        help="Position tolerance (m) for ee_trace replay convergence checks",
    )
    parser.add_argument(
        "--replay_ori_action_scale",
        type=float,
        default=1.0,
        help="Orientation delta scale for legacy OSC delta helpers (ee_trace_controller=osc)",
    )
    parser.add_argument(
        "--final_converge_steps",
        type=int,
        default=40,
        help="OSC steps to converge on the trajectory endpoint position after replay",
    )
    parser.add_argument(
        "--final_orient_steps",
        type=int,
        default=30,
        help="Optional OSC steps to trim endpoint orientation after position convergence (0 to skip)",
    )
    parser.add_argument(
        "--final_orient_action_scale",
        type=float,
        default=0.15,
        help="Orientation delta scale during the optional endpoint orientation trim",
    )
    parser.add_argument(
        "--waypoint_stride",
        type=int,
        default=30,
        help="Draw a green orientation cylinder every N recorded steps along the trajectory",
    )
    args = parser.parse_args()

    env_info_path = os.path.join(args.episode_dir, "env_info.json")
    if not os.path.exists(env_info_path):
        raise FileNotFoundError(f"Missing env_info.json at: {env_info_path}")

    env_info = _load_env_info(env_info_path)
    env_name = env_info.get("env_name", "Lift")
    robots = env_info.get("robots", ["Panda"])
    replay_source = args.replay_source
    if replay_source == "auto":
        replay_source = "hdf5_actions" if os.path.exists(_episode_hdf5_path(args.episode_dir)) else "ee_trace"

    ref_frame = _resolve_action_ref_frame(env_info, args.action_ref_frame)
    if replay_source == "ee_trace":
        controller_configs = _controller_configs_for_playback(
            env_info,
            robots,
            ref_frame,
            ee_trace_controller=args.ee_trace_controller,
            osc_kp=args.osc_kp,
        )
    else:
        controller_configs = copy.deepcopy(env_info.get("controller_configs", None))

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

    if replay_source == "hdf5_actions":
        playback_hdf5_actions_episode(
            env,
            args.episode_dir,
            max_fr=args.max_fr,
            realtime_from_delta_t=args.realtime,
        )
    elif replay_source == "hdf5_states":
        playback_hdf5_states_episode(
            env,
            args.episode_dir,
            max_fr=args.max_fr,
            realtime_from_delta_t=args.realtime,
        )
    else:
        playback_puma_episode(
            env,
            args.episode_dir,
            env_info=env_info,
            max_fr=args.max_fr,
            realtime_from_delta_t=args.realtime,
            action_ref_frame=args.action_ref_frame,
            ee_trace_controller=args.ee_trace_controller,
            ik_max_iters=args.ik_max_iters,
            ik_settle_steps=args.ik_settle_steps,
            waypoint_max_steps=args.waypoint_max_steps,
            waypoint_pos_tol=args.waypoint_pos_tol,
            replay_ori_action_scale=args.replay_ori_action_scale,
            final_converge_steps=args.final_converge_steps,
            final_orient_steps=args.final_orient_steps,
            final_orient_action_scale=args.final_orient_action_scale,
            waypoint_stride=args.waypoint_stride,
        )

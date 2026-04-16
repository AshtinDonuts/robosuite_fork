"""
demo_fabric_guided_osc.py
=========================
Demonstrates the FabricGuidedOSCController in a Lift-like environment
that contains a single fixed red sphere as the fabric obstacle.

The policy sends a constant +x delta action so the EEF moves toward the
sphere.  The fabrics avoidance layer deflects the arm, keeping a safe
distance from the sphere.

Run:
    cd /home/khw/Projects/robosuite-master
    python -m robosuite.demos.demo_fabric_guided_osc

Dependencies:
    pumafabrics must be installed:
        pip install -e /home/khw/Projects/pumafabrics
    robosuite must be installed in editable mode:
        pip install -e /home/khw/Projects/robosuite-master
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.controllers.parts.arm.fabric_guided_osc import PANDA_FABRICS_PARAMS


# ---------------------------------------------------------------------------
# Obstacle parameters (fixed in world frame)
# ---------------------------------------------------------------------------
OBSTACLE_POS    = np.array([0.20, 0.0, 1.3])   # [0.50, 0.0, 1.15]
OBSTACLE_RADIUS = 0.02                           # [m]
SCALE_ACTION = 0.1


# ---------------------------------------------------------------------------
# Environment: Lift + one fixed red sphere
# ---------------------------------------------------------------------------
class LiftWithObstacle(Lift):
    """
    Standard Lift task augmented with a fixed spherical obstacle.

    The sphere is added to the existing ManipulationTask model after Lift
    builds it.  Using ``joints=[]`` means no free joint → body is welded at
    its MJCF ``pos``, so its world position never changes and no placement
    initializer update is needed.
    """

    def _load_model(self):
        # Build the standard Lift model (arena, robot, cube)
        super()._load_model()

        # Create the obstacle sphere: fixed in place, red, semi-transparent
        self.obstacle_sphere = BallObject(
            name="obstacle_sphere",
            size=[OBSTACLE_RADIUS],
            rgba=[1.0, 0.0, 0.0, 0.4],
            joints=[],          # no joint → body welded at OBSTACLE_POS
            obj_type="all",     # render collision + visual geoms
        )
        # Set the MJCF body position before merging into the compiled model.
        # BallObject (PrimitiveObject) has no set_pos; manipulate the XML element directly.
        self.obstacle_sphere.get_obj().set("pos", array_to_string(OBSTACLE_POS.tolist()))

        # Append obstacle assets (none in practice) and body to the task model
        self.model.merge_assets(self.obstacle_sphere)
        self.model.worldbody.append(self.obstacle_sphere.get_obj())

    def _setup_references(self):
        super()._setup_references()
        # Cache the MuJoCo body ID for fast runtime position lookup
        self.obstacle_body_id = self.sim.model.body_name2id(
            self.obstacle_sphere.root_body
        )

    def get_obstacle_pos_radius(self):
        """Live callback: returns (world_pos, radius) for the obstacle."""
        pos = self.sim.data.body_xpos[self.obstacle_body_id].copy()
        return pos, OBSTACLE_RADIUS


# ---------------------------------------------------------------------------
# Controller configuration
# ---------------------------------------------------------------------------
def make_controller_config():
    """
    Full composite controller config: FABRIC_OSC_POSE arm + GRIP gripper.
    The ``fabrics_params`` key is a nested dict that passes through the
    factory to ``FabricGuidedOSCController.__init__``.
    """
    arm_cfg = {
        "type": "FABRIC_OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "input_type": "delta",
        "input_ref_frame": "base",
        "interpolation": None,
        "ramp_ratio": 0.2,
        # Gripper config is nested inside the arm config (robosuite convention)
        "gripper": {"type": "GRIP"},
        # Fabrics-specific config (nested dict, passed through to FabricGuidedOSCController)
        "fabrics_params": PANDA_FABRICS_PARAMS,
    }
    # When passing a Python dict directly (not loading from JSON), "right" must
    # be a top-level key inside "body_parts".  The "arms" sub-key is a JSON
    # convention that load_composite_controller_config() flattens automatically.
    return {
        "type": "BASIC",
        "body_parts": {"right": arm_cfg},
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main(n_steps: int = 500, render: bool = True):
    print("=" * 65)
    print("Fabric-Guided OSC Demo")
    print(f"  Obstacle : sphere at {OBSTACLE_POS},  radius={OBSTACLE_RADIUS} m")
    print(f"  Policy   : constant +x delta  (drives EEF toward obstacle)")
    print(f"  Fabrics  : repulsive avoidance added on top of OSC torques")
    print("=" * 65)

    env = LiftWithObstacle(
        robots="Panda",
        controller_configs=make_controller_config(),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    obs = env.reset()

    # The controller is constructed before env.reset() populates body IDs,
    # so we wire the live obstacle callback in afterwards.
    arm_ctrl = env.robots[0].composite_controller.part_controllers["right"]
    arm_ctrl.obstacle_fn = env.get_obstacle_pos_radius

    print(f"\nFabrics planner active: {arm_ctrl._fabrics_ready}")

    # Constant policy: push EEF in +x direction; zero all other dims + gripper.
    # 0.2 → scaled to 0.2 * output_max[0] = 0.01 m per policy step (slow approach)
    action = np.zeros(env.action_dim)
    action[0] = 1.0 * SCALE_ACTION

    eef_positions  = []
    avoid_norms    = []

    for step in range(n_steps):
        obs, reward, done, _ = env.step(action)

        if render:
            env.render()

        # EEF position from observation
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_positions.append(eef_pos.copy())

        # Measure fabric correction magnitude (computed from current state)
        if arm_ctrl._fabrics_ready and arm_ctrl.obstacle_fn is not None:
            try:
                tau_fabric = arm_ctrl._compute_fabric_torques(*env.get_obstacle_pos_radius())
                avoid_norms.append(float(np.linalg.norm(tau_fabric)))
            except Exception:
                avoid_norms.append(0.0)

        if step % 50 == 0:
            dist_eef = np.linalg.norm(eef_pos - OBSTACLE_POS)
            avg_avoid = np.mean(avoid_norms[-50:]) if avoid_norms else 0.0

            # Per-link distances: the fabrics planner repels each collision link,
            # not the EEF.  The closest link drives the repulsion, so we report
            # all of them to see which one is actually near the sphere.
            # Robosuite bodies are named "robot0_link3", while fabrics params
            # use "panda_link3" – strip "panda_" and prepend "robot0_".
            link_dists = {}
            for link in arm_ctrl._fabrics_params.get("collision_links", []):
                try:
                    sim_name = "robot0_" + link.replace("panda_", "")
                    body_id = env.sim.model.body_name2id(sim_name)
                    link_pos = env.sim.data.body_xpos[body_id]
                    link_dists[link] = float(np.linalg.norm(link_pos - OBSTACLE_POS))
                except Exception:
                    pass

            closest = min(link_dists, key=link_dists.get) if link_dists else "?"
            min_link_dist = link_dists.get(closest, float("nan"))
            link_str = "  ".join(f"{k[-1]}:{v:.2f}" for k, v in link_dists.items())

            print(
                f"step {step:4d} | EEF {np.round(eef_pos, 3)} "
                f"| EEF→sphere: {dist_eef:.3f} m "
                f"| closest link: {closest}({min_link_dist:.3f} m) "
                f"| links [{link_str}] "
                f"| |τ_fabric| avg: {avg_avoid:.2f} Nm"
            )

        if done:
            obs = env.reset()
            arm_ctrl.obstacle_fn = env.get_obstacle_pos_radius

    env.close()

    # ---- Summary -------------------------------------------------------
    print("\n--- Summary ---")
    if avoid_norms:
        print(f"Mean |τ_fabric| : {np.mean(avoid_norms):.3f} Nm")
        print(f"Max  |τ_fabric| : {np.max(avoid_norms):.3f} Nm")
    if eef_positions:
        min_dist = min(np.linalg.norm(p - OBSTACLE_POS) for p in eef_positions)
        print(f"Min EEF→obstacle dist : {min_dist:.3f} m  (radius = {OBSTACLE_RADIUS} m)")
        if min_dist < OBSTACLE_RADIUS:
            print("  [WARN] EEF entered sphere – consider increasing kp_fabrics or reducing action magnitude.")
        else:
            print("  [OK] EEF stayed outside obstacle boundary.")


if __name__ == "__main__":
    main()

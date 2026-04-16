"""
Fabric-Guided OSC Controller
=============================
Wraps OperationalSpaceController with a Fabrics avoidance layer from the
pumafabrics library (https://github.com/tud-amr/pumafabrics).

At every call to run_controller():
  1. The base OSC policy computes joint torques tau_osc as normal.
  2. The fabrics avoidance planner computes a joint-space repulsion
     acceleration `a_avoid` given the current joint state and a live
     obstacle position/radius callback.
  3. The final torques are: tau = tau_osc + M @ a_avoid
     where M is the MuJoCo mass matrix.  This is the "FPM" (Fabric
     Potential Method) simple sum, equivalent to the fallback path in
     the pumafabrics kuka_TamedPUMA example.

The fabrics planner is compiled once at __init__ via CasADi; it adds a
few seconds to the first env.reset() but negligible overhead per step.

Usage
-----
Pass the controller type ``"FABRIC_OSC_POSE"`` in the composite
controller config.  The optional ``fabrics_params`` key (nested dict)
overrides the built-in Panda defaults.  Set ``obstacle_fn`` on the
controller instance after env.reset() to wire in a live obstacle source:

    arm_ctrl = env.robots[0].composite_controller.part_controllers["right"]
    arm_ctrl.obstacle_fn = lambda: (obstacle_pos_xyz, obstacle_radius)
"""

import logging

import numpy as np

from robosuite.controllers.parts.arm.osc import OperationalSpaceController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Panda fabrics configuration
# Matches panda_collision_links.urdf inside pumafabrics (links 0-8, 7 DOF).
# ---------------------------------------------------------------------------
PANDA_FABRICS_PARAMS = {
    # Robot kinematics
    "dof": 7,
    "robot_name": "panda_collision_links",   # URDF file stem in pumafabrics/config/urdfs/
    "root_link": "panda_link0",
    "end_links": ["panda_link8"],
    # Links whose collision spheres participate in avoidance
    "collision_links": [
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
    ],
    # Sphere radii [m] for each collision link
    "collision_radii": {
        "panda_link3": 0.08,
        "panda_link4": 0.08,
        "panda_link5": 0.08,
        "panda_link6": 0.06,
        "panda_link7": 0.06,
    },
    # Joint limits [rad] - from panda_collision_links.urdf
    # Key name "iiwa_limits" is expected by FabricsController.set_planner()
    "iiwa_limits": [
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ],
    # Obstacle count fed to the planner (number of x_obst_i symbols created)
    "nr_obst": 1,
    # Simulation timestep [s] – used by fabrics speed-control term
    "dt": 0.002,
    # Fabrics output mode: "acc" → returns joint accelerations [rad/s²]
    "mode": "acc",
    # Fabrics tuning (CasADi symbolic expressions)
    "collision_geometry": "-0.01 / (x ** 1) * xdot ** 2",
    "collision_finsler": "0.01/(x**2) * xdot**2",
    # Planner options
    "bool_speed_control": True,
    "bool_extensive_concretize": True,
}


class FabricGuidedOSCController(OperationalSpaceController):
    """
    OSC controller augmented with a fabrics-based obstacle avoidance layer.

    The avoidance planner from pumafabrics is built once at construction.
    Obstacle information is supplied via ``obstacle_fn``, a callable that
    returns ``(position: np.ndarray[3], radius: float)`` each control step.
    When ``obstacle_fn`` is ``None`` the controller behaves identically to a
    plain ``OperationalSpaceController``.

    Parameters
    ----------
    fabrics_params : dict, optional
        Configuration passed to ``FabricsController``.  Defaults to
        ``PANDA_FABRICS_PARAMS`` (Franka Panda, 7 DOF, 1 sphere obstacle).
    obstacle_fn : callable, optional
        Zero-argument callable returning ``(np.ndarray[3], float)``
        – the obstacle centre in world coordinates and its radius.
        May be set or replaced after construction.
    **kwargs
        All remaining kwargs are forwarded to ``OperationalSpaceController``.
    """

    def __init__(self, fabrics_params=None, obstacle_fn=None, **kwargs):
        super().__init__(**kwargs)

        self._fabrics_params = fabrics_params if fabrics_params is not None else PANDA_FABRICS_PARAMS.copy()
        self._obstacle_fn = obstacle_fn

        self._avoidance_planner = None
        self._fabrics_ready = False
        self._build_fabrics_planner()

    # ------------------------------------------------------------------
    # Public attribute: allow replacing obstacle_fn at any time
    # ------------------------------------------------------------------
    @property
    def obstacle_fn(self):
        return self._obstacle_fn

    @obstacle_fn.setter
    def obstacle_fn(self, fn):
        self._obstacle_fn = fn

    # ------------------------------------------------------------------
    # Planner construction (called once at init)
    # ------------------------------------------------------------------
    def _build_fabrics_planner(self):
        try:
            from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
        except ImportError:
            logger.warning(
                "pumafabrics not found – FabricGuidedOSCController will "
                "run as plain OSC without avoidance."
            )
            return

        logger.info("Building fabrics avoidance planner (CasADi compilation)…")
        try:
            fc = FabricsController(self._fabrics_params)
            self._avoidance_planner, _ = fc.set_avoidance_planner(goal=None)
            self._fabrics_ready = True
            logger.info("Fabrics planner ready.")
        except Exception as exc:
            logger.warning("Fabrics planner build failed (%s); running as plain OSC.", exc)

    # ------------------------------------------------------------------
    # Core control loop
    # ------------------------------------------------------------------
    def run_controller(self):
        """
        Compute OSC torques and add the fabrics avoidance correction.

        Returns
        -------
        np.ndarray
            Joint torques (same shape as plain OSC).
        """
        # --- Base OSC torques ----------------------------------------
        tau_osc = super().run_controller()

        # --- Fabric correction (skip if planner unavailable/no obstacle) -
        if not self._fabrics_ready or self._obstacle_fn is None:
            return tau_osc

        try:
            obs_pos, obs_radius = self._obstacle_fn()
            tau_fabric = self._compute_fabric_torques(obs_pos, float(obs_radius))
            self.torques = tau_osc + tau_fabric
        except Exception as exc:
            logger.debug("Fabrics step failed (%s); using plain OSC torques.", exc)

        return self.torques

    def _compute_fabric_torques(self, obs_pos: np.ndarray, obs_radius: float) -> np.ndarray:
        """
        Run the avoidance planner and convert its joint-acceleration output
        to joint torques via the inertia matrix.

        Parameters
        ----------
        obs_pos : np.ndarray, shape (3,)
            Obstacle centre in world coordinates.
        obs_radius : float
            Obstacle radius [m].

        Returns
        -------
        np.ndarray
            Corrective joint torques, same DOF as the arm controller.
        """
        params = self._fabrics_params
        q = self.joint_pos          # shape (dof,)
        qdot = self.joint_vel       # shape (dof,)

        arguments = {
            "q": q,
            "qdot": qdot,
            "x_obst_0": obs_pos,
            "radius_obst_0": obs_radius,
        }
        # Per-link body radii expected by the planner
        for link, radius in params["collision_radii"].items():
            arguments[f"radius_body_{link}"] = radius

        # compute_M_f_action_avoidance returns (M, f, action_avoidance, xddot_speed)
        _, _, a_avoid, _ = self._avoidance_planner.compute_M_f_action_avoidance(**arguments)
        a_avoid = np.array(a_avoid).flatten()

        # tau = M * a  (same mapping as JointPositionController's feedforward)
        tau_fabric = self.mass_matrix @ a_avoid
        return tau_fabric

    @property
    def name(self):
        return "FABRIC_OSC_POSE"

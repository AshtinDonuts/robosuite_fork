from collections import OrderedDict

import mujoco
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject, PlaneSurfaceObject
from robosuite.models.objects.xml_objects import PLANE_SURFACE_XML
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, make_pose, mat2quat, quat2mat


def _scale_wiping_gripper(root, sx, sy, sz):
    """
    Non-uniformly scale every geometric element inside a WipingGripper XML tree.

    The helper rescales in the body-local coordinate frame of the gripper pad.
    Positions of bodies, geoms and sites are multiplied component-wise bys
    (sx, sy, sz).  Sizes are rescaled according to geometry type:

        box       – each half-extent is multiplied by the matching scale factor
        sphere    – radius is multiplied by the cube-root of sx*sy*sz
        capsule   – radius by sqrt(sx*sy), half-length by sz
        cylinder  – same as capsule
        plane     – half-widths scaled by sx and sy; spacing by sz

    Args:
        root: XML Element that is the root of the gripper worldbody tree
              (e.g. ``self.robots[0].gripper.worldbody``).
        sx (float): scale along the gripper's local X axis.
        sy (float): scale along the gripper's local Y axis.
        sz (float): scale along the gripper's local Z axis.
    """
    scale = np.array([sx, sy, sz], dtype=float)
    sphere_scale = float((sx * sy * sz) ** (1.0 / 3.0))
    radial_scale = float(np.sqrt(sx * sy))

    def _rescale_pos(elem):
        pos = elem.get("pos")
        if pos:
            p = np.array([float(v) for v in pos.split()])
            if len(p) == 3:
                elem.set("pos", " ".join(f"{v:.8g}" for v in p * scale))

    def _rescale_size(elem):
        raw = elem.get("size")
        if not raw:
            return
        parts = [float(v) for v in raw.split()]
        gtype = elem.get("type", "sphere").lower()
        if gtype == "box" and len(parts) == 3:
            new_parts = [parts[0] * sx, parts[1] * sy, parts[2] * sz]
        elif gtype == "sphere" and len(parts) == 1:
            new_parts = [parts[0] * sphere_scale]
        elif gtype in ("capsule", "cylinder") and len(parts) == 2:
            new_parts = [parts[0] * radial_scale, parts[1] * sz]
        elif gtype == "plane" and len(parts) == 3:
            new_parts = [parts[0] * sx, parts[1] * sy, parts[2] * sz]
        else:
            new_parts = [v * sphere_scale for v in parts]
        elem.set("size", " ".join(f"{v:.8g}" for v in new_parts))

    for geom in root.iter("geom"):
        _rescale_pos(geom)
        _rescale_size(geom)

    for body in root.iter("body"):
        _rescale_pos(body)

    for site in root.iter("site"):
        _rescale_pos(site)
        _rescale_size(site)


class PlaneFollow(ManipulationEnv):
    """
    Tabletop environment with a simple mesh workpiece (flat box, tilted box, or curved surface). There is no task completion signal based on lifting height:
    episodes end only at the horizon (unless ``ignore_done`` is set). Optional dense shaping rewards encourage
    reaching and contacting the workpiece.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper. Must be ``WipingGripper`` or
            ``WipingGripperVX300S`` (the flat-pad force-sensing tool used by the Wipe environment).
            For ``VX300S`` robots ``WipingGripper`` is automatically promoted to ``WipingGripperVX300S``.
            Should either be a single str or a list of the same length as "robots" param.

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        table_offset (3-tuple): (x, y, z) offset for `TableArena` placement; z sets the tabletop height.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include workpiece information in
            the observation.

        reward_scale (None or float): Scales the reward by this factor when not None.

        reward_shaping (bool): if True, use dense rewards (reaching and contact only).

        surface_type (str): Workpiece mesh variant: ``"flat"``, ``"tilted"``, ``"curved"`` (convex
            arc along Y), ``"curved_inverse"`` (concave trough along Y), or ``"curved_wave"``
            (single-period sine: crest then trough along Y).

        surface_scale (float or 3-tuple): Uniform or per-axis scale applied to the workpiece mesh.

        wiping_gripper_scale (3-tuple): (sx, sy, sz) scale factors for the WipingGripper pad
            in its local coordinate frame. (1, 1, 1) leaves the gripper unchanged.

        surface_static (bool): If True, the workpiece is welded to the world (no free joint): it does not
            move when pushed, but collision geoms, friction, and contact forces still apply.

        surface_friction (None or float or 3-tuple): MuJoCo geom friction ``(sliding, torsional, rolling)``
            applied to every workpiece collision geom. If None, friction values from ``objects/plane_*.xml`` are kept.
            A scalar sets sliding friction only; torsional and rolling use ``5e-3`` and ``1e-4`` (same secondary
            values as the default table friction tuple).

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="WipingGripper",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.5, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.15, 0.0, 0.9),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        surface_type="flat",  # flat | tilted | curved | curved_inverse | curved_wave
        surface_scale=(1.0, 1.0, 1.0),
        # surface_scale is now exposed as either:
        # Uniform: a scalar, e.g. surface_scale=1.5
        # Non-uniform: a 3-vector (sx, sy, sz), e.g. surface_scale=(1.0, 1.0, 1.0)
        wiping_gripper_scale=(0.25, 0.5, 3.0),  # @USER TODO
        surface_static=True, # @USER TODO
        surface_friction=0.01,  #@USER TODO
        # Default: (0.95 0.3 0.1)
        # Scalar - sets the sliding friction, and sets torsion = 5e-3 , rolling = 1e-4
        # Tuple  - sets (sliding, torsional, rolling)
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # Resolve WipingGripper variant exactly as Wipe does (auto-upgrade for VX300S)
        _wipe_grippers = frozenset({"WipingGripper", "WipingGripperVX300S"})
        robots_list = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        if isinstance(gripper_types, str):
            gt = gripper_types
            primary = str(robots_list[0]).upper()
            if gt == "WipingGripper" and primary == "VX300S":
                gt = "WipingGripperVX300S"
            assert gt in _wipe_grippers, "PlaneFollow only supports WipingGripper / WipingGripperVX300S."
            assert not (gt == "WipingGripperVX300S" and primary != "VX300S"), (
                "WipingGripperVX300S is only valid for robot VX300S."
            )
            gripper_types = gt
        else:
            assert len(gripper_types) == len(robots_list), "gripper_types must match robots length."
            resolved = []
            for rname, gt_in in zip(robots_list, gripper_types):
                primary = str(rname).upper()
                gt = gt_in
                if gt == "WipingGripper" and primary == "VX300S":
                    gt = "WipingGripperVX300S"
                assert gt in _wipe_grippers, "PlaneFollow only supports WipingGripper / WipingGripperVX300S."
                assert not (gt == "WipingGripperVX300S" and primary != "VX300S"), (
                    "WipingGripperVX300S is only valid for robot VX300S."
                )
                resolved.append(gt)
            gripper_types = resolved

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset, dtype=float)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.surface_type = str(surface_type).lower()
        assert self.surface_type in PLANE_SURFACE_XML, (
            f"surface_type must be one of {tuple(PLANE_SURFACE_XML)}; got {surface_type!r}"
        )

        # workpiece scale factor (uniform scalar or xyz 3-vector)
        self.surface_scale = surface_scale

        # WipingGripper scale (sx, sy, sz) in pad-local coordinates
        self.wiping_gripper_scale = tuple(wiping_gripper_scale)

        self.surface_static = bool(surface_static)

        if surface_friction is None:
            self._surface_friction = None
        elif isinstance(surface_friction, (int, float, np.floating, np.integer)):
            f = float(surface_friction)
            self._surface_friction = (f, 5e-3, 1e-4)
        else:
            bf = np.array(surface_friction, dtype=float).reshape(-1)
            assert bf.size == 3, f"surface_friction must be None, a scalar, or length-3; got {surface_friction!r}"
            self._surface_friction = tuple(bf.tolist())

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

        # Force/torque bias at the initial state (populated on first step, like Wipe)
        self.ee_force_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}
        self.ee_torque_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}

    def reward(self, action=None):
        """
        Reward function for the task. There is no sparse completion reward (no height-based success).

        If ``reward_shaping`` is True, returns a dense reward with:
            - Reaching: in [0, 1], to encourage the arm to reach the workpiece
            - Contact: in {0, 0.25}, non-zero if the gripper has contact force on the workpiece

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        if self.reward_shaping:
            surface_pos = np.array(self.sim.data.body_xpos[self.surface_body_id])
            eef_pos = self._get_eef_xpos(self.robots[0].arms[0])
            dist = np.linalg.norm(eef_pos - surface_pos)
            reward += 1 - np.tanh(10.0 * dist)

            if self._has_gripper_contact:
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Scale the WipingGripper before merging into ManipulationTask.
        # self.robots[0].gripper is a dict {arm_name: GripperModel}.
        sx, sy, sz = self.wiping_gripper_scale
        if not np.allclose([sx, sy, sz], [1.0, 1.0, 1.0]):
            for gripper in self.robots[0].gripper.values():
                _scale_wiping_gripper(gripper.worldbody, sx, sy, sz)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        xml_path = PLANE_SURFACE_XML[self.surface_type]
        if self.surface_static:
            self.surface = MujocoXMLObject(
                xml_path_completion(xml_path),
                name="surface",
                joints=None,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
        else:
            self.surface = PlaneSurfaceObject(name="surface", surface_type=self.surface_type)
        if self.surface_scale is not None:
            # Allow both uniform and non-uniform scaling
            if isinstance(self.surface_scale, (int, float, np.floating, np.integer)):
                if float(self.surface_scale) != 1.0:
                    self.surface.set_scale(float(self.surface_scale))
            else:
                scale = np.array(self.surface_scale, dtype=float).reshape(-1)
                assert scale.size == 3, f"surface_scale must be a scalar or length-3 (x,y,z); got {self.surface_scale}"
                if not np.allclose(scale, np.ones(3)):
                    self.surface.set_scale(scale.tolist())

        if self._surface_friction is not None:
            friction_str = array_to_string(np.array(self._surface_friction, dtype=float))
            for geom in self.surface.worldbody.iter("geom"):
                ct = int(geom.get("contype", "1"))
                ca = int(geom.get("conaffinity", "1"))
                if ct == 0 and ca == 0:
                    continue
                geom.set("friction", friction_str)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.surface)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.surface,
                x_range=[-0.1, 0.1],
                y_range=[-0.03, 0.03],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.015,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.surface,
        )

        # The WipingGripper surface geoms were tuned for sliding on a flat table:
        # they use solimp="0.2 0.9 0.01" (10 mm penetration zone) and solmix=10000
        # which makes the gripper's soft parameters dominate contact mixing, causing
        # the pad to clip through curved workpiece edges.  Override them here to use a
        # 1 mm zone with near-rigid impedance and neutral mixing so workpiece mesh
        # contact parameters contribute equally.
        for geom in self.model.worldbody.iter("geom"):
            name = geom.get("name", "")
            if "wiping_surface" in name or "wiping_corner" in name:
                geom.set("solimp", "0.9 0.99 0.001")
                geom.set("solmix", "1")
                geom.set("solref", "0.02 1")

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.surface_body_id = self.sim.model.body_name2id(self.surface.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            @sensor(modality=modality)
            def surface_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.surface_body_id])

            @sensor(modality=modality)
            def surface_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.surface_body_id]), to="xyzw")

            sensors = [surface_pos, surface_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, "surface_pos", f"{arm_pf}gripper_to_surface_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset EEF force/torque biases
        self.ee_force_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}
        self.ee_torque_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            forward_needed = False
            for obj_pos, obj_quat, obj in object_placements.values():
                if len(obj.joints) == 0:
                    bid = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[bid] = np.array(obj_pos)
                    self.sim.model.body_quat[bid] = np.array(obj_quat)
                    forward_needed = True
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            if forward_needed:
                self.sim.forward()

    def _get_eef_xpos(self, arm):
        """
        Returns the end-effector position for the given arm, read from the EEF site (same as Wipe).

        Args:
            arm (str): Arm name

        Returns:
            np.array: End-effector (x, y, z) position
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])

    @property
    def _has_gripper_contact(self):
        """
        True if any gripper EEF force exceeds the contact threshold (mirrors Wipe implementation).

        Returns:
            bool: True if contact force exceeds threshold
        """
        contact_threshold = 1.0
        return any(
            np.linalg.norm(self.robots[0].ee_force[arm] - self.ee_force_bias[arm]) > contact_threshold
            for arm in self.robots[0].arms
        )

    def _post_action(self, action):
        """
        In addition to the super method, update the EEF force/torque bias on the first step.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple: (reward, done, info)
        """
        reward, done, info = super()._post_action(action)

        if all(np.linalg.norm(self.ee_force_bias[arm]) == 0 for arm in self.ee_force_bias):
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        return reward, done, info

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the workpiece.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the workpiece
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.surface)

    def _check_success(self):
        """
        No height-based or other task completion criterion; success is never signaled so policies are not
        terminated early on a lift threshold (episode length is controlled by ``horizon`` / ``ignore_done``).
        """
        return False

    def get_robot_base_pose(self):
        """
        Homogeneous pose of the robot base (root body) in the MuJoCo world frame.

        Returns:
            np.ndarray: 4x4 pose matrix
        """
        root = self.robots[0].robot_model.root_body
        pos = np.array(self.sim.data.get_body_xpos(root))
        rot = np.array(self.sim.data.get_body_xmat(root).reshape(3, 3))
        return make_pose(pos, rot)

    def world_to_robot_base(self, points_world):
        """
        Express world-frame 3D points in the robot base frame (analogous to ``panda_link0`` on hardware).

        Args:
            points_world (np.ndarray): shape (N, 3)

        Returns:
            np.ndarray: shape (N, 3)
        """
        points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
        base_pose = self.get_robot_base_pose()
        base_pos = base_pose[:3, 3]
        base_rot = base_pose[:3, :3]
        return (base_rot.T @ (points_world - base_pos).T).T

    def robot_base_to_world(self, points_base):
        """
        Map robot-base-frame points into the MuJoCo world frame.

        Args:
            points_base (np.ndarray): shape (N, 3)

        Returns:
            np.ndarray: shape (N, 3)
        """
        points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
        base_pose = self.get_robot_base_pose()
        base_pos = base_pose[:3, 3]
        base_rot = base_pose[:3, :3]
        return (base_rot @ points_base.T).T + base_pos

    def get_eef_pose_robot_base(self, arm=None):
        """
        End-effector pose in the robot base frame for policy-transportation logging.

        Args:
            arm (str or None): arm name; defaults to the first arm of robot 0.

        Returns:
            tuple: (position (3,), quaternion (4,) in **w, x, y, z** order)
        """
        arm = arm or self.robots[0].arms[0]
        site_id = self.robots[0].eef_site_id[arm]
        pos_world = np.array(self.sim.data.site_xpos[site_id])
        mat_world = np.array(self.sim.data.site_xmat[site_id].reshape(3, 3))
        pos_base = self.world_to_robot_base(pos_world.reshape(1, 3))[0]
        base_pose = self.get_robot_base_pose()
        rot_base = base_pose[:3, :3].T @ mat_world
        quat_xyzw = mat2quat(rot_base)
        quat_wxyz = convert_quat(quat_xyzw, to="wxyz")
        return pos_base, quat_wxyz

    def _surface_scale_vector(self):
        """Return (sx, sy, sz) applied to the workpiece mesh."""
        if self.surface_scale is None:
            return np.ones(3, dtype=float)
        if isinstance(self.surface_scale, (int, float, np.floating, np.integer)):
            s = float(self.surface_scale)
            return np.array([s, s, s], dtype=float)
        return np.array(self.surface_scale, dtype=float).reshape(3)

    def _surface_local_half_extents(self):
        """Half-extents (x, y, z) of the workpiece in the surface body frame."""
        scale = self._surface_scale_vector()
        half = np.array(self.surface.get_bounding_box_half_size(), dtype=float) * scale
        return half

    def _raycast_surface_points_world(self, points_world_xy, z_offset=0.2):
        """
        Refine XY locations onto the workpiece top using vertical MuJoCo rays.

        Args:
            points_world_xy (np.ndarray): shape (N, 2) world-frame XY samples
            z_offset (float): ray origin height above the surface body origin

        Returns:
            np.ndarray: shape (N, 3) world-frame points on the mesh
        """
        model = self.sim.model._model
        data = self.sim.data._data
        body_z = float(self.sim.data.body_xpos[self.surface_body_id][2])
        geomid = np.array([-1], dtype=np.int32)
        refined = []
        for xy in points_world_xy:
            pnt = np.array([xy[0], xy[1], body_z + z_offset], dtype=np.float64)
            vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
            if dist is None or dist < 0:
                refined.append(np.array([xy[0], xy[1], body_z], dtype=float))
            else:
                refined.append(pnt + dist * vec)
        return np.array(refined, dtype=float)

    def sample_surface_keypoints(
        self,
        nx=20,
        ny=20,
        frame="robot_base",
        contact_offset=0.0,
        use_raycast=True,
    ):
        """
        Build a dense keypoint grid on the workpiece (same 20x20 layout as the real cleaning experiment).

        Points are sampled on the workpiece bounding footprint in the surface body frame, optionally
        projected onto the collision mesh, then expressed in ``robot_base`` or ``world`` coordinates.

        Args:
            nx (int): grid resolution along local X
            ny (int): grid resolution along local Y
            frame (str): ``"robot_base"`` or ``"world"``
            contact_offset (float): offset along surface outward normal (m); positive lifts off the mesh
            use_raycast (bool): if True, refine heights with ``mj_ray`` (recommended for curved workpieces)

        Returns:
            np.ndarray: shape (nx * ny, 3) keypoint coordinates
        """
        assert frame in ("robot_base", "world"), f"frame must be 'robot_base' or 'world', got {frame!r}"

        half = self._surface_local_half_extents()
        scale = self._surface_scale_vector()
        top_local = np.array(self.surface.top_offset, dtype=float) * scale

        xs = np.linspace(-half[0], half[0], nx)
        ys = np.linspace(-half[1], half[1], ny)
        local_xy = np.array([[x, y] for x in xs for y in ys])

        body_pos = np.array(self.sim.data.body_xpos[self.surface_body_id])
        body_mat = np.array(self.sim.data.body_xmat[self.surface_body_id].reshape(3, 3))
        local_pts = np.hstack([local_xy, np.full((local_xy.shape[0], 1), top_local[2])])
        world_pts = (body_mat @ local_pts.T).T + body_pos

        if use_raycast:
            world_pts = self._raycast_surface_points_world(world_pts[:, :2])
            if contact_offset != 0.0:
                world_pts = world_pts + body_mat[:, 2] * contact_offset

        if frame == "robot_base":
            return self.world_to_robot_base(world_pts)
        return world_pts

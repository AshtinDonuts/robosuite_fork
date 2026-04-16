"""
Table workspace (Wipe-style table geometry) with a single pick-place-style object.
"""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BreadObject, CanObject, CerealObject, MilkObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

# Match DEFAULT_WIPE_CONFIG table settings in wipe.py
_DEFAULT_TABLE_FULL_SIZE = (0.5, 0.8, 0.05)
_DEFAULT_TABLE_FRICTION = (0.03, 0.005, 0.0001)
_DEFAULT_TABLE_OFFSET = (0.15, 0.0, 0.9)

_OBJECT_CLASS = {
    "milk": MilkObject,
    "bread": BreadObject,
    "cereal": CerealObject,
    "can": CanObject,
}


class TablePick(ManipulationEnv):
    """
    Single-object pick task on a Wipe-style table (same default table size, friction, and offset as :class:`Wipe`).

    The episode succeeds when the object is lifted above the tabletop by at least ``lift_height_margin``.
    Optional dense rewards follow the same staged structure as :class:`PickPlace` (reach, grasp, lift; hover stage
    is unused).

    Args:
        robots (str or list of str): Single-arm robot specification.

        object_type (str): One of ``"milk"``, ``"bread"``, ``"cereal"``, ``"can"``.

        table_full_size (3-tuple): Table (L, W, H); defaults match ``Wipe`` task config.

        table_friction (3-tuple): MuJoCo friction for the table.

        table_offset (3-tuple): Arena placement offset; ``z`` sets tabletop height (same convention as ``Wipe``).

        z_rotation (float, tuple, or None): Passed to :class:`UniformRandomSampler` for object yaw at reset.

        lift_height_margin (float): Success when object center is at least this far above the tabletop plane.

        reward_scale (None or float): If not None, multiplies the returned reward.

        reward_shaping (bool): If True, add dense staged rewards (see :class:`PickPlace`).

        use_object_obs (bool): Expose object state and gripper-relative observations.

        placement_initializer: Optional custom sampler; default samples uniformly on the table surface. If you pass a
            :class:`SequentialCompositeSampler`, the pick object is attached to the first sub-sampler that is a
            :class:`UniformRandomSampler` with an empty object list (``add_objects()`` is not called on the composite).

        use_table_obstacle (bool): If True, spawn a fixed box obstacle near the table center (see ``obstacle_offset``).

        obstacle_offset (2-tuple): (dx, dy) offset in meters from the table placement anchor (``table_offset`` xy) for
            the obstacle center. The default places it at the middle of the table in the horizontal plane.

        obstacle_half_size (3-tuple): Half-extents (hx, hy, hz) of the obstacle box in meters.

        Other arguments match :class:`ManipulationEnv`.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        object_type="can",
        table_full_size=_DEFAULT_TABLE_FULL_SIZE,
        table_friction=_DEFAULT_TABLE_FRICTION,
        table_offset=_DEFAULT_TABLE_OFFSET,
        z_rotation=None,
        lift_height_margin=0.08,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        use_table_obstacle=True,
        obstacle_offset=(0.0, 0.0),
        obstacle_half_size=(0.04, 0.04, 0.08),
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
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        assert object_type in _OBJECT_CLASS, "object_type must be one of {}".format(list(_OBJECT_CLASS.keys()))

        self.object_type = object_type
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array(table_offset, dtype=float)
        self.z_rotation = z_rotation
        self.lift_height_margin = lift_height_margin

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.use_table_obstacle = use_table_obstacle
        self.obstacle_offset = np.array(obstacle_offset, dtype=float)
        assert self.obstacle_offset.shape == (2,), "obstacle_offset must be a 2-tuple (dx, dy)"
        self.obstacle_half_size = tuple(obstacle_half_size)
        self.table_obstacle = None
        self._builtin_obstacle_composite = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
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

    def reward(self, action=None):
        """
        Sparse reward 1.0 when the object is lifted above the table; optional dense shaping like :class:`PickPlace`.
        """
        reward = float(self._check_success())
        if self.reward_shaping:
            staged = self.staged_rewards()
            reward += max(staged)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def staged_rewards(self):
        """
        Stages: reaching, grasping, lifting (same coefficients as :class:`PickPlace`). Hover is 0 (no goal bin).
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5

        obj = self.object
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=obj.root_body,
            target_type="body",
            return_distance=True,
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=obj.contact_geoms,
                )
            )
            * grasp_mult
        )

        r_lift = 0.0
        if r_grasp > 0.0:
            z_target = float(self.table_offset[2] + 0.25)
            object_z = float(self.sim.data.body_xpos[self.obj_body_id][2])
            z_dist = max(z_target - object_z, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (lift_mult - grasp_mult)

        r_hover = 0.0
        return r_reach, r_grasp, r_lift, r_hover

    def _fixed_table_surface_pose(self, obj, xy_offset):
        """
        World (x, y, z) for @obj center and identity quat (wxyz), matching :class:`UniformRandomSampler` with
        ``on_top=True``, ``z_offset=0.01``, and horizontal offsets ``xy_offset`` from ``self.table_offset``.
        """
        ox, oy = float(xy_offset[0]), float(xy_offset[1])
        base = self.table_offset
        object_x = base[0] + ox
        object_y = base[1] + oy
        z_surface = 0.01
        object_z = z_surface + base[2] - obj.bottom_offset[-1]
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return (object_x, object_y, object_z), quat

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        obj_cls = _OBJECT_CLASS[self.object_type]
        self.object = obj_cls(name=obj_cls.__name__.replace("Object", ""))

        if self.use_table_obstacle:
            self.table_obstacle = BoxObject(
                name="table_obstacle",
                size=list(self.obstacle_half_size),
                rgba=[0.35, 0.35, 0.4, 1.0],
                joints="default",
            )
        else:
            self.table_obstacle = None

        if self.placement_initializer is not None:
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                for name, sampler in self.placement_initializer.samplers.items():
                    if isinstance(sampler, UniformRandomSampler) and len(sampler.mujoco_objects) == 0:
                        self.placement_initializer.add_objects_to_sampler(name, self.object)
                        break
            else:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(self.object)
        else:
            x_half = self.table_full_size[0] / 2 - 0.05
            y_half = self.table_full_size[1] / 2 - 0.05
            # Positive table x is treated as the robot's right when facing the table.
            x_min_right = 0.05
            pick_sampler = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.object,
                x_range=[x_min_right, x_half],
                y_range=[-y_half, y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )
            if self.table_obstacle is not None:
                ox, oy = float(self.obstacle_offset[0]), float(self.obstacle_offset[1])
                obstacle_sampler = UniformRandomSampler(
                    name="TableObstacleSampler",
                    mujoco_objects=self.table_obstacle,
                    x_range=[ox, ox],
                    y_range=[oy, oy],
                    rotation=0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                    rng=self.rng,
                )
                composite = SequentialCompositeSampler(name="TablePickCompositeSampler")
                composite.append_sampler(obstacle_sampler)
                composite.append_sampler(pick_sampler)
                self.placement_initializer = composite
                self._builtin_obstacle_composite = True
            else:
                self.placement_initializer = pick_sampler

        task_objects = [self.object]
        if self.table_obstacle is not None:
            task_objects.insert(0, self.table_obstacle)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=task_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = self.sim.model.body_name2id(self.object.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"
        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        # World-pose sensors are 4x4 matrices: keep them enabled for obs_cache (rel. object sensors) but inactive so they
        # are not concatenated into object-state (same pattern as PickPlace).
        sensors_wp = [
            self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=self.object.name, modality=modality)
        sensors = sensors_wp + obj_sensors
        names = [fn.__name__ for fn in sensors_wp] + obj_sensor_names
        enableds = [True] * len(sensors_wp) + [True] * len(obj_sensors)
        actives = [False] * len(sensors_wp) + [True] * len(obj_sensors)

        for name, s, enabled, active in zip(names, sensors, enableds, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                enabled=enabled,
                active=active,
            )
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, obj_name, f"{obj_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, obj_name, f"{obj_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
        ]
        names = [fn.__name__ for fn in sensors]
        sensors += [obj_pos, obj_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat"]
        return sensors, names

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return
        fixtures = None
        if self.table_obstacle is not None and not self._builtin_obstacle_composite:
            pos, quat = self._fixed_table_surface_pose(self.table_obstacle, self.obstacle_offset)
            fixtures = {self.table_obstacle.name: (pos, quat, self.table_obstacle)}
        object_placements = self.placement_initializer.sample(fixtures=fixtures)
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        cube_height = float(self.sim.data.body_xpos[self.obj_body_id][2])
        table_height = float(self.model.mujoco_arena.table_offset[2])
        return cube_height > table_height + self.lift_height_margin

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            for arm in self.robots[0].arms:
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=self.object.root_body,
                    target_type="body",
                )

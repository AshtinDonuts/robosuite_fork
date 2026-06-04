"""
Table workspace with a single object and a fixed table-relative marker cube.
"""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.table_shelf_pick import (
    _DEFAULT_MARKER_CUBE_KEYPOINT_LOCAL_OFFSETS,
    _DEFAULT_TABLE_FRICTION,
    _DEFAULT_TABLE_FULL_SIZE,
    _DEFAULT_TABLE_OFFSET,
    _KEYPOINT_SITE_SIZE,
    _MARKER_CUBE_KEYPOINT_RGBA,
    _MARKER_CUBE_KEYPOINT_SITE_PREFIX,
    _OBJECT_CLASS,
    _SHELF_MARKER_CUBE_HALF_SIZE,
    _add_keypoint_sites,
    _array_to_mjcf_string,
)
from robosuite.environments.manipulation.table_shelf_pick import TableShelfPick
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class TableCubePick(TableShelfPick):
    """
    Single-object table task with a fixed red marker cube placed relative to the table.

    This is the shelf-less counterpart to :class:`TableShelfPick`. The grasp object (milk by default) keeps the same
    object keypoint behavior, and the red marker cube keeps the marker-cube keypoints / visualization helpers.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        object_type="milk",
        table_full_size=_DEFAULT_TABLE_FULL_SIZE,
        table_friction=_DEFAULT_TABLE_FRICTION,
        table_offset=_DEFAULT_TABLE_OFFSET,
        marker_cube_table_offset=(-0.2, 0.2, 0.05),
        marker_cube_half_size=_SHELF_MARKER_CUBE_HALF_SIZE,
        milk_object_scale=0.8,
        object_x_range=-0.20, # milk obj
        object_y_range=-0.05, # milk obj
        z_rotation=0,
        lift_height_margin=0.08,
        visualize_keypoints=False,
        object_keypoint_local_offsets=None,
        marker_cube_keypoint_local_offsets=None,
        keypoint_site_size=_KEYPOINT_SITE_SIZE,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
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
        camera_names=("agentview", "all-eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        marker_cube_table_offset = np.array(marker_cube_table_offset, dtype=float)
        if marker_cube_table_offset.shape != (3,):
            raise ValueError(
                f"marker_cube_table_offset must be a 3-tuple relative to table_offset, got {marker_cube_table_offset}"
            )

        self.marker_cube_table_offset = marker_cube_table_offset
        self.marker_cube_half_size = float(marker_cube_half_size)
        if self.marker_cube_half_size <= 0.0:
            raise ValueError(f"marker_cube_half_size must be positive, got {self.marker_cube_half_size}")

        if marker_cube_keypoint_local_offsets is None:
            marker_cube_keypoint_local_offsets = (
                np.array(_DEFAULT_MARKER_CUBE_KEYPOINT_LOCAL_OFFSETS, dtype=float)
                * self.marker_cube_half_size
                / _SHELF_MARKER_CUBE_HALF_SIZE
            )

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            object_type=object_type,
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            milk_object_scale=milk_object_scale,
            object_x_range=object_x_range,
            object_y_range=object_y_range,
            z_rotation=z_rotation,
            lift_height_margin=lift_height_margin,
            visualize_keypoints=visualize_keypoints,
            object_keypoint_local_offsets=object_keypoint_local_offsets,
            marker_cube_keypoint_local_offsets=marker_cube_keypoint_local_offsets,
            keypoint_site_size=keypoint_site_size,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
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

    def _load_model(self):
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = list(xpos)
        xpos[2] -= 0.15
        xpos[0] += 0.05
        self.robots[0].robot_model.set_base_xpos(xpos)
        for robot in self.robots:
            self._add_wrist_cameras_to_robot_model(robot.robot_model)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            has_legs=True,
        )
        mujoco_arena.set_origin([0, 0, 0])

        obj_cls = _OBJECT_CLASS[self.object_type]
        obj_name = obj_cls.__name__.replace("Object", "")
        if self.object_type == "milk":
            self.object = obj_cls(name=obj_name, scale=self.milk_object_scale)
        else:
            self.object = obj_cls(name=obj_name)
        self.object_keypoint_site_names = self._add_object_keypoint_sites()

        self.marker_cube = BoxObject(
            name="MarkerCube",
            size=[self.marker_cube_half_size] * 3,
            rgba=[1.0, 0.0, 0.0, 1.0],
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=False,
        )
        self.marker_cube_keypoint_site_names = self._add_marker_cube_keypoint_sites()
        self.marker_cube.get_obj().set(
            "pos",
            _array_to_mjcf_string(self.table_offset + self.marker_cube_table_offset),
        )

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
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.object,
                x_range=self.object_x_range,
                y_range=self.object_y_range,
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.marker_cube, self.object],
        )

    def _setup_references(self):
        ManipulationEnv._setup_references(self)
        self.obj_body_id = self.sim.model.body_name2id(self.object.root_body)
        self.marker_cube_body_id = self.sim.model.body_name2id(self.marker_cube.root_body)
        self.keypoint_site_ids = {
            "object": [self.sim.model.site_name2id(name) for name in self.object_keypoint_site_names],
            "marker_cube": [self.sim.model.site_name2id(name) for name in self.marker_cube_keypoint_site_names],
        }
        self.set_keypoint_visualization(self.visualize_keypoints)

    def _create_obj_sensors(self, obj_name, modality="object"):
        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id], to="xyzw")

        @sensor(modality=modality)
        def marker_cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.marker_cube_body_id])

        @sensor(modality=modality)
        def marker_cube_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.marker_cube_body_id], to="xyzw")

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
        sensors += [obj_pos, obj_quat, marker_cube_pos, marker_cube_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat", "marker_cube_pos", "marker_cube_quat"]
        return sensors, names

    def _add_marker_cube_keypoint_sites(self):
        marker_body = self.marker_cube.get_obj()
        if marker_body is None:
            raise ValueError("Could not find marker cube body for keypoint sites")

        site_names = [
            self.marker_cube.naming_prefix + f"{_MARKER_CUBE_KEYPOINT_SITE_PREFIX}_{i}"
            for i in range(len(self.marker_cube_keypoint_local_offsets))
        ]
        _add_keypoint_sites(
            marker_body,
            self.marker_cube.naming_prefix + _MARKER_CUBE_KEYPOINT_SITE_PREFIX,
            self.marker_cube_keypoint_local_offsets,
            _MARKER_CUBE_KEYPOINT_RGBA,
            self.keypoint_site_size,
            self.visualize_keypoints,
        )
        return site_names

    def get_marker_cube_keypoints(self, local_offsets=None):
        """
        Returns ground-truth keypoints for the fixed table-relative marker cube.
        """
        if local_offsets is None:
            local_offsets = self.marker_cube_keypoint_local_offsets
        return self.get_body_keypoints(self.marker_cube_body_id, local_offsets)

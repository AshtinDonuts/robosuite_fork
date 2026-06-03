"""
Table workspace with a single object and a fixed side shelf.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BreadObject, CanObject, CerealObject, MilkObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import get_elements, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

_DEFAULT_TABLE_FULL_SIZE = (0.9, 1.5, 0.05)
# _DEFAULT_TABLE_FULL_SIZE = (0.9, 0.8, 0.05) # Default
_DEFAULT_TABLE_FRICTION = (1.0, 0.005, 0.0001)
_DEFAULT_TABLE_OFFSET = (0.0, 0.0, 0.8)

_OBJECT_CLASS = {
    "milk": MilkObject,
    "bread": BreadObject,
    "cereal": CerealObject,
    "can": CanObject,
}

_SHELF_XML = {
    "3level": "objects/shelf_3level.xml",
    "4level": "objects/shelf_4level.xml",
}

# Default marker cube z (shelf-body frame) between the 2nd and 3rd shelf levels.
_DEFAULT_MARKER_CUBE_SHELF_Z = {
    "3level": 0.75,   # level 2 top (0.605) and level 3 bottom (0.995)
    "4level": 0.55,   # level 2 top (0.472) and level 3 bottom (0.728)
}
_SHELF_MARKER_CUBE_HALF_SIZE = 0.04
_DEFAULT_OBJECT_KEYPOINT_LOCAL_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.0, 0.04, 0.0],
        [0.0, 0.0, 0.04],
    ],
    dtype=float,
)
_DEFAULT_MARKER_CUBE_KEYPOINT_LOCAL_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [_SHELF_MARKER_CUBE_HALF_SIZE, 0.0, 0.0],
        [0.0, _SHELF_MARKER_CUBE_HALF_SIZE, 0.0],
        [0.0, 0.0, _SHELF_MARKER_CUBE_HALF_SIZE],
    ],
    dtype=float,
)
_OBJECT_KEYPOINT_SITE_PREFIX = "object_keypoint"
_MARKER_CUBE_KEYPOINT_SITE_PREFIX = "marker_cube_keypoint"
_OBJECT_KEYPOINT_RGBA = (0.0, 0.35, 1.0, 1.0)
_MARKER_CUBE_KEYPOINT_RGBA = (0.0, 1.0, 0.15, 1.0)
_KEYPOINT_SITE_SIZE = 0.01


def _array_to_mjcf_string(array):
    return " ".join(str(float(x)) for x in array)


def _add_keypoint_sites(body, name_prefix, local_offsets, rgba, size, visible):
    alpha = float(rgba[3]) if visible else 0.0
    site_names = []
    for i, offset in enumerate(local_offsets):
        name = f"{name_prefix}_{i}"
        site_names.append(name)
        body.append(
            ET.Element(
                "site",
                attrib={
                    "name": name,
                    "type": "sphere",
                    "pos": _array_to_mjcf_string(offset),
                    "size": str(float(size)),
                    "rgba": _array_to_mjcf_string((*rgba[:3], alpha)),
                    "group": "1",
                },
            )
        )
    return site_names


class _ShelfObject(MujocoXMLObject):
    """
    Fixed shelf loaded from ``models/assets/objects/shelf_3level.xml`` or ``shelf_4level.xml``.
    """

    def __init__(
        self,
        name="shelf",
        shelf_type="3level",
        marker_cube_shelf_x=0.0,
        marker_cube_shelf_y=0.0,
        marker_cube_shelf_z=None,
        marker_cube_keypoint_local_offsets=None,
        visualize_keypoints=False,
        keypoint_site_size=_KEYPOINT_SITE_SIZE,
    ):
        if shelf_type not in _SHELF_XML:
            raise ValueError(f"shelf_type must be one of {list(_SHELF_XML.keys())}, got {shelf_type!r}")
        self.shelf_type = shelf_type
        self.marker_cube_shelf_x = float(marker_cube_shelf_x)
        self.marker_cube_shelf_y = float(marker_cube_shelf_y)
        if marker_cube_shelf_z is None:
            marker_cube_shelf_z = _DEFAULT_MARKER_CUBE_SHELF_Z[shelf_type]
        self.marker_cube_shelf_z = float(marker_cube_shelf_z)
        if marker_cube_keypoint_local_offsets is None:
            marker_cube_keypoint_local_offsets = _DEFAULT_MARKER_CUBE_KEYPOINT_LOCAL_OFFSETS
        self.marker_cube_keypoint_local_offsets = np.array(marker_cube_keypoint_local_offsets, dtype=float)
        self.visualize_keypoints = bool(visualize_keypoints)
        self.keypoint_site_size = float(keypoint_site_size)
        super().__init__(
            xml_path_completion(_SHELF_XML[shelf_type]),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    def _get_object_subtree(self):
        obj = self.worldbody.find("./body[@name='shelf']")
        if obj is None:
            raise ValueError(f"{_SHELF_XML[self.shelf_type]} must contain a top-level body named 'shelf'")

        obj = ET.fromstring(ET.tostring(obj))
        obj.attrib["name"] = "main"
        obj.attrib.pop("pos", None)

        # This scene uses the shelf as a fixed fixture beside the table.
        for joint in list(obj.findall("./joint")) + list(obj.findall("./freejoint")):
            obj.remove(joint)

        for i, (parent, geom) in enumerate(get_elements(obj, "geom")):
            if geom.get("name") is None:
                geom.set("name", f"shelf_geom_{i}")
            geom.set("group", "0")

            visual_geom = deepcopy(geom)
            visual_geom.set("name", geom.get("name") + "_visual")
            visual_geom.set("group", "1")
            visual_geom.set("contype", "0")
            visual_geom.set("conaffinity", "0")
            visual_geom.set("mass", "1e-8")
            parent.append(visual_geom)

        obj.append(
            ET.Element(
                "site",
                attrib={
                    "name": "default_site",
                    "pos": "0 0 0.5",
                    "size": "0.002 0.002 0.002",
                    "rgba": "1 0 0 0",
                    "type": "sphere",
                    "group": "0",
                },
            )
        )
        obj.append(
            ET.Element(
                "site",
                attrib={"name": "bottom_site", "pos": "0 0 0", "size": "0.002", "rgba": "1 0 0 0"},
            )
        )
        obj.append(
            ET.Element(
                "site",
                attrib={"name": "top_site", "pos": "0 0 1.005", "size": "0.002", "rgba": "1 0 0 0"},
            )
        )
        obj.append(
            ET.Element(
                "site",
                attrib={
                    "name": "horizontal_radius_site",
                    "pos": "0.34 0 0",
                    "size": "0.002",
                    "rgba": "1 0 0 0",
                },
            )
        )

        half = _SHELF_MARKER_CUBE_HALF_SIZE
        marker_cube = ET.Element(
            "body",
            attrib={
                "name": "level_marker_cube",
                "pos": (
                    f"{self.marker_cube_shelf_x} "
                    f"{self.marker_cube_shelf_y} "
                    f"{self.marker_cube_shelf_z}"
                ),
            },
        )
        marker_cube.append(
            ET.Element(
                "geom",
                attrib={
                    "name": "level_marker_cube_geom",
                    "type": "box",
                    "pos": "0 0 0",
                    "size": f"{half} {half} {half}",
                    "rgba": "1 0 0 1",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                    "mass": "1e-8",
                },
            )
        )
        _add_keypoint_sites(
            marker_cube,
            _MARKER_CUBE_KEYPOINT_SITE_PREFIX,
            self.marker_cube_keypoint_local_offsets,
            _MARKER_CUBE_KEYPOINT_RGBA,
            self.keypoint_site_size,
            self.visualize_keypoints,
        )
        obj.append(marker_cube)
        return obj

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, 0.0])

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, 1.005])

    @property
    def horizontal_radius(self):
        return 0.34

    def get_bounding_box_half_size(self):
        return np.array([0.34, 0.17, 0.5025])


class TableShelfPick(ManipulationEnv):
    """
    Single-object table task with a shelf fixture to the side of the workspace.

    The layout is intended to resemble a robot reaching over a table with the shelf on the robot's left side, as in
    the reference screenshot. The shelf is loaded from ``objects/shelf_3level.xml`` or ``objects/shelf_4level.xml``
    and fixed in place.
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
        shelf_type="4level",  # {3level, 4level}
        shelf_pos=(-0.10, 0.50, 0.8), #@user
        shelf_rotation=0,
        marker_cube_shelf_x=0.0,
        marker_cube_shelf_y=0.0,
        marker_cube_shelf_z=None,
        milk_object_scale=1.5,
        object_x_range=(-0.18, 0.18), # Default: (-0.28, -0.08)
        object_y_range=(-0.18, 0.18), # Default: (-0.30, -0.18)
        z_rotation=None,
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
        assert shelf_type in _SHELF_XML, "shelf_type must be one of {}".format(list(_SHELF_XML.keys()))

        self.object_type = object_type
        self.shelf_type = shelf_type
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array(table_offset, dtype=float)
        self.shelf_pos = np.array(shelf_pos, dtype=float)
        self.shelf_rotation = shelf_rotation
        self.marker_cube_shelf_x = float(marker_cube_shelf_x)
        self.marker_cube_shelf_y = float(marker_cube_shelf_y)
        if marker_cube_shelf_z is None:
            marker_cube_shelf_z = _DEFAULT_MARKER_CUBE_SHELF_Z[shelf_type]
        self.marker_cube_shelf_z = float(marker_cube_shelf_z)
        self.milk_object_scale = float(milk_object_scale)
        if self.milk_object_scale <= 0.0:
            raise ValueError(f"milk_object_scale must be positive, got {self.milk_object_scale}")
        self.object_x_range = tuple(object_x_range)
        self.object_y_range = tuple(object_y_range)
        self.z_rotation = z_rotation
        self.lift_height_margin = lift_height_margin
        using_default_object_keypoint_offsets = object_keypoint_local_offsets is None
        if object_keypoint_local_offsets is None:
            object_keypoint_local_offsets = _DEFAULT_OBJECT_KEYPOINT_LOCAL_OFFSETS
        if marker_cube_keypoint_local_offsets is None:
            marker_cube_keypoint_local_offsets = _DEFAULT_MARKER_CUBE_KEYPOINT_LOCAL_OFFSETS
        if using_default_object_keypoint_offsets and self.object_type == "milk" and self.milk_object_scale != 1.0:
            object_keypoint_local_offsets = np.array(object_keypoint_local_offsets, dtype=float) * self.milk_object_scale
        self.object_keypoint_local_offsets = self._validate_keypoint_local_offsets(object_keypoint_local_offsets)
        self.marker_cube_keypoint_local_offsets = self._validate_keypoint_local_offsets(
            marker_cube_keypoint_local_offsets
        )
        self.visualize_keypoints = bool(visualize_keypoints)
        self.keypoint_site_size = float(keypoint_site_size)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self._builtin_placement_initializer = placement_initializer is None

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
        reward = float(self._check_success())
        if self.reward_shaping:
            reward += max(self.staged_rewards())
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def staged_rewards(self):
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5

        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.object.root_body,
            target_type="body",
            return_distance=True,
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult
        r_grasp = int(self._check_grasp(self.robots[0].gripper, self.object.contact_geoms)) * grasp_mult

        r_lift = 0.0
        if r_grasp > 0.0:
            z_target = float(self.table_offset[2] + 0.25)
            object_z = float(self.sim.data.body_xpos[self.obj_body_id][2])
            z_dist = max(z_target - object_z, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (lift_mult - grasp_mult)

        return r_reach, r_grasp, r_lift, 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

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
        self.shelf = _ShelfObject(
            name="Shelf",
            shelf_type=self.shelf_type,
            marker_cube_shelf_x=self.marker_cube_shelf_x,
            marker_cube_shelf_y=self.marker_cube_shelf_y,
            marker_cube_shelf_z=self.marker_cube_shelf_z,
            marker_cube_keypoint_local_offsets=self.marker_cube_keypoint_local_offsets,
            visualize_keypoints=self.visualize_keypoints,
            keypoint_site_size=self.keypoint_site_size,
        )
        self.marker_cube_keypoint_site_names = [
            self.shelf.naming_prefix + f"{_MARKER_CUBE_KEYPOINT_SITE_PREFIX}_{i}"
            for i in range(len(self.marker_cube_keypoint_local_offsets))
        ]
        self.shelf.set_pos(self.shelf_pos)
        self.shelf.set_euler([0.0, 0.0, self.shelf_rotation])

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
            mujoco_objects=[self.shelf, self.object],
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = self.sim.model.body_name2id(self.object.root_body)
        self.shelf_body_id = self.sim.model.body_name2id(self.shelf.root_body)
        self.marker_cube_body_id = self.sim.model.body_name2id(self.shelf.naming_prefix + "level_marker_cube")
        self.keypoint_site_ids = {
            "object": [self.sim.model.site_name2id(name) for name in self.object_keypoint_site_names],
            "marker_cube": [self.sim.model.site_name2id(name) for name in self.marker_cube_keypoint_site_names],
        }
        self.set_keypoint_visualization(self.visualize_keypoints)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"
        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

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

        @sensor(modality=modality)
        def shelf_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.shelf_body_id])

        @sensor(modality=modality)
        def shelf_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.shelf_body_id], to="xyzw")

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
        sensors += [obj_pos, obj_quat, shelf_pos, shelf_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat", "shelf_pos", "shelf_quat"]
        return sensors, names

    def _add_object_keypoint_sites(self):
        target_body_names = {self.object.root_body, getattr(self.object, "_root_body", None)}
        object_body = None
        body_names = []
        for body in self.object.worldbody.findall(".//body"):
            body_name = body.get("name")
            body_names.append(body_name)
            if body_name in target_body_names:
                object_body = body
                break

        if object_body is None:
            raise ValueError(
                f"Could not find object root body {self.object.root_body!r} for keypoint sites. "
                f"Available object bodies: {body_names}"
            )

        site_names = [
            self.object.naming_prefix + f"{_OBJECT_KEYPOINT_SITE_PREFIX}_{i}"
            for i in range(len(self.object_keypoint_local_offsets))
        ]
        _add_keypoint_sites(
            object_body,
            self.object.naming_prefix + _OBJECT_KEYPOINT_SITE_PREFIX,
            self.object_keypoint_local_offsets,
            _OBJECT_KEYPOINT_RGBA,
            self.keypoint_site_size,
            self.visualize_keypoints,
        )
        return site_names

    @staticmethod
    def _validate_keypoint_local_offsets(local_offsets):
        local_offsets = np.array(local_offsets, dtype=float)
        if local_offsets.ndim != 2 or local_offsets.shape[1] != 3:
            raise ValueError(f"local_offsets must have shape (N, 3), got {local_offsets.shape}")
        if local_offsets.shape[0] < 3:
            raise ValueError(f"local_offsets must include at least 3 points, got {local_offsets.shape[0]}")
        if np.linalg.matrix_rank(local_offsets[1:] - local_offsets[0]) < 2:
            raise ValueError("local_offsets must include at least 3 non-collinear points")
        return local_offsets

    def get_body_keypoints(self, body_id, local_offsets):
        """
        Projects local keypoint offsets into the world frame from a MuJoCo body pose.

        Args:
            body_id (int): MuJoCo body id whose ``body_xpos`` and ``body_xmat`` define the object frame.
            local_offsets (array-like): shape (N, 3), with N >= 3 non-collinear local keypoints.

        Returns:
            np.ndarray: shape (N, 3) world-frame keypoint coordinates.
        """
        local_offsets = self._validate_keypoint_local_offsets(local_offsets)
        body_pos = np.array(self.sim.data.body_xpos[body_id], dtype=float)
        body_mat = np.array(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        return (body_mat @ local_offsets.T).T + body_pos

    def get_object_keypoints(self, local_offsets=None):
        """
        Returns ground-truth keypoints for the active grasp object, e.g. milk by default.
        """
        if local_offsets is None:
            local_offsets = self.object_keypoint_local_offsets
        return self.get_body_keypoints(self.obj_body_id, local_offsets)

    def get_marker_cube_keypoints(self, local_offsets=None):
        """
        Returns ground-truth keypoints for the fixed shelf marker cube.
        """
        if local_offsets is None:
            local_offsets = self.marker_cube_keypoint_local_offsets
        return self.get_body_keypoints(self.marker_cube_body_id, local_offsets)

    def set_keypoint_visualization(self, enabled):
        """
        Shows or hides the object and marker-cube keypoint sites.
        """
        self.visualize_keypoints = bool(enabled)
        self._set_keypoint_site_alpha(self.visualize_keypoints)

    def _set_keypoint_site_alpha(self, visible):
        if not hasattr(self, "keypoint_site_ids"):
            return

        alpha = 1.0 if visible else 0.0
        for site_ids in self.keypoint_site_ids.values():
            for site_id in site_ids:
                self.sim.model.site_rgba[site_id][3] = alpha

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return

        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """No automated success; demos end on manual reset (e.g. ``q`` in teleop)."""
        return False

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        self._set_keypoint_site_alpha(self.visualize_keypoints and vis_settings.get("env", True))
        if vis_settings["grippers"]:
            for arm in self.robots[0].arms:
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=self.object.root_body,
                    target_type="body",
                )

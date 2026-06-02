import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


class Empty(ManipulationEnv):
    """
    Minimal manipulation environment with an empty arena (no table, no task objects).

    By default the robot has no pedestal (``NullMount``) and is placed at the arena origin.
    Pass ``use_table_mount=True`` to use the robot's default pedestal mount (e.g.
    ``RethinkMount`` for IIWA/Panda) with the table-mount XY offset (same convention as
    Wipe / Lift). No workspace table or dirt is added.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise="default",
        use_table_mount=False,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        use_camera_obs=True,
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
        **kwargs,
    ):
        self.use_table_mount = use_table_mount
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        if not use_table_mount and base_types == "default":
            base_types = "NullMount"

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
        return 0.0

    def _load_model(self):
        super()._load_model()

        if self.use_table_mount:
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](0.5)
            self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = EmptyArena()
        mujoco_arena.set_origin([0, 0, 0])

        ET.SubElement(
            mujoco_arena.worldbody,
            "site",
            name="green_sphere_marker",
            type="sphere",
            size="0.05",
            pos="1 0 0",
            rgba="0 1 0 1",
        )
        # Cylinder markers for start (blue) and goal (red): oriented along EE Z-axis at runtime.
        ET.SubElement(
            mujoco_arena.worldbody,
            "site",
            name="start_position_marker",
            type="cylinder",
            size="0.02 0.05",
            pos="0 0 -100",
            rgba="0 0 1 0.9",
        )
        ET.SubElement(
            mujoco_arena.worldbody,
            "site",
            name="goal_position_marker",
            type="cylinder",
            size="0.02 0.05",
            pos="0 0 -100",
            rgba="1 0 0 0.9",
        )
        # Pre-allocated pool of subsampled trajectory waypoint markers (green cylinders).
        # demo_playback_puma_dataset.py uses these at runtime; unused slots stay transparent.
        # N_WAYPOINT_MARKER_SITES = 60
        for _i in range(60):
            ET.SubElement(
                mujoco_arena.worldbody,
                "site",
                name=f"waypoint_marker_{_i}",
                type="cylinder",
                size="0.008 0.03",
                pos="0 0 -100",
                rgba="0 0.75 0 0",
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _check_success(self):
        return False

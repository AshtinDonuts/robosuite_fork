import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


class Empty(ManipulationEnv):
    """
    Minimal manipulation environment with an empty arena (no table, no task objects).

    By default the robot uses the empty-arena base offset. Pass ``use_table_mount=True``
    to apply the table-mount offset (same convention as Wipe / Lift) when the robot
    should sit on a pedestal as if in front of a table. No workspace table or dirt is added.
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
        else:
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
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

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _check_success(self):
        return False

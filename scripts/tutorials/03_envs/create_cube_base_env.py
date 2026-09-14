# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a simple environment with a floating cube. The cube is controlled by a PD
controller to track an arbitrary target position.

While going through this tutorial, we recommend you to pay attention to how a custom action term
is defined. The action term is responsible for processing the raw actions and applying them to the
scene entities.

We also define an event term called 'randomize_scale' that randomizes the scale of
the cube. This event term has the mode 'prestartup', which means that it is applied on the USD stage
before the simulation starts. Additionally, the flag 'replicate_physics' is set to False,
which means that the cube is not replicated across multiple environments but rather each
environment gets its own cube instance.

The rest of the environment is similar to the previous tutorials.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cube_base_env.py --num_envs 32

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Tutorial on creating a floating cube environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# 自定义动作项
##


class CubeActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: ManagerBasedEnv):
        # 调用父类构造函数
        super().__init__(cfg, env)
        # 创建缓冲区
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        # 控制器增益
        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # 存储原始动作
        self._raw_actions[:] = actions
        # 无需处理
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        # 实现 PD 控制器以跟踪目标位置
        pos_error = self._processed_actions - (self._asset.data.root_pos_w - self._env.scene.env_origins)
        vel_error = -self._asset.data.root_lin_vel_w
        # 设置速度目标
        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        self._asset.write_root_velocity_to_sim(self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""

    p_gain: float = 5.0
    """Proportional gain of the PD controller."""
    d_gain: float = 0.5
    """Derivative gain of the PD controller."""


##
# 自定义观测项
##


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # 提取使用的实体（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


##
# 场景定义
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a ground plane, light source and floating cubes (gravity disabled).
    """

    # 添加地形
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # 添加立方体
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# 环境设置
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = CubeActionTermCfg(asset_name="cube")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 立方体位置
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 此事件项重置立方体的基础位置。
    # 模式设置为 'reset'，表示每当环境实例被重置时
    # （由于 'TerminationCfg' 中定义的终止条件），基础位置都会被重置。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # 此事件项随机化立方体的缩放比例。
    # 模式设置为 'prestartup'，表示在仿真开始前在 USD 阶段随机化缩放比例。
    # 注意：USD 级别的随机化需要将 'replicate_physics' 标志设置为 False。
    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (0.5, 1.5), "y": (0.5, 1.5), "z": (0.5, 1.5)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # 此事件项随机化立方体的视觉颜色。
    # 与缩放随机化类似，这也是 USD 级别的随机化，需要将 'replicate_physics' 标志设置为 False。
    randomize_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("cube"),
            "mesh_name": "geometry/mesh",
            "event_name": "rep_cube_randomize_color",
        },
    )


##
# 环境配置
##


@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # 场景设置
    # 'replicate_physics' 标志设置为 False，表示立方体不会在多个环境间复制，
    # 而是每个环境拥有自己独立的立方体实例。
    # 这允许独立修改每个环境中立方体的属性。
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)

    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # 常规设置
        self.decimation = 2
        # 仿真设置
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = 2  # render interval should be a multiple of decimation
        self.sim.device = args_cli.device
        # 查看器设置
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)


def main():
    """Main function."""

    # 搭建基础环境
    env_cfg = CubeEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # 设置目标位置指令
    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    target_position[:, 2] += 2.0
    # 偏移所有目标使其相对于世界原点
    target_position -= env.scene.env_origins

    # 仿真物理
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # 环境步进
            obs, _ = env.step(target_position)
            # 打印目标与当前位置的均方误差
            error = torch.norm(obs["policy"] - target_position).mean().item()
            print(f"[Step: {count:04d}]: Mean position error: {error:.4f}")
            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

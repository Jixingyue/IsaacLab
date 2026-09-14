# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 光照
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # 创建独立分组
    # 每个分组中都会放置一个机器人
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # 刚体物体
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    # 返回场景信息
    scene_entities = {"cone": cone_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # 提取场景实体
    # 注意：这里只是为了提高可读性。通常更推荐直接从字典中访问实体
    #   在下一个教程中，这个字典会被 InteractiveScene 类替代
    cone_object = entities["cone"]
    # 定义仿真步进参数
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # 运行物理仿真
    while simulation_app.is_running():
        # 重置
        if count % 250 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0
            # 重置根状态
            root_state = cone_object.data.default_root_state.clone()
            # 在原点周围的圆柱区域内随机采样位置
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # 将根状态写入仿真
            cone_object.write_root_pose_to_sim(root_state[:, :7])
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # 重置缓冲区
            cone_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # 写入仿真数据
        cone_object.write_data_to_sim()
        # 执行一步仿真
        sim.step()
        # 更新时间
        sim_time += sim_dt
        count += 1
        # 更新缓冲区
        cone_object.update(sim_dt)
        # 打印根位置
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_pos_w}")


def main():
    """Main function."""
    # 加载 Kit 辅助组件
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # 构建场景
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # 启动模拟器
    sim.reset()
    # 现在已准备就绪
    print("[INFO]: Setup complete...")
    # 运行模拟器
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

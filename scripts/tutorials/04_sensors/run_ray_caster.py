# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the ray-caster sensor.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
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
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def define_sensor() -> RayCaster:
    """Defines the ray-caster sensor to add to the scene."""
    # 创建射线投射传感器
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        ray_alignment="yaw",
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def design_scene() -> dict:
    """Design the scene."""
    # 构建场景
    # -- 粗糙地形
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- 光照
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)

    # 创建名为 "Origin0"、"Origin1"、"Origin2"、"Origin3" 的独立分组
    # 每个分组中都会放置一个物体
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    # -- 球体
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    balls = RigidObject(cfg)
    # -- 传感器
    ray_caster = define_sensor()

    # 返回场景信息
    scene_entities = {"balls": balls, "ray_caster": ray_caster}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # 提取实体以简化表示
    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]

    # 定义传感器的初始位置
    ball_default_state = balls.data.default_root_state.clone()
    ball_default_state[:, :3] = torch.rand_like(ball_default_state[:, :3]) * 10

    # 创建用于重置场景的计数器
    step_count = 0
    # 仿真物理
    while simulation_app.is_running():
        # 重置场景
        if step_count % 250 == 0:
            # 重置球体
            balls.write_root_pose_to_sim(ball_default_state[:, :7])
            balls.write_root_velocity_to_sim(ball_default_state[:, 7:])
            # 重置传感器
            ray_caster.reset()
            # 重置计数器
            step_count = 0
        # 执行一步仿真
        sim.step()
        # 更新射线投射器
        with Timer(
            f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of"
            f" {torch.max(ray_caster.data.pos_w).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # 更新计数器
        step_count += 1


def main():
    """Main function."""
    # 加载仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # 构建场景
    scene_entities = design_scene()
    # 启动模拟器
    sim.reset()
    # 现在已准备就绪
    print("[INFO]: Setup complete...")
    # 运行模拟器
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

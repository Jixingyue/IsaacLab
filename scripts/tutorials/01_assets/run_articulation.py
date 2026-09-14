# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
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
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 光照
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建名为 "Origin1"、"Origin2" 的独立分组
    # 每个分组中都会放置一个机器人
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # 原点 1
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # 原点 2
    sim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # 关节系统
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # 返回场景信息
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # 提取场景实体
    # 注意：这里只是为了提高可读性。通常更推荐直接从字典中访问实体
    #   在下一个教程中，这个字典会被 InteractiveScene 类替代
    robot = entities["cartpole"]
    # 定义仿真步进参数
    sim_dt = sim.get_physics_dt()
    count = 0
    # 仿真循环
    while simulation_app.is_running():
        # 重置
        if count % 500 == 0:
            # 重置计数器
            count = 0
            # 重置场景实体
            # 根状态
            # 由于状态是以仿真世界坐标系写入的，因此我们要按原点对根状态进行偏移
            # 否则机器人会生成在仿真世界的 (0, 0, 0) 位置
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 为关节位置添加一些噪声
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 清空内部缓冲区
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # 应用随机动作
        # -- 生成随机关节力矩
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- 将动作施加到机器人
        robot.set_joint_effort_target(efforts)
        # -- 将数据写入仿真
        robot.write_data_to_sim()
        # 执行一步仿真
        sim.step()
        # 递增计数器
        count += 1
        # 更新缓冲区
        robot.update(sim_dt)


def main():
    """Main function."""
    # 加载 Kit 辅助组件
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
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

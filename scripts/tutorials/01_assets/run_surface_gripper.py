# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a pick-and-place robot equipped with a surface gripper and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu

When running this script make sure the --device flag is set to cpu. This is because the surface gripper is
currently only supported on the CPU.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")
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
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets import PICK_AND_PLACE_CFG  # isort:skip


def design_scene():
    """Designs the scene."""
    # 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 光照
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建名为 "Origin1"、"Origin2" 的独立分组
    # 每个分组中都会放置一个机器人
    origins = [[2.75, 0.0, 0.0], [-2.75, 0.0, 0.0]]
    # 原点 1
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # 原点 2
    sim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # 关节系统：先定义机器人配置
    pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
    pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
    pick_and_place_robot = Articulation(cfg=pick_and_place_robot_cfg)

    # 表面夹爪：接着定义表面夹爪配置
    surface_gripper_cfg = SurfaceGripperCfg()
    # 需要告诉视图表面夹爪使用哪个 prim
    surface_gripper_cfg.prim_path = "/World/Origin.*/Robot/picker_head/SurfaceGripper"
    # 接着可以为表面夹爪设置不同参数，注意如果这些参数未设置
    # 视图会尝试从该 prim 中读取它们
    surface_gripper_cfg.max_grip_distance = 0.1  # [m] 夹爪能够抓取物体的最大距离
    surface_gripper_cfg.shear_force_limit = 500.0  # [N] 垂直方向上的力限制
    surface_gripper_cfg.coaxial_force_limit = 500.0  # [N] 沿夹爪轴线方向的力限制
    surface_gripper_cfg.retry_interval = 0.1  # [s] 夹爪保持抓取状态的时间
    # 现在可以生成表面夹爪
    surface_gripper = SurfaceGripper(cfg=surface_gripper_cfg)

    # 返回场景信息
    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    """Runs the simulation loop."""
    # 提取场景实体
    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]

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
            # 打开夹爪并确保夹爪处于张开状态
            surface_gripper.reset()
            print("[INFO]: Resetting gripper state...")

        # 在 -1 到 1 之间采样随机指令
        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
        # 夹爪行为如下：
        # -1 < command < -0.3 --> 夹爪正在张开
        # -0.3 < command < 0.3 --> 夹爪空闲
        # 0.3 < command < 1 --> 夹爪正在闭合
        print(f"[INFO]: Gripper commands: {gripper_commands}")
        mapped_commands = [
            "Opening" if command < -0.3 else "Closing" if command > 0.3 else "Idle" for command in gripper_commands
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")
        # 设置夹爪指令
        surface_gripper.set_grippers_command(gripper_commands)
        # 将数据写入仿真
        surface_gripper.write_data_to_sim()
        # 执行一步仿真
        sim.step()
        # 递增计数器
        count += 1
        # 从仿真中读取夹爪状态
        surface_gripper.update(sim_dt)
        # 从缓冲区中读取夹爪状态
        surface_gripper_state = surface_gripper.state
        # 夹爪状态是整数列表，可映射为以下含义：
        # -1 --> 张开
        # 0 --> 闭合中
        # 1 --> 已闭合
        # 打印夹爪状态
        print(f"[INFO]: Gripper state: {surface_gripper_state}")
        mapped_commands = [
            "Open" if state == -1 else "Closing" if state == 0 else "Closed" for state in surface_gripper_state.tolist()
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")


def main():
    """Main function."""
    # 加载 Kit 辅助组件
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
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

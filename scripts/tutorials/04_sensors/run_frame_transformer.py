# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the FrameTransformer sensor by visualizing the frames that it creates.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(
    description="This script checks the FrameTransformer sensor by visualizing the frames that it creates."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def define_sensor() -> FrameTransformer:
    """Defines the FrameTransformer sensor to add to the scene."""
    # 定义偏移
    rot_offset = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
    pos_offset = math_utils.quat_apply(rot_offset, torch.tensor([0.08795, 0.01305, -0.33797]))

    # 使用 .* 获取全身 + LF_FOOT 的示例
    frame_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/Robot/.*"),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/Robot/LF_SHANK",
                name="LF_FOOT_USER",
                offset=OffsetCfg(pos=tuple(pos_offset.tolist()), rot=tuple(rot_offset[0].tolist())),
            ),
        ],
        debug_vis=False,
    )
    frame_transformer = FrameTransformer(frame_transformer_cfg)

    return frame_transformer


def design_scene() -> dict:
    """Design the scene."""
    # 构建场景
    # -- 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- 光照
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- 机器人
    robot = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Robot"))
    # -- 传感器
    frame_transformer = define_sensor()

    # 返回场景信息
    scene_entities = {"robot": robot, "frame_transformer": frame_transformer}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # 定义仿真步进参数
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 提取实体以简化表示
    robot: Articulation = scene_entities["robot"]
    frame_transformer: FrameTransformer = scene_entities["frame_transformer"]

    # 我们一次只需要一个可视化。此可视化器将逐步遍历每个坐标系，
    # 以便用户可以验证在控制台打印坐标系名称时是否正确可视化了相应坐标系
    if not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        transform_visualizer = VisualizationMarkers(cfg)
        # 用于绘制连接坐标系连线的调试绘制接口
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    else:
        transform_visualizer = None
        draw_interface = None

    frame_index = 0
    # 仿真物理
    while simulation_app.is_running():
        # 按策略控制频率（50 Hz）执行此循环
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        # 执行一步仿真
        sim.step()
        # 更新仿真时间
        sim_time += sim_dt
        count += 1
        # 从仿真中读取数据
        robot.update(sim_dt)
        frame_transformer.update(dt=sim_dt)

        # 更改当前可视化的坐标系，以确保坐标系名称
        # 与坐标系正确关联
        if not args_cli.headless:
            if count % 50 == 0:
                # 获取坐标系名称
                frame_names = frame_transformer.data.target_frame_names
                # 递增坐标系索引
                frame_index += 1
                frame_index = frame_index % len(frame_names)
                print(f"Displaying Frame ID {frame_index}: {frame_names[frame_index]}")

            # 可视化坐标系
            source_pos = frame_transformer.data.source_pos_w
            source_quat = frame_transformer.data.source_quat_w
            target_pos = frame_transformer.data.target_pos_w[:, frame_index]
            target_quat = frame_transformer.data.target_quat_w[:, frame_index]
            # 绘制坐标系
            transform_visualizer.visualize(
                torch.cat([source_pos, target_pos], dim=0), torch.cat([source_quat, target_quat], dim=0)
            )
            # 绘制连接坐标系的连线
            draw_interface.clear_lines()
            # 连线的统一颜色
            lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
            line_thicknesses = [5.0] * source_pos.shape[0]
            draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)


def main():
    """Main function."""
    # 加载 Kit 辅助组件
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # 构建场景
    scene_entities = design_scene()
    # 启动模拟器
    sim.reset()
    # 现在已准备就绪
    print("[INFO]: Setup complete...")
    # 运行模拟器
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

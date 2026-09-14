# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray-cast camera sensor from the Isaac Lab framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster_camera.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth


def define_sensor() -> RayCasterCamera:
    """Defines the ray-cast camera sensor to add to the scene."""
    # 相机基础坐标系
    # 与 USD 相机不同，我们将传感器关联到这些位置的 prim。
    # 这意味着传感器的父级 prim 就是此位置处的 prim。
    sim_utils.create_prim("/World/Origin_00/CameraSensor", "Xform")
    sim_utils.create_prim("/World/Origin_01/CameraSensor", "Xform")

    # 设置相机传感器
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=True,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
    )
    # 创建相机
    camera = RayCasterCamera(cfg=camera_cfg)

    return camera


def design_scene():
    # 构建场景
    # -- 粗糙地形
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- 光照
    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- 传感器
    camera = define_sensor()

    # 返回场景信息
    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # 提取实体以简化表示
    camera: RayCasterCamera = scene_entities["camera"]

    # 创建 replicator 写入器
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # 设置位姿：有两种方式设置相机的位姿。
    # -- 方式一：使用 view 设置位姿
    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)
    # -- 方式二：使用 ROS 设置位姿
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_poses(position, orientation, indices=[0], convention="ros")

    # 仿真物理
    while simulation_app.is_running():
        # 执行一步仿真
        sim.step()
        # 更新相机数据
        camera.update(dt=sim.get_physics_dt())

        # 打印相机信息
        print(camera)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # 提取相机数据
        if args_cli.save:
            # 提取相机数据
            camera_index = 0
            # 注意：BasicWriter 仅支持以 numpy 格式保存数据，因此我们需要将数据转换为 numpy。
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # 提取其他信息
            single_cam_info = camera.data.info[camera_index]

            # 将数据重新打包为 replicator 格式以使用其写入器保存
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # 保存图像
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            # 世界坐标系下的点云
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )

            # 验证方法是否有效
            im_height, im_width = camera.image_shape
            # -- 将点投影到 (u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """Main function."""
    # 加载 Kit 辅助组件
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
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

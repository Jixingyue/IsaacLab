# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    # 指定日志必须写入 logs/docker_tutorial
    log_dir_path = os.path.join("logs")
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # 在容器中，绝对路径将会是
    # /workspace/isaaclab/logs/docker_tutorial，因为
    # 所有 Python 执行都通过 /workspace/isaaclab/isaaclab.sh 完成
    # 并且调用进程的路径会是 /workspace/isaaclab
    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")

    # 初始化仿真上下文
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # 设置主相机
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 启动模拟器
    sim.reset()
    # 现在已准备就绪
    print("[INFO]: Setup complete...")

    # 准备累计 sim_time
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    # 打开日志文件
    with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:
        # 运行物理仿真
        while simulation_app.is_running():
            log_file.write(f"{sim_time}" + "\n")
            # 执行一步仿真
            sim.step()
            sim_time += sim_dt


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

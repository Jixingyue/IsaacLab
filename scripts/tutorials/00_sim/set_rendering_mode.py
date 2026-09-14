# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(
    description="Tutorial on viewing a warehouse scene with a given rendering mode preset."
)
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # 渲染模式包括 performance、balanced 和 quality
    # 注意，命令行参数（--rendering_mode）中指定的 rendering_mode 优先级高于
    # 这里的 Render Config 设置
    rendering_mode = "performance"

    # carb 设置字典可以包含任意 rtx carb 设置，并覆盖原生预设
    carb_settings = {"rtx.reflections.enabled": True}

    # 初始化渲染配置
    render_cfg = sim_utils.RenderCfg(
        rendering_mode=rendering_mode,
        carb_settings=carb_settings,
    )

    # 使用渲染配置初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(render=render_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 将相机放置在医院大厅区域
    sim.set_camera_view([-11, -0.5, 2], [0, 0, 0.5])

    # 加载医院场景
    hospital_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=hospital_usd_path)
    cfg.func("/Scene", cfg)

    # 启动模拟器
    sim.reset()

    # 现在已准备就绪
    print("[INFO]: Setup complete...")

    # 运行仿真并查看场景
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

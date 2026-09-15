# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


def main():
    """Main function."""
    # 创建环境配置
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # 搭建 RL 环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 仿真物理
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # 采样随机动作
            joint_efforts = torch.randn_like(env.action_manager.action)
            # 执行环境步进
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # 打印当前杆的角度
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # 翻译过来就是："把 0 号那台游戏机里，杆子现在歪了几弧度，打印出来给我看看。"

            # 更直观的进一步验证：
            print("[Env 0] 奖励:", rew[0].item(), "  失败:", terminated[0].item(), "  超时:", truncated[0].item())
            # 你会看到：因为动作是随机瞎推的（torch.randn_like），分数普遍很低，
            # 而且很快就有一堆环境的 失败 变成 True——这就是 AI 训练开始前"还啥都不会"的样子。

            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()

'''

关于强化学习的说明：


1.强化学习训练的时候，程序就靠这5个东西学习：
看一眼现在的情况（obs）
    ↓
决定往哪推（action）
    ↓
仿真器算出结果，给你 5 张纸条
    ↓
如果 got 高分 → 记住"这动作好，下次还这么干"
如果 GAME OVER → 记住"这动作不行，别再犯"
    ↓
回到最上面，重复几百万次

AI 啥都不懂，只会一件事：想办法让 rew 这个分数越高越好。
terminated 就是告诉它"这条路走死了，重来"，
truncated 告诉它"这局到此为止，但别灰心"。


2.这 5 个东西就是仿真器在告诉你："这一步做得怎么样"。
你在里面同时控制几千台"小车顶着杆子"的游戏机，每走一步，机器就还给你 5 张纸条：
变量                    大白话                     就像你玩游戏的……
obs      现在场上是什么情况（杆子歪了多少、车跑多快）    眼睛看到的画面
rew      这一步我给你打几分                         得分
terminated   游戏失败了（杆子倒了 / 车冲出边界）       GAME OVER
truncated     时间到了，本局结束（不是你的错）         倒计时归零
info          一些额外的备注信息                     结算画面上的小字

'''
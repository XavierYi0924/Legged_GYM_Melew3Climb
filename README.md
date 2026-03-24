# Legged_GYM_Melew3Climb 

本仓库包含了基于深度强化学习（DRL）的 **MELEW-3** 四轮足机器人运动控制框架。项目依托于 NVIDIA Isaac Gym 与 `legged_gym`，利用 PPO（Proximal Policy Optimization）算法，结合**课程学习（Curriculum Learning）**，成功训练出了 MELEW-3 的垂直攀爬策略。

## 核心算法与特性
* **强化学习算法 (PPO)**：使用高度优化的 `rsl_rl` 库中的 PPO 算法进行策略迭代与训练。
* **课程学习 (Curriculum Learning)**：在训练攀爬策略时引入了课程学习机制，通过逐步增加环境难度（如地形倾斜度、摩擦力或指令复杂度），引导机器人平稳收敛到最终的攀爬动作。
* **自定义训练环境**：在 `melew3_flat` 环境中定义了针对攀爬任务的复杂奖励函数（Reward Functions）与观测空间（Observations）。
* **Isaac Gym 物理仿真**：利用 GPU 加速的大规模并行仿真，极大缩短了训练周期。

## 核心项目结构
* `isaacgym/` : NVIDIA 官方提供的硬件加速强化学习仿真环境包。
* `rsl_rl/` : 包含 PPO 算法核心实现的算法库。
* `gym/legged_gym/` : 机器人的训练主目录。
  * `legged_gym/envs/melew3_flat/` : **核心环境配置目录**。
    * `melew3_flat.py` : 包含攀爬任务的环境定义、奖励函数及物理交互逻辑。
    * `melew3_flat_config.py` : 包含 PID 参数、课程学习阈值、动作空间比例等核心训练超参数。
* `logs/` : 存放训练生成的 TensorBoard 日志以及**训练成功的策略模型节点（Checkpoints / .pt 文件）**。

## 快速运行 (Quick Start)

*(注意：运行前请确保已激活包含 PyTorch 与 Isaac Gym 的 Conda 虚拟环境)*

**1. 启动训练 (Training)**
```bash
python legged_gym/scripts/train.py --task=melew3_flat

#!/bin/bash

# 测试改进的碰撞检测训练脚本
echo "=== MELEW3 改进碰撞检测训练 ==="

# 切换到正确目录
cd /home/mrp/gym/legged_gym/legged_gym/scripts

# 激活conda环境
source /home/mrp/anaconda3/etc/profile.d/conda.sh
conda activate gym

echo "当前目录: $(pwd)"
echo "Python环境: $(which python)"
echo "开始训练 - 使用改进的碰撞检测..."

# 启动训练
python train.py --task=melew3_climb --headless

echo "训练结束"

# Biped Demo — Isaac Lab 双足机器人强化学习项目

## 概述

本项目基于 Isaac Lab 模板，为自定义 12-DOF 双足机器人（v2.4.1）搭建 Manager-based 强化学习训练环境。

**机器人结构：**
- 12 个关节（每条腿 6 自由度）
- 每条腿关节链：hip_pitch → hip_roll → hip_yaw → knee_pitch → ankle_pitch → ankle_roll

**当前任务：**
- `Biped-velocity-v0` — 平面速度跟踪

## 安装

1. 安装 Isaac Lab（参考[官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)）

2. 以可编辑模式安装本扩展：
```bash
python -m pip install -e source/biped_demo
```

## 使用方法

### 列出可用环境
```bash
python scripts/list_envs.py
```

### 验证环境（零动作智能体）
```bash
python scripts/zero_agent.py --task Biped-velocity-v0
```

### 验证环境（随机动作智能体）
```bash
python scripts/random_agent.py --task Biped-velocity-v0 --num_envs 14
```

### 训练
```bash
python scripts/rsl_rl/train.py --task Biped-velocity-v0 --num_envs=400 --max_iterations 1000 --headless
```

### 可视化训练曲线
```bash
tensorboard --logdir logs
```

### 评估
```bash
python scripts/rsl_rl/play.py --task Biped-velocity-v0 --num_envs 16
```

## 项目结构

```
biped_demo/
├── scripts/           # 训练和评估脚本
├── source/biped_demo/ # Python 扩展包
│   └── biped_demo/
│       └── tasks/manager_based/biped_demo/
│           ├── biped_env_cfg.py      # 场景 + MDP 配置
│           ├── biped_velocity.py     # 速度跟踪环境入口
│           ├── assets/robot/biped.py # 机器人模型配置
│           ├── agents/               # PPO 配置
│           └── mdp/                  # 自定义 MDP 项
└── logs/              # 训练日志和检查点
```

## 机器人模型

URDF 文件位于 `go2/unitree_model/rc_v.2.4.1/urdf/v2.4.1.urdf`。

训练前需要将 URDF 转换为 USD 格式：
1. 打开 Isaac Sim
2. File → Import → URDF，选择 `v2.4.1.urdf`
3. 设置：Moveable Base、Stiffness、Force Drive、Allow Self-Collision
4. 导出 USD 到 `source/biped_demo/biped_demo/tasks/manager_based/biped_demo/assets/robot/usd/`
5. 更新 `biped.py` 中的 `usd_path`

# Isaac Lab 双足机器人强化学习项目 — 配置经验总结

> 生成日期: 2026-07-19
> 适用: Isaac Lab 4.x + Newton 物理引擎 + RTX 3050 Laptop (4GB)
> 机器人: 自定义 12-DOF 双足 (v2.4.1 URDF)

---

## 一、项目结构

```
RobotProject/biped_demo/
├── source/biped_demo/biped_demo/tasks/manager_based/biped_demo/
│   ├── __init__.py                  # Gym 环境注册 (Biped-velocity-v0)
│   ├── biped_velocity.py            # 环境入口（继承 BipedEnvCfg）
│   ├── biped_env_cfg.py             # 场景 + MDP 全配置
│   ├── assets/robot/
│   │   ├── biped.py                 # 机器人 ArticulationCfg
│   │   └── usd/                     # URDF→USD 转换后的模型
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py        # PPO 训练参数
│   └── mdp/
│       ├── __init__.py              # lazy_export()
│       ├── __init__.pyi             # from isaaclab.envs.mdp import *  ← 关键！
│       └── rewards.py               # 自定义 MDP 项（占位）
├── scripts/
│   ├── debug_obs.py                 # 观测调试工具
│   ├── import_urdf.py               # URDF→USD 转换脚本
│   ├── random_agent.py / zero_agent.py
│   └── rsl_rl/train.py / play_rsl_rl.py / cli_args.py
└── logs/rsl_rl/biped_demo/           # 训练输出（按时间戳分目录）
```

---

## 二、机器人关节结构（12 DOF × 5 Nm）

```
每条腿 6 关节链: base_link
  ├── {l,r}_leg_pitch_joint   → hip pitch   (pitch, ±1.57 rad)
  ├── {l,r}_leg_roll_joint    → hip roll    (roll,  -1.57~0.5 / -0.5~1.57)
  ├── {l,r}_leg_yaw_joint     → hip yaw     (yaw,   ±1.57 rad)
  ├── {l,r}_knee_pitch_joint  → knee pitch  (pitch, ±1.57 rad)
  ├── {l,r}_ankle_pitch_joint → ankle pitch (pitch, ±0.5 rad)
  └── {l,r}_ankle_roll_joint  → ankle roll  (roll,  ±0.5 rad)

总质量: ~3.4 kg | 电机: 5 Nm / 10 rad/s (URDF)
```

### 机器人配置 (biped.py) 关键参数

```python
stiffness=40.0    # PD 位置增益（越高跟踪越硬）
damping=2.0       # PD 阻尼（越高越不抖）
effort_limit=5.0  # 电机最大力矩（来自 URDF）
velocity_limit=10.0

# 初始站姿（膝微弯 + 踝补偿）
init_state.joint_pos = {
    ".*_leg_pitch_joint": 0.0,
    ".*_leg_roll_joint": 0.0,
    ".*_leg_yaw_joint": 0.0,
    ".*_knee_pitch_joint": -0.5,
    ".*_ankle_pitch_joint": 0.25,
    ".*_ankle_roll_joint": 0.0,
}
init_state.pos = (0, 0, 0.35)   # 基座高度 ~0.35m
```

**执行器调试经验：**
- stiffness 太低 (20) + damping 太低 (0.5) → 又软又抖
- stiffness=40, damping=2.0 → 干脆、不抖
- 电机 < 5 Nm，effort_limit 保持 URDF 原值

---

## 三、奖励函数最终配置（converged version）

```python
track_lin_vel_xy_exp     weight=2.0    # 速度跟踪（主目标）
track_ang_vel_z_exp      weight=1.0    # 转向跟踪
termination_penalty      weight=-200   # 摔倒惩罚
flat_orientation_l2      weight=-1.0   # 保持直立
action_rate_l2           weight=-0.01  # 动作平滑
```

**奖励设计踩坑记录：**
1. 惩罚项太多（joint_deviation×3 + desired_contacts + base_height + ...）→ 策略被惩罚淹没，entropy 不降
2. 保持简单——只保留"往前走 + 别倒 + 别抖"
3. 等基本步态收敛后再逐步加回足部接触、关节约束等

---

## 四、PPO 超参数

```python
actor:  [512, 256, 128]   # 观测 48 维足够
critic: [256, 128, 64]    # 比 Go2 深（Go2 用 [32,32]）

entropy_coef = 0.001      # 低熵 → 策略专注收敛（不是 0.01！）
learning_rate = 1e-3
schedule = "adaptive"
desired_kl = 0.01
clip_param = 0.2
gamma = 0.99, lam = 0.95
num_steps_per_env = 124
decimation = 4            # 50 Hz 策略频率
sim.dt = 0.005            # 200 Hz 物理
```

**PPO 踩坑记录：**
- entropy_coef=0.01 → entropy 在 700 步后不降反升（21 → 21），策略靠随机探索得分
- 改为 0.001 后 entropy 稳定下降（18 → 5）

---

## 五、观测项 × 真实传感器对照

| 观测 (48 维) | 真实获取 | 可靠性 |
|-------------|---------|--------|
| base_lin_vel (3) | IMU积分 + 视觉里程计 / EKF | 中等 |
| base_ang_vel (3) | IMU 陀螺仪 | 高 |
| projected_gravity (3) | IMU 加速度计 | 高 |
| velocity_commands (3) | 上位机下发 | 直接可用 |
| joint_pos (12) | 关节编码器 | 高 |
| joint_vel (12) | 编码器差分 | 高 |
| last_action (12) | 策略自身缓存 | 直接可用 |

**sim-to-real 关键:** 线速度在实物上无法直接测量，需要 EKF 融合 IMU + 足端里程计。

---

## 六、所有踩过的坑

### 坑 1: `mdp/__init__.pyi` 缺失
- **现象**: `AttributeError: module 'mdp' has no attribute 'JointPositionActionCfg'`
- **原因**: `lazy_export()` 需要 `.pyi` stub 声明 `from isaaclab.envs.mdp import *`
- **修法**: 创建 `mdp/__init__.pyi`，加 `from isaaclab.envs.mdp import *`

### 坑 2: URDF 导入失败
- **现象**: Isaac Sim GUI 报 "Failed to convert assets"
- **原因**: 中文路径名 + `package://` mesh 引用 + GUI bug
- **修法**: 清理 URDF（ASCII robot_name + 绝对 mesh 路径），用 Python API 导入

### 坑 3: `sim.physx` 不存在
- **现象**: `AttributeError: 'SimulationCfg' object has no attribute 'physx'`
- **原因**: Isaac Lab 4.x 用 Newton 引擎，不再是 PhysX
- **修法**: 删掉 `self.sim.physx.gpu_max_rigid_patch_count`

### 坑 4: 函数名不匹配
- **现象**: `AttributeError: 'joint_deviation_l2'`
- **原因**: 实际名是 `joint_deviation_l1`（L1 不是 L2）
- **修法**: 用 `grep "def xxx"` 在 Isaac Lab 源码里确认函数名

### 坑 5: 多环境 NaN
- **现象**: 单环境 (num_envs=1) 正常，64 环境 NaN
- **原因**: 部分 env 中机器人摔倒后物理爆炸，观测出 NaN
- **修法**: 加 `bad_orientation` 终止条件（摔倒前就杀掉），减少 reset 随机化范围

### 坑 6: entropy 不降
- **现象**: entropy 从 17 → 21，reward 不涨
- **原因**: entropy_coef=0.01 + 13 个惩罚项淹没主奖励信号
- **修法**: entropy_coef → 0.001，奖励项 13 → 5

### 坑 7: 视频录制显存不足
- **现象**: `_ArrayMemoryError: Unable to allocate 3.52 MiB`
- **原因**: 长时间训练后 GPU 显存碎片化，渲染管线找不到连续内存
- **修法**: 重开终端，减少 `--video_length`

---

## 七、训练指标解读

| 指标 | 好 | 差 | 含义 |
|------|----|----|------|
| entropy | ↓ 下降 | ↑ 上升或不变 | 策略探索程度 |
| track_lin_vel_xy_exp | > 0.9 | < 0.3 | 速度跟踪质量 |
| action_std | < 0.5 | > 1.0 | 动作噪声水平 |
| mean episode length | = 1000 | < 500 | 存活时间（摔倒即死） |
| termination_penalty | = 0 | < 0 | -200 表示有机器人摔倒 |
| success_rate | > 0.8 | < 0.2 | 完美跟踪比例（波动大） |

---

## 八、常用命令

```powershell
# 激活环境
F:\isaacsim\env_isaacsim\Scripts\Activate.ps1

# debug: 单环境观测检查
python scripts/debug_obs.py --task Biped-velocity-v0 --num_envs 1 --headless

# 随机动作验证
python scripts/random_agent.py --task Biped-velocity-v0 --num_envs 1 --headless

# 训练 (64 envs, 4GB 显存上限)
python scripts/rsl_rl/train.py --task Biped-velocity-v0 --num_envs=64 --max_iterations 8000 --headless

# 续训
python scripts/rsl_rl/train.py --task Biped-velocity-v0 --num_envs=64 --resume --load_run <时间戳目录> --headless

# 录视频
python scripts/rsl_rl/play_rsl_rl.py --task Biped-velocity-v0 --num_envs 1 --checkpoint "<路径>/model_XXXX.pt" --video --video_length 500 --headless

# 查看训练曲线
tensorboard --logdir F:\RobotProject\biped_demo\logs
```

---

## 九、高算迁移 check list

1. 上传整个 `biped_demo/` + `go2/unitree_model/rc_v.2.4.1/` (URDF)
2. 在高算上装 Isaac Lab（同版本）
3. `pip install -e source/biped_demo`
4. URDF → USD 转换（或直接传 USD 文件）
5. `--num_envs=4096`（充分利用 GPU）
6. `--num_envs=64` 仅适用于 4GB 显存

---

## 十、新对话快速启动提示

> 我有一个 Isaac Lab 双足机器人强化学习项目在 `F:\RobotProject\biped_demo`，
> 任务是自定义 12-DOF 双足机器人的速度追踪。
> CPU: i5 + RTX 3050 Laptop (4GB VRAM)。
> Isaac Lab 在 `F:\IsaacLab`，环境在 `F:\isaacsim\env_isaacsim`。
> 详细配置经验在 `F:\RobotProject\biped_demo\ISAACLAB_BIPED_NOTES.md`（本文件）。
> 渲染窗口不能弹出（显存不足），用 `--video` + `--headless`。
> Go2 项目经验在 `F:\RobotProject\go2\ISAACLAB_NOTES.md`。

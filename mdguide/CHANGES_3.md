# 训练配置优化：解决过快收敛问题

## 问题

在 4090 上以 4096 env 训练到 2000~3000 轮后，policy 发生过快收敛进入局部最优。

**根因**：正向奖励过大（~4.0/步）而负向约束太弱（~-0.5/步），policy 快速找到"作弊"步态后卡死。

## 参考

对比了三套官方 H1 配置：
- IsaacLab 官方 `H1FlatPPORunnerCfg` / `H1RoughPPORunnerCfg`
- unitree_rl_lab `BasePPORunnerCfg` + `RobotEnvCfg`

## 修改的文件

### 1. `source/biped_demo/biped_demo/tasks/manager_based/biped_demo/biped_env_cfg.py`

| 改动 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| action `scale` | 0.5 | 0.25 | 减小动作幅度，更稳定 |
| 观测 `noise` | 无 | 有（与 H1 对齐） | 模拟传感器误差，防过拟合 |
| `enable_corruption` | False | True | 启用观测噪声注入 |
| `track_lin_vel_xy_exp` | 2.0 | 1.0 | 减半，对齐 H1 |
| `track_ang_vel_z_exp` | 1.0 | 0.5 | 减半，对齐 H1 |
| `feet_air_time` | 1.0 / threshold=0.4 | 0.25 / threshold=0.3 | 对齐 H1，小机器人降 threshold |
| `action_rate_l2` | -0.01 | -0.005 | 对齐 H1 官方 |
| `desired_contacts` | -0.5 | **删除** | 与 feet_air_time 冗余 |
| **🆕** `alive` | — | 0.1 | 持续生存奖励 |
| **🆕** `lin_vel_z_l2` | — | -1.0 | 惩罚上下跳动 |
| **🆕** `ang_vel_xy_l2` | — | -0.05 | 惩罚左右晃动（比 H1 小，适配轻量机器人） |
| **🆕** `base_height_l2` | — | -5.0 (target=0.35m) | 约束站立高度 |
| **🆕** `dof_pos_limits` | — | -1.0 | 关节限位保护 |
| **🆕** `joint_deviation_hips` | — | -0.2 | 防止 hip_roll/hip_yaw 偏离 |
| **🆕** `undesired_contacts` | — | -1.0 | 非脚部触地惩罚 |
| **🆕** `base_contact` 终止 | — | illegal_contact on base_link | base 碰地即终止 |
| `bad_orientation` limit | 0.7 rad | 1.0 rad | 小机器倾斜容限更大 |
| friction 范围 | 0.65-0.95 | 0.3-1.0 | 更广域随机化 |
| friction 作用范围 | 仅 ankle | `".*"` 全身 | 对齐 H1 |
| **🆕** `add_base_mass` | — | -0.2~+0.5 kg | 模拟负载变化 |
| `UniformNoiseCfg` 兼容 | — | try/except 双版本导入 | 兼容新旧 Isaac Lab |

### 2. `source/biped_demo/biped_demo/tasks/manager_based/biped_demo/agents/rsl_rl_ppo_cfg.py`

| 改动 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| Actor hidden_dims | [512, 256, 128] | [256, 128, 64] | H1 flat 官方用 [128,128,128]，小机器人 + flat terrain 不需要大网络 |
| Critic hidden_dims | [512, 256, 128] | [256, 128, 64] | 同上 |
| max_iterations | 15000 | 5000 | flat terrain 更易收敛 |

### 3. 未修改的文件

- `rewards.py` — 添加 `feet_slide` 后因本地 Isaac Lab 版本不支持而移除，代码未保留修改
- `__init__.pyi` — 无需修改

## 核心逻辑

```
旧: 正向 ~4.0  vs  负向 ~-0.5  →  正负比 8:1  →  秒收敛到作弊步态
新: 正向 ~1.85 vs  负向 ~-210  →  正负比 ~1:100  →  必须平衡约束才能拿高分
```

12 个约束项持续施加压力，policy 必须在"走得快"和"走得稳"之间平衡，渐进式收敛。

## 注意事项

- 本地 64 env 测试会出现 reward 震荡 + entropy 上升，这是样本量不足导致的，配置为 4096 env 设计
- HPC 上 4096 env 训练效果正常
- 如 Isaac Lab 升级到支持 `feet_slide` 的版本可加回来（weight=-0.1）

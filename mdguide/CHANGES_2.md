# 第二步改动: 教师-学生架构 + 步态修复

> 在第一版（能站稳、速度跟踪基本收敛）基础上的增量改动

---

## 一、新增 Privileged 观测（教师额外 11 维）

在 `biped_env_cfg.py` 的 `ObservationsCfg` 里加了 `PrivilegedCfg` 组：

```python
class PrivilegedCfg(ObsGroup):
    base_lin_vel_truth = ObsTerm(func=mdp.base_lin_vel)       # 3 维
    foot_contact = ObsTerm(                                    # 2 维
        func=mdp.foot_contact_state,
        params={"sensor_cfg": ..., "threshold": 2.0})
    foot_grf = ObsTerm(func=mdp.foot_grf, params={...})       # 6 维
```

对应在 `mdp/rewards.py` 里写了两个自定义函数：

```python
foot_contact_state(env, sensor_cfg, threshold)  # [N, 2] 0/1 接触状态
foot_grf(env, sensor_cfg)                       # [N, 6] 双脚 GRF XYZ
```

**踩坑**: `contact_sensor.data.net_forces_w` 返回 ALL bodies（13 个），不是只返回过滤后的脚踝。必须 `[:, sensor_cfg.body_ids, :]` 索引。

---

## 二、新增 `feet_air_time` 奖励

在 `biped_env_cfg.py` 的 `RewardsCfg` 里加了：

```python
feet_air_time = RewTerm(
    func=mdp.feet_air_time_positive_biped,
    weight=1.0,
    params={"command_name": "base_velocity",
            "sensor_cfg": "...ankle_roll_link", "threshold": 0.6})
```

来源: `isaaclab_tasks.core.velocity.mdp`（H1 官方用这个）

**效果**: 奖励"恰好一只脚着地"的时间，自然产生交替迈步。唯一能直接修复瘸腿步态的奖励。

**依赖**: 在 `mdp/__init__.pyi` 加了 `from isaaclab_tasks.core.velocity.mdp import feet_air_time_positive_biped`

---

## 三、新增足端摩擦域随机化

在 `EventsCfg` 里加了 startup 事件：

```python
randomize_friction = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
        "static_friction_range": (0.65, 0.95),
        "num_buckets": 256,
    })
```

**效果**: 每 env 随机分配足端摩擦系数。策略隐式学会应对不同地面，不需要把 μ 暴露为观测。

---

## 四、删除 `foot_symmetry`

之前尝试用 `foot_symmetry`（双脚承重 50/50 奖励）修瘸腿，但正常走路时体重在双脚间交替转移，这个奖励直接跟走路矛盾。**已删。**

---

## 五、PPO 超参数

`entropy_coef` 从 0.01 改到 0.001（太高反而不收敛，策略靠随机探索得分）。

---

## 六、传感器修复

`contact_forces` 加回 SceneCfg（`desired_contacts` 和 `foot_contact_state` 都需要它）。

`foot_contact_state` / `foot_grf` 两次修复维度 bug：
1. `net_forces_w` 是 `[N, bodies*3]` 不是 `[N, bodies, 3]`
2. `net_forces_w` 返回 ALL bodies，需用 `body_ids` 索引

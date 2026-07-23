# 4096-env PPO 训练崩溃分析与修复

> 2026-07-23 | 问题：训练在 2000 轮附近剧烈震荡然后崩溃

---

## 一、现象

从 1800 轮开始续训后，2000 轮附近发生：
- **Reward** 剧烈震荡 → 混乱下降
- **action_rate_l2** 剧烈震荡
- **Entropy** 降到 -40（sigma ≈ 0.04）
- **time_out** 和 **episode_length** 大幅波动后下降

→ **典型过拟合崩溃（policy collapse）**

---

## 二、根因

**本地 64 envs 的 PPO 参数直接搬到了 HPC 4096 envs，完全不匹配。**

| 参数 | 本地 64 env | HPC 4096 env | 官方 H1 4096-env | 问题 |
|------|------------|-------------|-----------------|------|
| `num_steps_per_env` | 124 | 124 | **24** | 每轮 508k 样本，是官方的 **5 倍** |
| 每轮样本量 | ~8k | ~508k | ~98k | 单轮样本太多 → 梯度方差太低 → 过拟合一轮 |
| `entropy_coef` | 0.001 | 0.005 | **0.01** | 探索力只有官方一半 |
| `critic dims` | [256,128,64] | [256,128,64] | **[512,256,128]** | 值网络太弱，价值估计不准 |
| `std_range` | 无 (min=1e-6) | 无 | 无 | sigma 可降到接近 0 → 策略完全僵化 |

**核心矛盾**：`num_steps_per_env=124` × 4096 envs = **508,000 样本/轮**。PPO 用 4 个 mini-batch 跑 5 个 epoch，每个 sample 被反复学了 5 遍。几百轮后策略就把这 508k 样本过拟合了——开始对训练数据产生极度精确的预测（sigma → 0），但对环境微扰（domain randomization、重置波动）失去泛化能力，一碰就崩。

**为什么本地 64 envs 没这个问题？**
- 64 × 124 = 7,936 样本/轮，批量大但不过分
- 小 batch 自带更高梯度噪声 → 天然正则化 → 不会过拟合
- 批大小差了 64 倍，训练动力学会完全不同

---

## 三、修复（对齐 Isaac Lab 官方 H1 4096-env 配置）

文件：`source/biped_demo/biped_demo/tasks/manager_based/biped_demo/agents/rsl_rl_ppo_cfg.py`

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24          # H1 官方：24（曾为 124，仅适合 <100 envs）
    max_iterations = 15000          # 更多轮补偿短步长
    save_interval = 100
    experiment_name = "biped_demo"
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_range=(0.15, 1.0),     # sigma 不低于 0.15
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],            # H1 官方，曾为 [256,128,64]
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,             # H1 官方，曾为 0.005
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

额外修改：`F:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\rl_cfg.py` 加上 `std_range` 字段使配置可传递。

> ⚠️ **这个修改在 Isaac Lab 源码仓库内，没有 git commit。** 如果重新 clone Isaac Lab 或切换分支，需要重新加一次。具体改动：

```python
# F:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\rl_cfg.py
# 在 GaussianDistributionCfg 类中，std_type 下面加一行：

        std_range: tuple[float, float] = (1e-6, 1.0e6)
        """The (min, max) clamping range for the standard deviation. Default is (1e-6, 1e6)."""
```

---

## 四、教训

1. **不能直接把小环境参数放大**——64 envs 的配置扔到 4096 envs 上，批大小差 64 倍，训练动力学会完全不同
2. **看官方 reference**——Isaac Lab 的 H1 任务就是 4096 envs，`num_steps_per_env=24` 不是随便选的
3. **熵崩塌是症状不是病因**——调高 entropy_coef 只是延缓，真正的修复是调整 batch size（num_steps_per_env）
4. **改训练超参后不能续训**——优化器状态不兼容，必须重头来。合理做法是保留旧 checkpoint 当 baseline

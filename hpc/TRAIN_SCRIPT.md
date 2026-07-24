# HPC 训练脚本说明

> 2026-07-24 | v2 配方 | Isaac Lab 10.0.0 + Isaac Sim 6.0.0 | 4096 envs

---

## 当前脚本（v2）

PPO 参数对齐 Isaac Lab 官方 H1 4096-env 配置。详见 `mdguide/PPO_COLLAPSE_FIX.md`。

```bash
#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -J biped_v2
#SBATCH -o train_v2_%j.log
#SBATCH -e train_v2_%j.log

ISAACLAB_PATHS=$(find $HOME/IsaacLab/source -maxdepth 1 -mindepth 1 -type d | tr '\n' ':')
export SINGULARITYENV_PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind $HOME/biped_demo:/workspace/biped_demo \
    --bind $HOME/IsaacLab/source:/opt/isaaclab_source \
    --env DISPLAY=:0 \
    --env QT_QPA_PLATFORM=offscreen \
    $HOME/biped-sandbox.sif \
    /isaac-sim/python.sh /workspace/biped_demo/scripts/rsl_rl/train.py \
    --task Biped-velocity-v0 \
    --num_envs 4096 \
    --max_iterations 15000 \
    --headless
```

---

## v2 PPO 参数 vs v1

| 参数 | v1（崩溃） | v2（当前） | 依据 |
|------|-----------|-----------|------|
| `num_steps_per_env` | 124 | **24** | H1 官方 4096-env |
| `entropy_coef` | 0.005 | **0.01** | H1 官方值 |
| `std_range` | (0.15, 1.0) | **(0.05, 10.0)** | 上限放开，防 sigma 僵化 |
| `critic dims` | [256,128,64] | **[512,256,128]** | H1 官方值 |
| `max_iterations` | 5000 | **15000** | 24步需更多轮 |

---

## 成功关键（SLURM/容器层）

| 配置 | 解决的问题 |
|------|-----------|
| `--env DISPLAY=:0` + `QT_QPA_PLATFORM=offscreen` | OmniHub `XOpenDisplay` segfault |
| `--bind /dev/shm:/dev/shm` | 4096 环境共享内存不足 → SIGBUS |
| `--mem=64G` | PhysX 缓冲 + PyTorch 模型内存 |
| PYTHONPATH 注入 | 覆盖容器内置旧版 isaaclab |
| `.kit` `asset_root.cloud` 改本地 | 计算节点无外网，S3 资产不可达 |
| `terrain_type="usd"` + 本地 usda | 绕过云端 ground plane USD |
| Torch 2.5.1 cu124 | HPC 驱动 CUDA 12.4，torch 2.11 需要 12.8 |

---

## 已知限制

- **GPU 驱动 550 (CUDA 12.4)**：torch 永久锁在 2.5.1+cu124，无法升级
- **Isaac Lab `rl_cfg.py` 修改未 commit**：加了 `std_range` 字段，重新 clone 后需补回去
- **警告刷屏** — PhysX 的 `Failed to find rigid body` / `contact report API` 是良性警告，4096 环境会刷几万行，忽略即可

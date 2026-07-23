# HPC 训练脚本说明

> 2026-07-23 | 4096 环境，Isaac Lab 10.0.0 + Isaac Sim 6.0.0，成功运行

---

## 脚本

```bash
#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J biped_train
#SBATCH -o train_%j.log
#SBATCH -e train_%j.log

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
    --max_iterations 5000 \
    --headless
```

---

## 成功关键

| 配置 | 问题 | 为什么关键 |
|------|------|------------|
| `--env DISPLAY=:0` | OmniHub 后台线程调 `XOpenDisplay` → segfault 崩溃 | 骗过 Omniverse 平台检测，告诉它"有显示" |
| `--env QT_QPA_PLATFORM=offscreen` | Qt 尝试打开 X11 显示 → 崩溃 | 强制 Qt 用离屏模式，不碰物理显示 |
| `--bind /dev/shm:/dev/shm` | 容器内 `/dev/shm` 太小（SIF 默认 64MB）→ SIGBUS | 挂载宿主机 `/dev/shm`，4096 环境共享内存够用 |
| `--mem=64G` | 32G 不够 4096 环境 | 4096 个 PhysX 刚体缓冲 + PyTorch 模型 ≈ 40-50G |
| `--mem=32G`（测试用） | 4 环境测试够用 | 测试阶段不需要大内存 |
| PYTHONPATH 注入 | 容器 isaaclab 版本太老/API 不匹配 | 用宿主 isaaclab 10.0.0 源码覆盖容器内置版本 |
| `.kit` 文件 `asset_root.cloud` 改本地 | 计算节点无外网，S3 资产拉不下来 | 改为本地路径 + 下载对应 USD 文件 |
| `terrain_type="usd"` + 本地 usda | `terrain_type="plane"` 走 GroundPlaneCfg 调 S3 URL | 完全绕过云端资产依赖 |
| Torch 2.5.1 cu124（用户目录） | 容器内置 torch 2.11 需要 CUDA 12.8，HPC 驱动只到 12.4 | 在 `~/.local` 装兼容版本，通过 PYTHONPATH 覆盖 |

---

## 注意事项

1. **不要删除 `train_*.log`** — 删除后无法实时看训练输出，只能通过 checkpoint 文件判断进度
2. **实时监控**：
   ```bash
   tail -f ~/train_1623303.log       # 看 SLURM 输出
   find ~/logs/rsl_rl/biped_demo -name "model_*.pt"  # 看 checkpoint
   ```
3. **TensorBoard** — 日志目录 `~/logs/rsl_rl/biped_demo/`，可下载到本地查看
4. **警告刷屏** — `Failed to find rigid body` / `contact report API` 是 PhysX 在几何子网格上的良性警告，4096 环境下会刷几万行，无需理会

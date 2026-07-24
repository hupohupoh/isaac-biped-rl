# HPC 常用命令

> 2026-07-24 | 当前版本: v2 (num_steps_per_env=24, std_range=(0.05,10.0))

---

## 一、从头训练（v2 配方）

```bash
cat << 'EOF' > ~/biped_demo/hpc/train_biped_v2.sh
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
EOF

sbatch ~/biped_demo/hpc/train_biped_v2.sh
```

---

## 二、续训脚本

```bash
cat << 'EOF' > ~/biped_demo/hpc/train_biped_resume.sh
#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -J biped_resume
#SBATCH -o resume_%j.log
#SBATCH -e resume_%j.log

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
    --resume \
    --load_run ".*" \
    --headless
EOF

sbatch ~/biped_demo/hpc/train_biped_resume.sh
```

> 如果续训时需要指定精确 checkpoint：加上 `--checkpoint "model_XXXX.pt"`，`--load_run` 改成具体目录名如 `"2026-07-23_14-30-17"`。但注意：**改了 num_steps_per_env 等结构参数后不能续训**，只能从头来。

---

## 三、HPC 常用指令

```bash
# ---- 作业管理 ----
squeue -u $USER                                    # 看自己的作业
scancel <JOBID>                                    # 取消作业

# ---- 看训练进度 ----
ls -lt ~/logs/rsl_rl/biped_demo/*/model_*.pt | head -10   # checkpoint 列表
tail -f ~/train_v2_<JOBID>.log                    # 实时看输出
grep "Learning iteration" ~/train_v2_*.log | tail -5       # 最近几轮

# ---- 文件被删后看输出 ----
squeue -j <JOBID>                                  # 先查节点名
ssh <节点名> "tail -f /proc/\$(pgrep -f 'train.py' | head -1)/fd/1"

# ---- 容器内操作 ----
source ~/biped_demo/hpc/env.sh                     # 加载环境变量
singularity exec $HOME/biped-sandbox.sif /isaac-sim/python.sh -m pip install -e /workspace/biped_demo/source/biped_demo
singularity exec $HOME/biped-sandbox.sif /isaac-sim/python.sh -c "from isaaclab.app.sim_launcher import add_launcher_args; print('OK')"
```

---

## 四、当前配方参数（v2）

| 参数 | 值 | 说明 |
|------|----|------|
| `num_steps_per_env` | 24 | H1 官方 4096-env 值 |
| `entropy_coef` | 0.01 | H1 官方值 |
| `std_range` | (0.05, 10.0) | sigma 下限 0.05，上限放开 |
| `critic dims` | [512, 256, 128] | H1 官方值 |
| `num_envs` | 4096 | |
| `max_iterations` | 15000 | 24 步需更多轮 |

## 五、已知限制

- **GPU 驱动 550 (CUDA 12.4)**：torch 永久锁在 2.5.1+cu124，RSL-RL 锁在 ≤5.x。详见 `mdguide/PPO_COLLAPSE_FIX.md`。
- **Isaac Lab 源码修改未 commit**：`F:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\rl_cfg.py` 加了 `std_range` 字段。重新 clone 后需补回去。

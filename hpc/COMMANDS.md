# HPC 常用命令

> 2026-07-23

---

## 一、训练脚本

```bash
cat << 'EOF' > ~/biped_demo/hpc/train_biped.sh
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
EOF

sbatch ~/biped_demo/hpc/train_biped.sh
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
#SBATCH -t 48:00:00
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
    --max_iterations 5000 \
    --resume \
    --headless
EOF

sbatch ~/biped_demo/hpc/train_biped_resume.sh
```

---

## 三、HPC 常用指令

```bash
# ---- 作业管理 ----
squeue -u $USER                          # 看自己的作业
scancel <JOBID>                          # 取消作业

# ---- 看训练进度 ----
ls -lt ~/logs/rsl_rl/biped_demo/*/model_*.pt | head -10   # checkpoint 列表
tail -f ~/train_<JOBID>.log             # 实时看输出
grep "Learning iteration" ~/train_*.log | tail -5          # 最近几轮

# ---- 文件被删后看输出 ----
squeue -j <JOBID>                        # 先查节点名
ssh <节点名> "tail -f /proc/\$(pgrep -f 'train.py' | head -1)/fd/1"

# ---- 容器内操作 ----
source ~/biped_demo/hpc/env.sh           # 加载环境变量
singularity exec $HOME/biped-sandbox.sif /isaac-sim/python.sh -m pip install -e /workspace/biped_demo/source/biped_demo
singularity exec $HOME/biped-sandbox.sif /isaac-sim/python.sh -c "from isaaclab.app.sim_launcher import add_launcher_args; print('OK')"

# ---- 联网（登录节点） ----
# 你们平台的联网指令
```

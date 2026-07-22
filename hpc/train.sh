#!/bin/bash
#SBATCH -J biped-train
#SBATCH -p gpu                     # ← 确认: 可能是 gpu / a100 / compute，查集群文档
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1               # ← 确认: 可能是 --gpus=a100:1 还是 --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -e

# ---- cache dirs (prevent container write errors) ----
mkdir -p $HOME/.cache/nvidia/GLCache
mkdir -p $HOME/.cache/ov
mkdir -p $HOME/biped_demo/logs

# ---- run training inside Singularity ----
singularity exec --nv \
    --bind $HOME/biped_demo:/workspace/biped_demo \
    --bind $HOME/.cache/nvidia/GLCache:/root/.cache/nvidia/GLCache \
    --bind $HOME/.cache/ov:/root/.local/share/ov \
    --bind $HOME/biped_demo/logs:/workspace/biped_demo/logs \
    /share/home/$USER/biped-train.sif \
    python /workspace/biped_demo/scripts/rsl_rl/train.py \
        --task Biped-velocity-v0 \
        --num_envs 4096 \
        --max_iterations 10000 \
        --headless

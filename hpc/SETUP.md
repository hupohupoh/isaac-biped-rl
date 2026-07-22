# HPC 最终配置步骤

## 前提

- SIF 已上传到 `~/biped-train.sif`（WinSCP 传完）
- 代码已解压到 `~/biped_demo/`（`tar -xzf biped_demo_hpc.tar.gz`）
- 三个最新 Python 文件已覆盖（WinSCP 拖进去）

---

## Step 1: 安装项目依赖

```bash
cd ~/biped_demo
singularity exec --nv --bind $HOME/biped_demo:/workspace/biped_demo \
    $HOME/biped-train.sif \
    pip install -e /workspace/biped_demo/source/biped_demo
```

## Step 2: 测试 10 步

```bash
singularity exec --nv --bind $HOME/biped_demo:/workspace/biped_demo \
    $HOME/biped-train.sif \
    python /workspace/biped_demo/scripts/rsl_rl/train.py \
    --task Biped-velocity-v0 --num_envs 4 --max_iterations 10 --headless
```

## Step 3: 正式提交

```bash
sbatch ~/biped_demo/hpc/train.sh
squeue -u $USER
```

## 查看结果

```bash
ls -lt ~/biped_demo/logs/rsl_rl/biped_demo/ | head -5
tensorboard --logdir ~/biped_demo/logs  # 需端口转发
```

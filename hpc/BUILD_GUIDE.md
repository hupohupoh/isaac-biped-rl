# HPC 容器构建与部署 — 操作手册

> 2026-07-23 | 目标: XJTU HPC (SLURM + Singularity)

---

## 整体架构

```
容器 (biped-sandbox.sif)
  ├── isaac-sim:6.0.0 基础镜像
  └── pip 版 isaaclab (3.0 beta, API 不兼容)

宿主源码 (bind mount 覆盖)
  └── ~/IsaacLab/source/  (isaaclab 10.0.0, 与本地开发环境完全一致)
```

**不重建镜像**，用 `PYTHONPATH` 注入本地源码覆盖容器内的旧版 isaaclab。

---

## 零、构建 SIF 镜像（本地，仅需重建时执行）

### Dockerfile 关键点

`Dockerfile.biped` 位于 `F:\RobotProject\biped_demo\hpc\`，需包含：

```dockerfile
# 权限修复（HPC Singularity 以普通用户运行，非 root）
RUN chmod -R a+rwX /isaac-sim /root /workspace 2>/dev/null || true
```

### 构建流程

```powershell
# 1. 构建 Docker 镜像
cd F:\RobotProject\biped_demo\hpc
docker build -t biped-train:latest -f Dockerfile.biped .

# 2. 导出 tar
docker save biped-train:latest -o F:\RobotProject\biped_demo\hpc\biped-train.tar
```

```bash
# 3. WSL2 内转 SIF（必须拷进 ext4，/mnt/f 的 9p 协议极慢）
cp /mnt/f/RobotProject/biped_demo/hpc/biped-train.tar /home/kiki/
TMPDIR=/home/kiki/apptainer_tmp apptainer build /home/kiki/biped-sandbox.sif docker-archive:///home/kiki/biped-train.tar

# 4. 拷回 F 盘
rsync --bwlimit=50000 /home/kiki/biped-sandbox.sif /mnt/f/RobotProject/biped_demo/hpc/

# 5. 清理 WSL2 空间
rm /home/kiki/biped-train.tar /home/kiki/biped-sandbox.sif
rm -rf /home/kiki/apptainer_tmp /home/kiki/.apptainer
```

---

## 前提条件

- WinSCP 连接: 主机 `10.131.16.13` | 用户 `2242211591` | 协议 SCP
- HPC 有 Singularity (`/bin/singularity`)
- 本地 Isaac Lab 源码位于 `F:\IsaacLab\`
- SIF 镜像已构建 → 上传至 `~/biped-sandbox.sif`

---

## 一、首次部署（仅一次）

### 1. 上传文件

| 本地路径 | HPC 路径 | 方式 |
|----------|----------|------|
| `F:\IsaacLab\` (整个仓库) | `~/IsaacLab/` | WinSCP（约 30MB，秒传） |
| `F:\RobotProject\biped_demo\` | `~/biped_demo/` | WinSCP |
| `F:\RobotProject\biped_demo\hpc\biped-sandbox.sif` | `~/biped-sandbox.sif` | WinSCP（约 13GB，放着过夜） |

### 2. 创建环境脚本

```bash
cat << 'EOF' > ~/biped_demo/hpc/env.sh
#!/bin/bash
# ============================================================
# Isaac Lab 源码注入 — 用宿主机 10.0.0 源码覆盖容器内置版本
# ============================================================

ISAACLAB_PATHS=$(find $HOME/IsaacLab/source -maxdepth 1 -mindepth 1 -type d | tr '\n' ':')

export SINGULARITYENV_PYTHONPATH="${ISAACLAB_PATHS}"
export SINGULARITY_BIND="$HOME/biped_demo:/workspace/biped_demo,$HOME/IsaacLab/source:/opt/isaaclab_source"
EOF
```

### 3. 安装 biped_demo

```bash
source ~/biped_demo/hpc/env.sh
singularity exec $HOME/biped-sandbox.sif \
    /isaac-sim/python.sh -m pip install -e /workspace/biped_demo/source/biped_demo
```

### 4. 验证导入

```bash
source ~/biped_demo/hpc/env.sh
singularity exec $HOME/biped-sandbox.sif \
    /isaac-sim/python.sh -c "
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
import biped_demo.tasks
print('>>> 所有导入成功！')
"
```

---

## 二、测试训练

```bash
source ~/biped_demo/hpc/env.sh

cat << 'EOF' > ~/test_biped.sh
#!/bin/bash
#SBATCH -p gpu4090,gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -J test_biped
#SBATCH -o test_biped_%j.log
#SBATCH -e test_biped_%j.log

source ~/biped_demo/hpc/env.sh
singularity exec --nv $HOME/biped-sandbox.sif \
    /isaac-sim/python.sh /workspace/biped_demo/scripts/rsl_rl/train.py \
    --task Biped-velocity-v0 --num_envs 4 --max_iterations 10 --headless
EOF

sbatch ~/test_biped.sh
```

---

## 三、正式训练（4096 环境）

```bash
source ~/biped_demo/hpc/env.sh

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

### 关键参数说明

| 配置 | 问题 |
|------|------|
| `--env DISPLAY=:0` + `QT_QPA_PLATFORM=offscreen` | Omniverse 后台线程 `XOpenDisplay` → segfault 崩溃 |
| `--bind /dev/shm:/dev/shm` | 4096 环境共消耗大量共享内存，容器默认 64MB 不够 → SIGBUS |
| `--mem=64G` | 4096 个 PhysX 缓冲 + PyTorch 模型 ≈ 40-50G RAM |
| 测试时 `--mem=32G` 即可 | 4 环境测试内存完全够 |

---

## 四、后续改代码

只传改动的 Python 文件（几 KB，秒传）：

```
WinSCP 覆盖:
  scripts/rsl_rl/train.py
  source/biped_demo/biped_demo/tasks/.../biped_env_cfg.py
  source/biped_demo/biped_demo/tasks/.../agents/rsl_rl_ppo_cfg.py
  source/biped_demo/biped_demo/tasks/.../mdp/rewards.py
```

然后 `sbatch` 即可，镜像不用重建。

---

## 五、Import 规范

容器内置的 isaaclab 版本较老，本地脚本需使用**直接路径导入**：

```python
# ✅ 正确（兼容所有版本）
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

# ❌ 错误（仅新版 isaaclab 支持，pip 版无此导出）
# from isaaclab.app import add_launcher_args, launch_simulation
```

已修改的文件：
- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play_rsl_rl.py`
- `scripts/debug_obs.py`
- `scripts/random_agent.py`
- `scripts/zero_agent.py`

---

## 六、踩坑清单

| 坑 | 解法 |
|----|------|
| Permission Denied (pip/缓存写入) | 重建 SIF 时在 Dockerfile 加 `chmod -R a+rwX` |
| `add_launcher_args` 不存在 | pip 版 isaaclab API 不兼容 → 用 PYTHONPATH 注入本地源码 |
| 登录节点无外网 | 联网指令或提交计算节点 |
| sbatch `check account: no response` | SlurmDBD 服务抽风，等恢复后重试 |
| Web 门户登录不了 | 认证服务故障，SSH 不受影响 |
| SIF 只读无法 rm 旧包 | bind mount 源码 + PYTHONPATH 覆盖 |
| 4096 环境 segfault (`XOpenDisplay`) | `--env DISPLAY=:0 --env QT_QPA_PLATFORM=offscreen` |
| 4096 环境 SIGBUS | `--bind /dev/shm:/dev/shm` + `--mem=64G` |

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

## 三、正式训练（v2 配方，4096 环境）

> PPO 参数对齐 Isaac Lab 官方 H1 4096-env 配置。详见 `mdguide/PPO_COLLAPSE_FIX.md`。

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

### 关键参数说明

| 配置 | 问题 |
|------|------|
| `--env DISPLAY=:0` + `QT_QPA_PLATFORM=offscreen` | Omniverse 后台线程 `XOpenDisplay` → segfault 崩溃 |
| `--bind /dev/shm:/dev/shm` | 4096 环境共消耗大量共享内存，容器默认 64MB 不够 → SIGBUS |
| `--mem=64G` | 4096 个 PhysX 缓冲 + PyTorch 模型 ≈ 40-50G RAM |
| `num_steps_per_env=24` | H1 官方 4096-env 值，防止 batch 过大过拟合 |
| `entropy_coef=0.01` / `std_range=(0.05,10.0)` | 防止 sigma 僵化和熵崩塌 |

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
| 4096 环境 PPO 崩溃（entropy 塌 / 震荡） | `num_steps_per_env: 124→24`，对齐 H1 官方 4096-env 配方 |

---

## 七、SIF 权限问题的两种解法

### 弯路（但走通了）：SIF → sandbox → chmod → SIF

问题：最早 SIF 里目录是 Docker 构建时的 root 权限，HPC 上 Singularity 以普通用户运行，`pip install` 等操作 Permission Denied。

```bash
# 1. SIF 解压为 sandbox（一条命令，成功）
singularity build --sandbox $HOME/biped-sandbox $HOME/biped-sandbox.sif

# 2. 修复权限
chmod -R 777 $HOME/biped-sandbox

# 3. 提交 SLURM 作业，借计算节点本地 NVMe SSD 打包回 SIF
sbatch build_sif_ultimate.sh
```

`build_sif_ultimate.sh` 完整内容：

```bash
#!/bin/bash
#SBATCH -p gpu4090,gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16                 # 16 核 CPU，压缩速度提升
#SBATCH --mem=64G             # 64G 内存，防止高并发爆内存
#SBATCH -t 02:00:00           # 2 小时，留足余量
#SBATCH -J build_sif_ultimate
#SBATCH -o build_%j.log
#SBATCH -e build_%j.log

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 1. 检查节点信息与本地 /tmp 磁盘空间 ==="
hostname
df -h /tmp

# 创建计算节点本地 NVMe 上的临时缓存区
NODE_LOCAL_TMP="/tmp/sif_build_$SLURM_JOB_ID"
mkdir -p $NODE_LOCAL_TMP

# 强制 Singularity/Apptainer 临时文件写在本地 SSD，不写回 $HOME
export SINGULARITY_TMPDIR="$NODE_LOCAL_TMP"
export APPTAINER_TMPDIR="$NODE_LOCAL_TMP"

# 无论成功失败，退出时自动清理本地 SSD
trap "rm -rf $NODE_LOCAL_TMP" EXIT

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 2. 开始多线程打包（读取 $HOME → 本地 SSD 压缩）==="

singularity build $HOME/biped-sandbox.sif $HOME/biped-sandbox

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 3. 打包完成！==="
ls -lh $HOME/biped-sandbox.sif
```

`build_sif_ultimate.sh`（16 核 / 64G / 2h，`SINGULARITY_TMPDIR` 指向 `/tmp` 本地 SSD）跑通了。但折腾——额外申请一次作业专门打包，且 sandbox 几万小文件散落 NFS 上不稳定。

### 正道（后续采用）：Dockerfile 里修权限，从源头解决

```dockerfile
RUN chmod -R a+rwX /isaac-sim /root /workspace 2>/dev/null || true
```

重新 `docker build → save → apptainer build → 上传 HPC`，拿到的 SIF 自带正确权限。

### 教训

- **权限问题在 Dockerfile 阶段解决**，不要等 SIF 生成再事后打补丁
- sandbox → SIF **能走通**（靠 SLURM 本地 SSD），但不是首选
- 以后重建 SIF 直接改 Dockerfile 加 `chmod` 行

# HPC 部署指南 — 最终版

> 更新时间: 2026-07-22  
> SIF: 未完成（本地 WSL2 转换 9p 协议太慢，待切到 ext4 原生文件系统）  
> tar: `F:\RobotProject\biped_demo\hpc\biped-train.tar` (13.2GB) ✅ 已构建

---

## 架构

```
本地 Windows                              XJTU HPC (SLURM + Singularity)
┌─────────────────────────┐              ┌──────────────────────────┐
│ Docker Desktop (WSL2)   │    WinSCP    │ login02 / 10.131.16.12  │
│                         │   上传 SIF   │                          │
│ biped-train.tar (13GB)  │ ──────────→  │ biped-train.sif          │
│   → apptainer build     │              │   singularity exec --nv  │
│     biped-train.sif     │              │     python train.py       │
│                         │              │                          │
│ biped_demo (代码, 12MB)  │  rsync/scp   │ /share/home/2242211591/  │
│                         │ ──────────→  │   biped_demo/            │
└─────────────────────────┘              └──────────────────────────┘
```

## 关键教训

| 坑 | 原因 | 解法 |
|----|------|------|
| WSL2 `/mnt/f` 转 SIF 极慢 | 9p 协议跨文件系统 IO | **tar 先 cp 进 /tmp/ (ext4) 再转** |
| C 盘爆满 | Docker 默认存 C 盘 | Docker Desktop 设置里 Disk image → F:\DockerData |
| pip install 下载 40KB/s | nvidia PyPI 无国内 CDN | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| VPN scp 112KB/s | VPN 隧道限速 | 用 WinSCP 传，或网页上传 |
| NCCL 登录节点断网 | HPC 安全策略 | 容器内下载，登录节点只转 SIF |
| `isaaclab-tasks` pip 找不到 | 子包不单独发布 | 只装 `isaaclab` 元包 |

---

## 当前进度

### ✅ 已完成

1. Docker 镜像构建完成 → `biped-train.tar` (13.2GB)
2. HPC 环境确认：SLURM + Singularity (`/bin/singularity`)
3. 项目代码就绪：`biped_demo/` 完整
4. 本地训练验证通过（7968 步，entropy 5.8，success_rate 89%）

### ⏳ 待完成

1. **转 SIF**（在 WSL2 原生 ext4 文件系统转换，避免 9p 瓶颈）
2. WinSCP 上传 SIF 到 HPC
3. 上传 `biped_demo` 代码到 HPC
4. HPC 上 `pip install -e source/biped_demo`
5. 提交 SLURM 作业测试

---

## 继续操作步骤

### Step 1: 本地转 SIF（快路径）

```bash
# WSL2 终端里
# 核心: 先拷进 /tmp (ext4 原生文件系统)，再转——避过 /mnt/f 的 9p 瓶颈

cp /mnt/f/RobotProject/biped_demo/hpc/biped-train.tar /tmp/
TMPDIR=/tmp apptainer build /tmp/biped-train.sif docker-archive:///tmp/biped-train.tar
cp /tmp/biped-train.sif /mnt/f/RobotProject/biped_demo/hpc/
```

### Step 2: WinSCP 上传 SIF

1. 下载 WinSCP: `https://winscp.net/eng/download.php`
2. 连接：主机 `10.131.16.13`、用户 `2242211591`、端口 `22`、协议 SCP
3. 拖 `biped-train.sif` 到 HPC `~/`
4. 同步项目代码：拖 `F:\RobotProject\biped_demo\` 到 HPC `~/biped_demo/`

### Step 3: HPC 网页终端配置

```bash
# 安装项目
cd ~/biped_demo
pip install -e source/biped_demo 2>&1 | tail -5

# 测试 10 步（确认容器能跑）
singularity exec --nv \
    --bind $HOME/biped_demo:/workspace/biped_demo \
    $HOME/biped-train.sif \
    python /workspace/biped_demo/scripts/rsl_rl/train.py \
    --task Biped-velocity-v0 --num_envs 4 --max_iterations 10 --headless
```

### Step 4: 正式训练

```bash
cd ~/biped_demo
sbatch hpc/train.sh
squeue -u $USER
```

---

## 文件清单

```
F:\RobotProject\biped_demo\hpc\
├── README.md              # 本文件
├── Dockerfile.biped       # 容器构建文件
├── build.sh               # 完整构建脚本（Docker build → tar → sif）
├── train.sh               # SLURM 作业提交脚本
└── sync.sh                # 代码增量同步
```

---

## 参考

- XJTU HPC 登录节点: `10.131.16.12` (login02), `10.131.16.13`
- 用户目录: `/share/home/2242211591`
- Singularity: `/bin/singularity`
- GPU 分区: `gpu`（待确认，可能 `a100` 或 `compute`）
- 管理员: hpc@xjtu.edu.cn, QQ 群 207668091

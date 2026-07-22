# HPC 容器构建与部署 — 操作手册

> 2026-07-22 | 目标: XJTU HPC (SLURM + Singularity)

---

## 前提条件

- Docker Desktop 已安装，Disk image 设在 **F 盘**
- WSL2 Ubuntu 已安装，VHDX 在 **F 盘**（不在 C 盘）
- NGC 已登录: `docker login nvcr.io` (用户 `$oauthtoken`, 密码 API Key)
- HPC 有 Singularity (`/bin/singularity`)

---

## 一、Dockerfile（纯环境，不含项目代码）

```dockerfile
# F:\RobotProject\biped_demo\hpc\Dockerfile.biped

FROM nvcr.io/nvidia/isaac-sim:6.0.0

ENV ISAACSIM_ROOT_PATH=/isaac-sim
ENV HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# carb env shim（防止 Omniverse 多线程 glibc 崩溃）
RUN src=$(find ${ISAACSIM_ROOT_PATH} -name libcarb.env.shim.so -type f 2>/dev/null | head -n1) && \
    if [ -n "${src}" ] && [ -f "${src}" ]; then \
        install -m 0755 "${src}" /usr/local/lib/libcarb.env.shim.so && \
        echo "/usr/local/lib/libcarb.env.shim.so" > /etc/ld.so.preload; \
    fi

# Isaac Lab（清华镜像加速国内下载）
RUN ${ISAACSIM_ROOT_PATH}/python.sh -m pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.nvidia.com \
    isaaclab rsl-rl-lib==5.4.1 \
    && rm -rf /root/.cache/pip/*

# Singularity 兼容目录
RUN mkdir -p \
    ${ISAACSIM_ROOT_PATH}/kit/cache \
    /root/.cache/ov /root/.cache/pip \
    /root/.cache/nvidia/GLCache /root/.nv/ComputeCache \
    /root/.nvidia-omniverse/logs /root/.local/share/ov/data \
    /root/Documents \
    /var/run/nvidia-persistenced \
    && touch /bin/nvidia-smi /bin/nvidia-debugdump /bin/nvidia-persistenced \
             /bin/nvidia-cuda-mps-control /bin/nvidia-cuda-mps-server /etc/localtime

WORKDIR /workspace/biped_demo
```

---

## 二、构建镜像

```powershell
cd F:\RobotProject\biped_demo\hpc
docker build -t biped-train:latest -f Dockerfile.biped .
```

导出 tar:

```powershell
docker save biped-train:latest -o F:\RobotProject\biped_demo\hpc\biped-train.tar
```

---

## 三、tar → SIF 转换

**关键**: 不能直接在 `/mnt/f` 下操作（9p 协议极慢，几小时都跑不完）。

**必须把 tar 拷进 WSL2 原生 ext4 再转：**

```bash
# WSL2 里
cp /mnt/f/RobotProject/biped_demo/hpc/biped-train.tar /home/kiki/
TMPDIR=/home/kiki/apptainer_tmp apptainer build /home/kiki/biped-train.sif docker-archive:///home/kiki/biped-train.tar
```

转完后拷回 F 盘：

```bash
cp /home/kiki/biped-train.sif /mnt/f/RobotProject/biped_demo/hpc/

# 清理 WSL2 内部空间
rm /home/kiki/biped-train.tar
rm /home/kiki/biped-train.sif
rm -rf /home/kiki/apptainer_tmp
rm -rf /home/kiki/.apptainer
```

如果 cp 到 F 盘报 `Cannot allocate memory`，用 PowerShell 替代：

```powershell
Copy-Item "\\wsl$\Ubuntu\home\kiki\biped-train.sif" "F:\RobotProject\biped_demo\hpc\"
```
或者用 rsync 分块传：
实际采用的这个。
rsync --bwlimit=50000 /home/kiki/biped-train.sif /mnt/f/RobotProject/biped_demo/hpc/
---

## 四、上传 HPC

**用 WinSCP**（别用 scp——VPN 限速 112KB/s 传不动 13GB）。

1. 下载 WinSCP: `https://winscp.net/eng/download.php`
2. 连接: 主机 `10.131.16.13` | 用户 `2242211591` | 协议 SCP
3. 上传 `biped-train.sif` → HPC `~/`
4. 上传 `biped_demo` 项目代码 → HPC `~/biped_demo/`

---

## 五、HPC 初始化（仅一次）

```bash
# 解压代码
cd ~ && tar -xzf biped_demo_hpc.tar.gz

# 容器内安装项目依赖
singularity exec --nv --bind $HOME/biped_demo:/workspace/biped_demo \
    $HOME/biped-train.sif \
    pip install -e /workspace/biped_demo/source/biped_demo

# 测试 10 步
singularity exec --nv --bind $HOME/biped_demo:/workspace/biped_demo \
    $HOME/biped-train.sif \
    python /workspace/biped_demo/scripts/rsl_rl/train.py \
    --task Biped-velocity-v0 --num_envs 4 --max_iterations 10 --headless
```

---

## 六、正式训练

```bash
cd ~/biped_demo
sbatch hpc/train.sh
squeue -u $USER
```

---

## 后续改代码

只传 Python 文件（几 KB，秒传），镜像不用重建：

```
WinSCP 覆盖:
  biped_env_cfg.py
  agents/rsl_rl_ppo_cfg.py
  mdp/rewards.py + __init__.pyi
```

然后 `sbatch hpc/train.sh` 即可。

---

## 踩坑清单

| 坑 | 解法 |
|----|------|
| WSL2 `/mnt/f` 9p 太慢 | 拷进 ext4 再操作 |
| `/tmp` 只有 4GB | `TMPDIR=/home/kiki/apptainer_tmp` |
| C 盘被 Docker/WSL 占满 | 移到 F 盘 |
| NGC 登录报 unauthorized | 用户名填 `$oauthtoken` |
| pip 下载 40KB/s | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 登录节点断网 | 镜像本地构建，上传 SIF |
| WinSCP 1MB/s | 放着过夜，3 小时传完 |
| SIF 构建崩溃 (SIGBUS) | VHDX 空间不足，清缓存重来 |

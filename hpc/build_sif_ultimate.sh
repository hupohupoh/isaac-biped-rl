#!/bin/bash
#SBATCH -p gpu4090,gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16                 # 提高到 16 核 CPU，压缩速度直接提升 2~3 倍！
#SBATCH --mem=64G             # 给足 64G 内存，防止高并发解压缩爆内存
#SBATCH -t 02:00:00           # 直接拉满到 2 小时！宁可让它早完工，绝不给系统强杀的机会
#SBATCH -J build_sif_ultimate
#SBATCH -o build_%j.log
#SBATCH -e build_%j.log

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 1. 检查节点信息与本地 /tmp 磁盘空间 ==="
hostname
df -h /tmp

# 2. 创建计算节点本地 NVMe 上的临时缓存区
NODE_LOCAL_TMP="/tmp/sif_build_$SLURM_JOB_ID"
mkdir -p $NODE_LOCAL_TMP

# 3. 双重保险：同时设置 Singularity 和 Apptainer 的环境变量，强制所有写操作在本地 SSD 进行
export SINGULARITY_TMPDIR="$NODE_LOCAL_TMP"
export APPTAINER_TMPDIR="$NODE_LOCAL_TMP"

# 4. 自动清理：无论成功失败，退出时自动擦除本地 SSD 临时垃圾
trap "rm -rf $NODE_LOCAL_TMP" EXIT

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 2. 开始多线程极速打包（读取共享盘 -> 本地 SSD 极速压缩）==="

# 5. 执行打包命令
singularity build $HOME/biped-sandbox.sif $HOME/biped-sandbox

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 3. 打包任务完成！检查生成的镜像文件： ==="
ls -lh $HOME/biped-sandbox.sif

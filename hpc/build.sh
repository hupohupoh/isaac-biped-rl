#!/bin/bash
# build.sh — run on YOUR LOCAL WSL2 / Linux machine (not HPC)
# Converts the Docker tar into a Singularity SIF image.
#
# Prerequisites:
#   sudo apt install singularity-container   # or apptainer
#   docker build -t biped-train:latest -f Dockerfile.biped .
#   docker save biped-train:latest -o biped-train.tar

set -e

echo "[1/2] Building Docker image..."
docker build -t biped-train:latest -f Dockerfile.biped .

echo "[2/2] Exporting tar → SIF..."
docker save biped-train:latest -o biped-train.tar

echo "Converting tar → sif (this may take several minutes)..."
singularity build biped-train.sif docker-archive://biped-train.tar

echo "Done. Upload biped-train.sif and train.sh to HPC."
echo "  scp biped-train.sif your_id@hpc.xjtu.edu.cn:~/"
echo "  scp train.sh your_id@hpc.xjtu.edu.cn:~/biped_demo/"

# sync.sh — run from LOCAL PowerShell to upload code changes to HPC
# Usage:  .\sync.ps1  (PowerShell)  or  bash sync.sh  (Git Bash)

# ---- CONFIG ----
HPC_USER="2242211591"
HPC_HOST="10.131.16.12"   # or the hostname that works with SSH
REMOTE_DIR="/share/home/${HPC_USER}/biped_demo"

# ---- exclude patterns ----
EXCLUDES="--exclude=__pycache__ --exclude=*.pyc --exclude=logs --exclude=biped_clean --exclude=hpc/Dockerfile.biped"

# ---- sync ----
rsync -avz --progress ${EXCLUDES} \
    F:/RobotProject/biped_demo/ \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/

echo ""
echo "Sync complete. On HPC, run:"
echo "  cd ${REMOTE_DIR} && sbatch hpc/train.sh"

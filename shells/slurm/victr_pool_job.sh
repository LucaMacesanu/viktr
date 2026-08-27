#!/bin/bash
# ============================================================================
# SLURM job script — precompute the 4-task retrieval-pool ChunkDictionaries
# (scripts/build_retrieval_pool.py) + the 85/10/5 splits. Run this once,
# before any of the 3 victr_train_job.sh arms.
#
#   sbatch --account=<account> --gres=gpu:h200:1 --time=01:00:00 \
#          shells/slurm/victr_pool_job.sh <repo_dir> <out_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/victr-pool-%j.out
#SBATCH --error=logs/victr-pool-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_pool_job.sh <repo_dir> <out_dir>"}
OUT_DIR=${2:?"Usage: sbatch ... victr_pool_job.sh <repo_dir> <out_dir>"}

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "============================================================"

module purge
set +u
[ -f ~/.bashrc ] && source ~/.bashrc
set -u

cd "$REPO_DIR"
"$REPO_DIR/.venv/bin/python3" scripts/build_retrieval_pool.py --out-dir "$OUT_DIR" --seed 42

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

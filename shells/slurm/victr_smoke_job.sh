#!/bin/bash
# ============================================================================
# SLURM job script — 10-step local smoke test for one VictrPolicy retrieval
# arm (plan Phase 4 verification, before submitting the full 15k-step training
# jobs via victr_train_job.sh). Confirms loss is finite, no shape errors, and
# (for vision+value) that the fusion MLP actually gets a gradient.
#
# DO NOT USE FOR REAL TRAINING RUNS -- this and victr_train_job.sh drive
# scripts/train_victr.py, which trains pi05-based policies via lerobot's native
# PyTorch stack. Per project decision (2026-08-25), openpi is the sole training
# pipeline for pi05-based policies; see train_victr.py's module docstring.
#
#   sbatch --account=<account> --gres=gpu:h200:1 --time=00:20:00 \
#          shells/slurm/victr_smoke_job.sh <metric> <repo_dir> <pool_dir> <output_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/victr-smoke-%j.out
#SBATCH --error=logs/victr-smoke-%j.err

set -euo pipefail

RETRIEVAL_METRIC=${1:?"Usage: sbatch ... victr_smoke_job.sh <metric> <repo_dir> <pool_dir> <output_dir>"}
REPO_DIR=${2:?"Usage: sbatch ... victr_smoke_job.sh <metric> <repo_dir> <pool_dir> <output_dir>"}
POOL_DIR=${3:?"Usage: sbatch ... victr_smoke_job.sh <metric> <repo_dir> <pool_dir> <output_dir>"}
OUTPUT_DIR=${4:?"Usage: sbatch ... victr_smoke_job.sh <metric> <repo_dir> <pool_dir> <output_dir>"}

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Retrieval metric (smoke): $RETRIEVAL_METRIC"
echo "Started:   $(date)"
echo "============================================================"

module purge
set +u
[ -f ~/.bashrc ] && source ~/.bashrc
set -u

cd "$REPO_DIR"

"$REPO_DIR/.venv/bin/python3" scripts/train_victr.py \
    --retrieval-metric "$RETRIEVAL_METRIC" \
    --pool-dir "$POOL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --steps 10 \
    --batch-size 4 \
    --num-workers 2 \
    --log-freq 1 \
    --save-freq 10

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

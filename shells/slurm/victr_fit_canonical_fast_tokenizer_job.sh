#!/bin/bash
# ============================================================================
# Fits scripts/fit_canonical_fast_tokenizer.py's FAST tokenizer on the ICRA
# canonical action encoding (notes/ICRA_plan.md Sec 1), scoped to the trimmed
# 20-task/1,186-episode canonical extended set. CPU-only, single process --
# action column only via pandas parquet reads (no LeRobotDataset object, no
# video decode). Runs in openpi's venv (needs openpi.policies.yor_rotation and
# lerobot.scripts.lerobot_train_tokenizer, both importable there).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_fit_canonical_fast_tokenizer_job.sh <repo_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-fit-canonical-fast-tokenizer-%j.out
#SBATCH --error=logs/victr-fit-canonical-fast-tokenizer-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_fit_canonical_fast_tokenizer_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/third_party/openpi/.venv/bin/python3" scripts/fit_canonical_fast_tokenizer.py \
    --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================================
# SLURM job script -- computes Reward-Aligned Behavior Cloning (RA-BC,
# notes/reward_aligned_bc.md) per-frame loss weights via
# scripts/precompute_rabc_weights.py. CPU-only (reads meta/episodes parquet +
# meta/value_estimates JSON, no video decode, no model forward pass) -- required
# once per (episodes_path, delta, phi_source) combination before the matching
# TrainConfig's rabc_weights_dir can be used (e.g. yor_icl_pi05_rabc).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_rabc_weights_job.sh \
#       <repo_dir> <episodes_path> <out_dir> [extra scripts/precompute_rabc_weights.py args...]
#
# Example (matches yor_icl_pi05_rabc's TrainConfig):
#   sbatch --account=<account> shells/slurm/victr_precompute_rabc_weights_job.sh \
#       /scratch/lim2045/icl_ws/viktr \
#       third_party/openpi/assets/yor_icl_pi05_easy_pnp_v2_episodes.json \
#       third_party/openpi/assets/rabc_weights/yor_icl_pi05_rabc
# ============================================================================
#SBATCH --job-name=victr-precompute-rabc-weights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-rabc-weights-%j.out
#SBATCH --error=logs/victr-precompute-rabc-weights-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_rabc_weights_job.sh <repo_dir> <episodes_path> <out_dir> [extra args...]"}
EPISODES_PATH=${2:?"Usage: sbatch ... victr_precompute_rabc_weights_job.sh <repo_dir> <episodes_path> <out_dir> [extra args...]"}
OUT_DIR=${3:?"Usage: sbatch ... victr_precompute_rabc_weights_job.sh <repo_dir> <episodes_path> <out_dir> [extra args...]"}
shift 3

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Episodes:  $EPISODES_PATH"
echo "Out dir:   $OUT_DIR"
echo "Started:   $(date)"
echo "============================================================"

cd "$REPO_DIR"

uv run python scripts/precompute_rabc_weights.py \
    --episodes-path "$EPISODES_PATH" \
    --out-dir "$OUT_DIR" \
    "$@"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

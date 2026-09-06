#!/bin/bash
# ============================================================================
# Vision-metric-only precompute of per-frame VICTR retrieval context for the
# 90/10-split expanded task set (see victr_pool_90_10_job.sh). Mirrors
# victr_precompute_expanded_vision_job.sh exactly (same NSHARDS=8/4-threads
# fix for DINOv2's compute-bound CPU forward pass), just pointed at the new
# pool/out dirs.
#
# Requires victr_pool_90_10_job.sh to have completed first (pool built,
# converted, and resized to 224x224).
#
# --all-episodes-json intentionally uses all_episodes.json (train+pool
# combined -- no test split in this 90/10 pool), not train_episodes.json:
# precomputing context for the pool's own episodes too costs little extra and
# matches the existing 85/10/5 pipeline's convention, keeping this reusable if
# a future arm wants broader query coverage. Training itself only ever looks
# up train_episodes.json's episode indices.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_90_10_vision_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-90-10-vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-90-10-vision-%j.out
#SBATCH --error=logs/victr-precompute-90-10-vision-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_90_10_vision_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="outputs/victr/icl_pool_expanded_90_10"
OPENPI_POOL_DIR_224="third_party/openpi/assets/victr_icl_pool_expanded_90_10_224"
OUT_DIR="outputs/victr/retrieval_context_expanded_90_10"
NSHARDS=8

test -f "$POOL_DIR/all_episodes.json" || { echo "missing $POOL_DIR/all_episodes.json -- run victr_pool_90_10_job.sh first" >&2; exit 1; }
test -d "$OPENPI_POOL_DIR_224" || { echo "missing $OPENPI_POOL_DIR_224 -- run victr_pool_90_10_job.sh first" >&2; exit 1; }

echo "--- metric=vision: launching $NSHARDS parallel shards, 4 threads each (openpi venv) ---"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir "$OPENPI_POOL_DIR_224" \
        --all-episodes-json "$POOL_DIR/all_episodes.json" \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --out-dir "$OUT_DIR" \
        --metric vision \
        --tasks all \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-90-10-vision-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=vision: one or more shards failed -- check logs/precompute-90-10-vision-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=vision: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

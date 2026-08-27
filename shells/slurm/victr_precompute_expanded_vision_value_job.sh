#!/bin/bash
# ============================================================================
# vision_value (fixed-weight fusion, see "Add a vision+value fusion VICTR arm"
# plan) precompute of per-frame VICTR retrieval context for the expanded task
# set (31 tasks, 1,784 episodes). Mirrors victr_precompute_expanded_vision_job.sh
# exactly (same NSHARDS=8/4-threads-each split, same time budget): vision_value
# still needs the same DINOv2 embedding pass per frame as the pure vision metric
# (it's the dominant cost, not the extra RoboDopamine value lookup, which is a
# cheap disk-backed O(1) read) -- so this job is priced the same as the vision
# job, not cheaper, despite reusing already-cached pool key_embeddings/
# key_values for the pool side of retrieval.
#
# Assumes the expanded-task pools already exist AND have been resized to 224x224
# (outputs/victr/icl_pool_expanded, third_party/openpi/assets/
# victr_icl_pool_expanded_224). Resumes cleanly: skips episodes whose output
# already exists (same windowed/resumable precompute_episode as vision/value).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_expanded_vision_value_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-expanded-vision-value
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-expanded-vision-value-%j.out
#SBATCH --error=logs/victr-precompute-expanded-vision-value-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_vision_value_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="outputs/victr/icl_pool_expanded"
OPENPI_POOL_DIR_224="third_party/openpi/assets/victr_icl_pool_expanded_224"
OUT_DIR="outputs/victr/retrieval_context_expanded"
NSHARDS=8

test -f "$POOL_DIR/all_episodes.json" || { echo "missing $POOL_DIR/all_episodes.json -- run victr_precompute_expanded_job.sh's pool-build step first" >&2; exit 1; }
test -d "$OPENPI_POOL_DIR_224" || { echo "missing $OPENPI_POOL_DIR_224 -- run shells/slurm/victr_resize_pool_job.sh first" >&2; exit 1; }

echo "--- metric=vision_value: launching $NSHARDS parallel shards, 4 threads each (openpi venv) ---"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir "$OPENPI_POOL_DIR_224" \
        --all-episodes-json "$POOL_DIR/all_episodes.json" \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --out-dir "$OUT_DIR" \
        --metric vision_value \
        --tasks all \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-expanded-vision-value-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=vision_value: one or more shards failed -- check logs/precompute-expanded-vision-value-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=vision_value: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

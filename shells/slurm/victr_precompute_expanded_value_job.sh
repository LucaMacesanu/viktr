#!/bin/bash
# ============================================================================
# Value-metric-only precompute of per-frame VICTR retrieval context for the
# expanded task set (31 of 36 tasks -- viktr.data.icl_dataset.expanded_tasks(),
# 1,784 episodes). Split out from victr_precompute_expanded_job.sh (which does
# both metrics) because the value metric never calls yor_retrieval.embed_frames
# (uses the lru_cache'd robodopamine_value_at instead) -- it isn't affected by
# that function's unbatched-DINOv2-forward-pass OOM bug (see
# yor_retrieval.py::embed_frames' docstring / precompute_retrieval_context.py's
# module docstring for the bug), so it can run at full scale right now, in
# parallel with fixing that bug for the vision metric.
#
# Assumes the expanded-task pools already exist AND have been resized to 224x224
# (outputs/victr/icl_pool_expanded, third_party/openpi/assets/
# victr_icl_pool_expanded_224 -- built by victr_precompute_expanded_job.sh's
# steps 1-2b, or shells/slurm/victr_resize_pool_job.sh directly). Resumes
# cleanly: skips episodes whose output already exists.
#
# NSHARDS=16, task-based sharding (precompute_retrieval_context.py's
# _assign_tasks_to_shards): the value metric's first full-scale attempt at
# NSHARDS=32 with un-resized, episode-interleaved sharding still OOM'd 32/32
# shards (job 16350893, MaxRSS 125.8G) -- see precompute_retrieval_context.py's
# module docstring for the real root cause (huge un-resized pools + many shards
# redundantly loading copies of the same one).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_expanded_value_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-expanded-value
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-expanded-value-%j.out
#SBATCH --error=logs/victr-precompute-expanded-value-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_value_job.sh <repo_dir>"}
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
NSHARDS=16

test -f "$POOL_DIR/all_episodes.json" || { echo "missing $POOL_DIR/all_episodes.json -- run victr_precompute_expanded_job.sh's pool-build step first" >&2; exit 1; }
test -d "$OPENPI_POOL_DIR_224" || { echo "missing $OPENPI_POOL_DIR_224 -- run shells/slurm/victr_resize_pool_job.sh first" >&2; exit 1; }

echo "--- metric=value: launching $NSHARDS parallel shards (openpi venv) ---"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir "$OPENPI_POOL_DIR_224" \
        --all-episodes-json "$POOL_DIR/all_episodes.json" \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --out-dir "$OUT_DIR" \
        --metric value \
        --tasks all \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-expanded-value-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=value: one or more shards failed -- check logs/precompute-expanded-value-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=value: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

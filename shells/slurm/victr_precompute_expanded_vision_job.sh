#!/bin/bash
# ============================================================================
# Vision-metric-only precompute of per-frame VICTR retrieval context for the
# expanded task set (31 of 36 tasks -- viktr.data.icl_dataset.expanded_tasks(),
# 1,784 episodes). Mirrors victr_precompute_expanded_value_job.sh -- split out
# from victr_precompute_expanded_job.sh (which builds pools from scratch) since
# the pools + 224x224 resize already exist by the time this runs.
#
# Requires both fixes now in place:
#   1. openpi/policies/yor_retrieval.py::embed_frames batches its DINOv2 forward
#      pass (batch_size=256) instead of running one unbatched pass over an
#      entire episode's frames at once (up to ~5,300 frames -- the original OOM
#      cause for this metric specifically).
#   2. Resized pools (see victr_resize_pool_job.sh) + task-based sharding (see
#      precompute_retrieval_context.py's _assign_tasks_to_shards) -- the deeper
#      root cause shared with the value metric (see that job's comment).
#
# Assumes the expanded-task pools already exist AND have been resized to 224x224
# (outputs/victr/icl_pool_expanded, third_party/openpi/assets/
# victr_icl_pool_expanded_224 -- built by victr_precompute_expanded_job.sh's
# steps 1-2b, or shells/slurm/victr_resize_pool_job.sh directly). Resumes
# cleanly: skips episodes whose output already exists.
#
# NSHARDS=8 with 4 OMP/MKL threads each (32 CPUs total, cpu_short's per-user
# cap), NOT the value job's NSHARDS=16/1-thread-each split: DINOv2's CPU
# forward pass is genuinely compute-bound and benefits a lot from
# multithreading (measured 345.6ms/frame at 1 thread vs. 102.3ms/frame at 4
# threads, 256-frame batch) -- a first attempt at this job single-threaded (the
# value metric's convention, harmless there since it never touches DINOv2)
# didn't finish embedding even ONE ~4,759-frame episode inside a 30-minute
# pilot. At the measured 4-thread rate, the full ~1.02M-frame expanded set is
# roughly 1.02M * 0.1023s / 8 shards =~ 3.6h -- within cpu_short's 6h cap, but
# closer to it than the value job, so this is sized at the full 32-CPU budget
# rather than leaving headroom to run concurrently with the value job.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_expanded_vision_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-expanded-vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-expanded-vision-%j.out
#SBATCH --error=logs/victr-precompute-expanded-vision-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_vision_job.sh <repo_dir>"}
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
        > "logs/precompute-expanded-vision-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=vision: one or more shards failed -- check logs/precompute-expanded-vision-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=vision: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

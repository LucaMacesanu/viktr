#!/bin/bash
# ============================================================================
# Single-job pipeline: builds the expanded-task-set (31 of 36 tasks --
# viktr.data.icl_dataset.expanded_tasks(), everything except gears/notebook/
# grocery-bag/octagon) retrieval pools, converts them for openpi, resizes chunk
# images to 224x224 (see scripts/resize_pool_images.py -- pools store native
# 480x640 images but only 224x224 is ever consumed), then precomputes per-frame
# VICTR retrieval context (both vision and value metrics) for every episode in
# that set.
#
# Runs entirely on cpu_short -- no GPU involved. Precompute itself is
# embarrassingly parallel across TASKS (not raw episodes -- each shard is
# assigned whole tasks, bin-packed by pool size via precompute_retrieval_
# context.py's _assign_tasks_to_shards, so a task's pool is only ever loaded by
# one shard process, never redundantly copied into many shards' memory at
# once). NSHARDS=16 (not 32): even after the 224x224 resize, the un-resized
# scheme's per-shard interleaving let many shards independently load a copy of
# the SAME huge task's pool at once, which -- combined with the raw 480x640
# storage -- OOM'd every full-scale attempt at this step (jobs 16289472,
# 16290358, 16292793, 16301528, 16301649, 16309502, 16350893).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_expanded_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-expanded
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-expanded-%j.out
#SBATCH --error=logs/victr-precompute-expanded-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="outputs/victr/icl_pool_expanded"
OPENPI_POOL_DIR="third_party/openpi/assets/victr_icl_pool_expanded"
OPENPI_POOL_DIR_224="third_party/openpi/assets/victr_icl_pool_expanded_224"
OUT_DIR="outputs/victr/retrieval_context_expanded"
NSHARDS=16

echo "--- step 1: build expanded-task pools (viktr venv) ---"
uv run python scripts/build_retrieval_pool.py \
    --tasks expanded \
    --out-dir "$POOL_DIR" \
    --seed 42

echo "--- step 2: convert pools for openpi (viktr venv) ---"
uv run python scripts/convert_pool_for_openpi.py \
    --tasks expanded \
    --pool-dir "$POOL_DIR" \
    --out-dir "$OPENPI_POOL_DIR"

echo "--- step 2b: resize pools to 224x224 (openpi venv) ---"
third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/resize_pool_images.py \
    --pool-dir "$OPENPI_POOL_DIR" \
    --out-dir "$OPENPI_POOL_DIR_224" \
    --primary-camera observation.images.zed

echo "--- step 3: precompute retrieval context, both metrics (openpi venv) ---"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for METRIC in vision value; do
    echo "--- metric=$METRIC: launching $NSHARDS parallel shards ---"
    pids=()
    for ((i = 0; i < NSHARDS; i++)); do
        third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
            --pool-dir "$OPENPI_POOL_DIR_224" \
            --all-episodes-json "$POOL_DIR/all_episodes.json" \
            --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
            --out-dir "$OUT_DIR" \
            --metric "$METRIC" \
            --tasks all \
            --num-shards "$NSHARDS" --shard-index "$i" \
            > "logs/precompute-expanded-${SLURM_JOB_ID}-${METRIC}-shard${i}.out" 2>&1 &
        pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "metric=$METRIC: one or more shards failed -- check logs/precompute-expanded-${SLURM_JOB_ID}-${METRIC}-shard*.out" >&2
        exit 1
    fi
    echo "--- metric=$METRIC: all $NSHARDS shards completed ---"
done

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

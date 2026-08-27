#!/bin/bash
# ============================================================================
# Pilot for shells/slurm/victr_precompute_expanded_job.sh's step 3 at reduced
# concurrency (8 shards instead of 32). The 32-shard run (job 16301649) was
# cancelled after its batch step hit OUT_OF_MEMORY (MaxRSS=125.8GB against a
# 120G cap) and was running ~49x slower per-worker than the single-process
# pilot rate -- almost certainly memory-pressure/thrashing from 32 concurrent
# DINOv2-CPU-inference processes, not just CPU contention.
#
# Reuses the already-built pools from that run (steps 1-2 are skipped here --
# this only re-times step 3 at num_shards=8). Runs a small --limit'd slice of
# each metric so we get real per-episode timing lines (stdout flushes at
# process exit) plus sstat-visible peak memory, to validate before resubmitting
# the full 1,784-episode run at 8 shards.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_expanded_pilot8_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-pilot8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-pilot8-%j.out
#SBATCH --error=logs/victr-precompute-pilot8-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_pilot8_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="outputs/victr/icl_pool_expanded"
OPENPI_POOL_DIR="third_party/openpi/assets/victr_icl_pool_expanded"
OUT_DIR="outputs/victr/retrieval_context_pilot8"
NSHARDS=8
LIMIT=4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for METRIC in vision value; do
    echo "--- metric=$METRIC: launching $NSHARDS parallel shards (limit=$LIMIT each) ---"
    pids=()
    for ((i = 0; i < NSHARDS; i++)); do
        third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
            --pool-dir "$OPENPI_POOL_DIR" \
            --all-episodes-json "$POOL_DIR/all_episodes.json" \
            --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
            --out-dir "$OUT_DIR" \
            --metric "$METRIC" \
            --tasks all \
            --num-shards "$NSHARDS" --shard-index "$i" --limit "$LIMIT" \
            > "logs/precompute-pilot8-${SLURM_JOB_ID}-${METRIC}-shard${i}.out" 2>&1 &
        pids+=($!)
    done
    # Sample RSS while shards run, to see peak memory at this concurrency.
    ( for _ in $(seq 1 60); do sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS -n 2>/dev/null; sleep 5; done ) > "logs/precompute-pilot8-${SLURM_JOB_ID}-${METRIC}-rss.log" &
    rss_pid=$!
    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    kill "$rss_pid" 2>/dev/null || true
    if [ "$fail" -ne 0 ]; then
        echo "metric=$METRIC: one or more pilot shards failed -- check logs/precompute-pilot8-${SLURM_JOB_ID}-${METRIC}-shard*.out" >&2
        exit 1
    fi
    echo "--- metric=$METRIC: pilot shards completed ---"
done

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

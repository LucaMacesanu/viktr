#!/bin/bash
# ============================================================================
# One-off SLURM-accounted pilot: value-metric precompute on the expanded set's
# longest-episode tasks ("clean the plate", "sort the items into their
# containers" -- up to ~5,300 frames/episode) after fixing
# precompute_retrieval_context.py's precompute_episode() to skip loading the
# episode's raw frames entirely for the value metric (it never used them --
# that unconditional load was the real cause of job 16350893's 32/32-shard
# OOM on a value-only run). A quick foreground run of the same thing directly
# on the login node was silently killed with no traceback (no cgroup memory
# accounting there -- same failure mode already seen with
# fit_expanded_fast_tokenizer.py) -- this exists to get a real, accounted
# answer before trusting the fix at full 32-shard scale.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_value_longtest_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-value-longtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-value-longtest-%j.out
#SBATCH --error=logs/victr-precompute-value-longtest-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_value_longtest_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

( while true; do
    sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS,AveRSS -P 2>/dev/null | tail -1
    sleep 10
  done ) > "logs/precompute-value-longtest-${SLURM_JOB_ID}-rss.log" &
RSS_PID=$!

third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
    --pool-dir third_party/openpi/assets/victr_icl_pool_expanded_224 \
    --all-episodes-json outputs/victr/icl_pool_expanded/all_episodes.json \
    --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
    --out-dir scratch_tmp/precompute_value_pilot \
    --metric value \
    --tasks "clean the plate,sort the items into their containers" \
    --num-shards 1 --shard-index 0 --limit 10

kill "$RSS_PID" 2>/dev/null || true

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

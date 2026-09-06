#!/bin/bash
# ============================================================================
# Finishes the 90/10-split vision-metric retrieval-context precompute for the
# one straggler task victr_precompute_90_10_vision_job.sh's 8-way shard split
# can't fit in its 5h45m budget: "sort the items into their containers" (50
# episodes, ~4-5k frames each, ~0.17s/frame -> ~13min/episode, ~11h total for
# that task alone). Mirrors victr_precompute_expanded_vision_sort_task_job.sh
# exactly (same straggler, same fix, for the 85/10/5 run's job 16357824),
# just pointed at the 90/10 pool/out dirs.
#
# Run this AFTER victr_precompute_90_10_vision_job.sh has stopped (either
# TIMEOUT or COMPLETED) -- if it timed out mid-episode on the sort task, this
# job's precompute_retrieval_context.py resumes from the last completed
# 256-frame window (per-episode *_progress.json + *.npy.partial), not from
# scratch. If the 8-way job somehow finished the sort task on its own, this
# job's _already_done check makes it a fast no-op.
#
# Single-shard, single-task run (not the 8-way split victr_precompute_90_10_
# vision_job.sh uses for the full task set) -- this is the only remaining
# work, so it gets the full cpu_short 32-CPU budget as one process instead of
# splitting it.
#
# cpu_short's own QOS caps MaxWall at 06:00:00 (cpu=32,mem=120G per user) --
# 05:45:00 here to match the 85/10/5 sort-task job's budget with a little
# margin; a bigger --time is silently rejected outright (sbatch errors with
# "partition 'cpu_short' is not valid for this job", not a clearer QOS-limit
# message).
#
# Submit (CPU account, not the GPU training account):
#   sbatch --account=torch_pr_564_tandon_advanced --qos=cpu48 \
#     shells/slurm/victr_precompute_90_10_vision_sort_task_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-90-10-vision-sort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-90-10-vision-sort-%j.out
#SBATCH --error=logs/victr-precompute-90-10-vision-sort-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_90_10_vision_sort_task_job.sh <repo_dir>"}
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
TASK="sort the items into their containers"

test -f "$POOL_DIR/all_episodes.json" || { echo "missing $POOL_DIR/all_episodes.json" >&2; exit 1; }
test -d "$OPENPI_POOL_DIR_224" || { echo "missing $OPENPI_POOL_DIR_224" >&2; exit 1; }

echo "--- metric=vision, task='$TASK': single shard, 32 threads ---"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
    --pool-dir "$OPENPI_POOL_DIR_224" \
    --all-episodes-json "$POOL_DIR/all_episodes.json" \
    --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
    --out-dir "$OUT_DIR" \
    --metric vision \
    --tasks "$TASK" \
    --num-shards 1 --shard-index 0

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

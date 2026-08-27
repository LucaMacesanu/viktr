#!/bin/bash
# ============================================================================
# Finishes the vision-metric retrieval-context precompute for the one straggler
# task job 16357824 never completed: "sort the items into their containers" (50
# episodes, 20 still missing as of 2026-08-26 -- see that job's TIMEOUT, 1764/1784
# episodes done, all 20 misses on this one task/shard).
#
# Single-shard, single-task run (not the 8-way split victr_precompute_expanded_
# vision_job.sh uses for the full task set) -- this is the only remaining work, so
# it gets the full cpu_short 32-CPU budget as one process instead of splitting it.
#
# Now safe to just re-run at coarse granularity: precompute_retrieval_context.py
# processes each episode in --frame-window-size windows and checkpoints per window
# (<out_dir>/<metric>/<episode>_progress.json + *.npy.partial), so if this job also
# times out, resubmitting the exact same command resumes mid-episode instead of
# redoing this task's episodes from scratch -- unlike job 16357824's run, which lost
# all progress on this task when it hit the wall clock.
#
# cpu_short's own QOS caps MaxWall at 06:00:00 (cpu=32,mem=120G per user) -- 05:45:00
# here to match job 16357824's original budget with a little margin; a bigger --time
# is silently rejected outright (sbatch errors with "partition 'cpu_short' is not
# valid for this job", not a clearer QOS-limit message).
#
# Submit (CPU account, not the GPU training account -- confirmed via job 16357824's
# own sacct record):
#   sbatch --account=torch_pr_564_tandon_advanced --qos=cpu48 \
#     shells/slurm/victr_precompute_expanded_vision_sort_task_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-expanded-vision-sort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-expanded-vision-sort-%j.out
#SBATCH --error=logs/victr-precompute-expanded-vision-sort-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_expanded_vision_sort_task_job.sh <repo_dir>"}
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

#!/bin/bash
# ============================================================================
# Adds a "Vision (+wrist)" column to the vision-vs-vision+value retrieval
# deep-dive (shells/slurm/victr_divergent_retrievals_job.sh must have already
# completed -- this reuses its cached shard_*.npz scores/selection, does NOT
# rescan the dataset): for the same top-10-per-task frames already selected,
# shows what a hypothetical zed+fish0+fish1 fused-distance vision metric would
# retrieve, next to what the production zed-only "vision" metric actually
# retrieved -- see scripts/render_divergent_retrieval_grids.py's module
# docstring for the exact z-score-per-camera-then-sum fusion math. Grids go
# from 4 to 5 columns: Observation, Vision, Vision (+wrist), Value,
# Vision+Value.
#
# Does NOT touch outputs/victr/retrieval_context_canonical_3cam_k1 or any
# third_party/openpi/assets pool -- those are read-only inputs here, and are
# actively being read by 3 live GPU training jobs
# (yor_icl_victr_{vision,value,vision_value}_canonical_ctx3cam) as of this
# writing. This job only reads scripts/build_pool_canonical_3cam.py's raw
# pool (third_party/openpi/assets/victr_icl_pool_canonical_3cam_224/*.pkl) to
# compute NEW fish0/fish1 DINOv2 embeddings (that pool's key_embeddings field
# is zed-only) and caches them under outputs/victr/wrist_vision_embeddings/,
# a brand-new directory.
#
# Two phases:
#   1. One scripts/precompute_wrist_vision_embeddings.py process PER TASK (20
#      canonical tasks), run in bounded-concurrency batches (POOL_PARALLELISM
#      at a time) rather than all 20 at once -- some task pools are 5GB+ on
#      disk and unpickling one temporarily uses several times that in RAM, so
#      running every task concurrently risks a much larger peak than the
#      job's --mem budget accounts for.
#   2. One single-process render step
#      (scripts/render_divergent_retrieval_grids.py --wrist-vision-pool-dir/
#      --wrist-vision-cache-dir), reusing phase 1's now-warm cache -- fast,
#      no more DINOv2 passes needed.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_wrist_vision_grids_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-wrist-vision-grids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --partition=cs
#SBATCH --output=logs/victr-wrist-vision-grids-%j.out
#SBATCH --error=logs/victr-wrist-vision-grids-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_wrist_vision_grids_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

ALL_EPISODES_JSON="outputs/victr/icl_pool_canonical/all_episodes.json"
CONTEXT_DIR="outputs/victr/retrieval_context_canonical_3cam_k1"
ICL_DATASET_ROOT="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs"
SHARDS_DIR="outputs/victr/divergent_retrievals/shards"
WRIST_POOL_DIR="third_party/openpi/assets/victr_icl_pool_canonical_3cam_224"
WRIST_CACHE_DIR="outputs/victr/wrist_vision_embeddings"
GRIDS_DIR="outputs/victr/divergent_retrieval_grids"
TOP_K_PER_TASK=10
POOL_PARALLELISM=6

test -f "$ALL_EPISODES_JSON" || { echo "missing $ALL_EPISODES_JSON" >&2; exit 1; }
test -d "$SHARDS_DIR" || { echo "missing $SHARDS_DIR -- run victr_divergent_retrievals_job.sh first" >&2; exit 1; }
test -d "$WRIST_POOL_DIR" || { echo "missing $WRIST_POOL_DIR" >&2; exit 1; }

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mapfile -d '' -t TASKS < <(third_party/openpi/.venv/bin/python3 -c "
import json, sys
tasks = json.load(open('$ALL_EPISODES_JSON'))
sys.stdout.write('\0'.join(tasks.keys()))
")
echo "--- phase 1: pre-warming wrist-vision embeddings for ${#TASKS[@]} tasks, $POOL_PARALLELISM at a time ---"

running=0
fail=0
for task in "${TASKS[@]}"; do
    third_party/openpi/.venv/bin/python3 scripts/precompute_wrist_vision_embeddings.py \
        --pool-dir "$WRIST_POOL_DIR" \
        --cache-dir "$WRIST_CACHE_DIR" \
        --task "$task" \
        > "logs/wrist-vision-embed-${SLURM_JOB_ID}-$(echo "$task" | tr -c 'a-zA-Z0-9' '_').out" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$POOL_PARALLELISM" ]; then
        wait -n || fail=1
        running=$((running - 1))
    fi
done
wait || fail=1
if [ "$fail" -ne 0 ]; then
    echo "phase 1: one or more tasks failed -- check logs/wrist-vision-embed-${SLURM_JOB_ID}-*.out" >&2
    exit 1
fi
echo "--- phase 1: all ${#TASKS[@]} tasks' wrist-vision embeddings cached ---"

echo "--- phase 2: rendering 5-column grids (top $TOP_K_PER_TASK per task) ---"
third_party/openpi/.venv/bin/python3 scripts/render_divergent_retrieval_grids.py \
    --shards-dir "$SHARDS_DIR" \
    --all-episodes-json "$ALL_EPISODES_JSON" \
    --context-dir "$CONTEXT_DIR" \
    --icl-dataset-root "$ICL_DATASET_ROOT" \
    --out-dir "$GRIDS_DIR" \
    --top-k-per-task "$TOP_K_PER_TASK" \
    --wrist-vision-pool-dir "$WRIST_POOL_DIR" \
    --wrist-vision-cache-dir "$WRIST_CACHE_DIR"

echo "============================================================"
echo "Done: $(date)"
echo "PNGs:  $GRIDS_DIR/"
echo "Zip:   ${GRIDS_DIR}.zip"
echo "============================================================"

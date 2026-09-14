#!/bin/bash
# ============================================================================
# Vision-vs-vision+value retrieval deep-dive: scans every precomputed context
# frame under outputs/victr/retrieval_context_canonical_3cam_k1/{vision,
# vision_value}/ (all 20 canonical tasks, 1,186 episodes, ~482k frames -- see
# shells/slurm/victr_precompute_canonical_3cam_k1_job.sh, which must have
# already completed for all 3 metrics), finds the top 100 frames where the
# "vision" and "vision_value" metrics retrieved visually different context,
# and renders each as a 3-row (top/zed, left/fish0, right/fish1 camera views)
# x 4-column (observation, vision, value, vision+value) comparison PNG --
# top-K is taken PER TASK (one subfolder per task in the output), not a
# single global top-K, so every one of the 20 canonical tasks is represented
# instead of the ranking being dominated by whichever task happens to have
# the largest-magnitude disagreements. See scripts/find_divergent_retrievals.py
# and scripts/render_divergent_retrieval_grids.py's own docstrings for the
# scoring/rendering details.
#
# Two phases, both in this one job:
#   1. NSHARDS parallel CPU shards, each scoring its own slice of episodes
#      (scripts/find_divergent_retrievals.py) -- pure numpy mmap reads over
#      the vision/vision_value _images.npy files, ~434GB total to scan, no
#      DINOv2/dataset/video decoding needed for this phase.
#   2. One single-process merge + render step
#      (scripts/render_divergent_retrieval_grids.py): takes the global top
#      100 by diff score, decodes each one's actual observation frame from
#      the icl-dataset LeRobotDataset (grouped by episode so a given
#      episode's video is only opened once), and zips the finished PNG folder
#      for easy transfer off-cluster.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_divergent_retrievals_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-divergent-retrievals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --partition=cs
#SBATCH --output=logs/victr-divergent-retrievals-%j.out
#SBATCH --error=logs/victr-divergent-retrievals-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_divergent_retrievals_job.sh <repo_dir>"}
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
GRIDS_DIR="outputs/victr/divergent_retrieval_grids"
TOP_K_PER_TASK=10
NSHARDS=20

test -f "$ALL_EPISODES_JSON" || { echo "missing $ALL_EPISODES_JSON" >&2; exit 1; }
test -d "$CONTEXT_DIR/vision" || { echo "missing $CONTEXT_DIR/vision -- run victr_precompute_canonical_3cam_k1_job.sh first" >&2; exit 1; }
test -d "$CONTEXT_DIR/vision_value" || { echo "missing $CONTEXT_DIR/vision_value -- run victr_precompute_canonical_3cam_k1_job.sh first" >&2; exit 1; }
test -d "$CONTEXT_DIR/value" || { echo "missing $CONTEXT_DIR/value -- run victr_precompute_canonical_3cam_k1_job.sh first" >&2; exit 1; }

echo "--- phase 1: scoring $NSHARDS parallel shards ---"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 scripts/find_divergent_retrievals.py \
        --all-episodes-json "$ALL_EPISODES_JSON" \
        --context-dir "$CONTEXT_DIR" \
        --out-dir "$SHARDS_DIR" \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/divergent-retrievals-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "phase 1: one or more shards failed -- check logs/divergent-retrievals-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- phase 1: all $NSHARDS shards completed ---"

echo "--- phase 2: merging + rendering top $TOP_K_PER_TASK grids per task ---"
third_party/openpi/.venv/bin/python3 scripts/render_divergent_retrieval_grids.py \
    --shards-dir "$SHARDS_DIR" \
    --all-episodes-json "$ALL_EPISODES_JSON" \
    --context-dir "$CONTEXT_DIR" \
    --icl-dataset-root "$ICL_DATASET_ROOT" \
    --out-dir "$GRIDS_DIR" \
    --top-k-per-task "$TOP_K_PER_TASK"

echo "============================================================"
echo "Done: $(date)"
echo "PNGs:  $GRIDS_DIR/"
echo "Zip:   ${GRIDS_DIR}.zip"
echo "============================================================"

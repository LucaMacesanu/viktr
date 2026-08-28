#!/bin/bash
# ============================================================================
# Pre-decodes and caches the 3 primary training cameras (observation.images.
# {zed,fish0,fish1}) for the expanded 1,784-episode icl-dataset subset, as a new
# image-mode LeRobotDataset under third_party/openpi/assets/ -- see
# third_party/openpi/scripts/precompute_frame_cache.py's docstring for the full
# motivation (job 16476252, yor_icl_pi05_expanded_frozen_vision_full, killed by
# the cluster's <80% GPU-util reaper: live video decode couldn't keep the GPU
# fed once the vision backbone was frozen).
#
# CPU-only (decode is CPU/IO-bound, no GPU needed). NSHARDS=32 at 1 CPU each
# (cpu_short's per-user cap, same budget victr_precompute_expanded_vision_job.sh
# uses) -- each shard is an independent process (pyav/torchcodec decode doesn't
# obviously benefit from intra-process threading the way DINOv2's forward pass
# does), writing its own independent LeRobotDataset under --shard-dir, then a
# final --merge pass combines them (see the script's docstring for why a single
# shared LeRobotDataset can't be written by concurrent processes directly).
#
# Sized from an actual 100-frame end-to-end pilot (decode + resize + PNG write,
# single process, torchcodec backend): 324ms/frame. 1,019,948 frames / 32 shards
# x 0.324s/frame =~ 2.9h/shard, comfortably inside cpu_short's 6h cap. Output
# size: measured ~50KB/image (PNG, embedded in parquet) x 3 cameras x 1,019,948
# frames =~ 150GB total.
#
# Resumable: rerunning skips any shard whose output dir already exists, and
# --merge always overwrites the final dir (safe to rerun once every shard is done).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_frame_cache_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-precompute-frame-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-frame-cache-%j.out
#SBATCH --error=logs/victr-precompute-frame-cache-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_frame_cache_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

# openpi's venv has GPU jax installed (needed for training); on this CPU-only
# partition jax's cuda plugin still tries to init CUDA for resize_with_pad and
# hard-errors (FAILED_PRECONDITION: No visible GPU devices) instead of falling
# back, since no GPU is allocated here. Force CPU explicitly.
export JAX_PLATFORMS=cpu

ICL_DATASET_ROOT=/scratch/lim2045/icl_ws/icl-dataset
EPISODES_JSON="third_party/openpi/assets/yor_icl_expanded_episodes.json"
SHARD_DIR="outputs/frame_cache_expanded_frozen_vision/shards"
OUT_DIR="third_party/openpi/assets/yor_icl_pi05_expanded_frozen_vision_cache_224"
NSHARDS=32

echo "--- launching $NSHARDS parallel shards (openpi venv) ---"
pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_frame_cache.py \
        --icl-dataset-root "$ICL_DATASET_ROOT" \
        --episodes-json "$EPISODES_JSON" \
        --shard-dir "$SHARD_DIR" \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-frame-cache-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "one or more shards failed -- check logs/precompute-frame-cache-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- all $NSHARDS shards completed, merging ---"

third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_frame_cache.py \
    --icl-dataset-root "$ICL_DATASET_ROOT" \
    --episodes-json "$EPISODES_JSON" \
    --shard-dir "$SHARD_DIR" \
    --num-shards "$NSHARDS" --merge \
    --out-dir "$OUT_DIR"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

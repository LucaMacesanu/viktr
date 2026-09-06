#!/bin/bash
# ============================================================================
# Builds the expanded-task-set (31 of 36 tasks) retrieval pool for a 90/10
# train/pool split (no held-out test set -- build_splits(test_frac=0.0), per
# viktr_pool_90_10_job.sh/train.py's own viktr_vision_50k/viktr_value_50k
# TrainConfigs), converts it for openpi, and resizes chunk images to 224x224.
#
# Distinct from victr_precompute_expanded_job.sh's steps 1-2b: that script's
# 85/10/5 split's openpi episodes_path (assets/yor_icl_expanded_episodes.json)
# is the full train+pool+test union, so ~10% of ITS training queries come from
# episodes also embedded in its own retrieval pool -- retrieve_chunks/_topk has
# no self-episode exclusion, so a query can retrieve chunks back out of its own
# episode. This 90/10 split's train_episodes.json (written by
# build_retrieval_pool.py) is train-only, zero overlap with pool_dir's
# episodes by construction -- see viktr_vision_50k's TrainConfig comment in
# third_party/openpi/src/openpi/training/config.py.
#
# Mirrors victr_precompute_expanded_job.sh's steps 1-2b resource sizing
# (cpu_short, no GPU -- DINOv2-embedding just the ~10% pool episodes fits
# within budget there).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_pool_90_10_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-pool-90-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-pool-90-10-%j.out
#SBATCH --error=logs/victr-pool-90-10-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_pool_90_10_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

# Without this, torchcodec fails to load (libavutil.so.57/56/... missing) and
# video decode silently falls back to lerobot's much slower pure-Python pyav
# backend -- this is what made job 16735906 blow through its 4h budget with
# only 28/31 expanded tasks' pools built (victr_precompute_expanded_job.sh
# already does this for the same reason; this script previously didn't).
"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="outputs/victr/icl_pool_expanded_90_10"
OPENPI_POOL_DIR="third_party/openpi/assets/victr_icl_pool_expanded_90_10"
OPENPI_POOL_DIR_224="third_party/openpi/assets/victr_icl_pool_expanded_90_10_224"

echo "--- step 1: build expanded-task 90/10 pool (viktr venv) ---"
uv run python scripts/build_retrieval_pool.py \
    --tasks expanded \
    --out-dir "$POOL_DIR" \
    --seed 42 \
    --train-frac 0.90 --pool-frac 0.10 --test-frac 0.0

echo "--- step 2: convert pool for openpi (viktr venv) ---"
uv run python scripts/convert_pool_for_openpi.py \
    --tasks expanded \
    --pool-dir "$POOL_DIR" \
    --out-dir "$OPENPI_POOL_DIR"

echo "--- step 2b: resize pool to 224x224 (openpi venv) ---"
third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/resize_pool_images.py \
    --pool-dir "$OPENPI_POOL_DIR" \
    --out-dir "$OPENPI_POOL_DIR_224" \
    --primary-camera observation.images.zed

echo "--- step 3: copy train-only episode list into openpi assets ---"
cp "$POOL_DIR/train_episodes.json" third_party/openpi/assets/yor_icl_expanded_90_10_train_episodes.json

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

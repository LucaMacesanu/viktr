#!/bin/bash
# ============================================================================
# Precompute per-frame VICTR retrieval context for
# yor_icl_victr_vision_interp_canonical_ctx3cam: same 3-camera/k=1 redesigned
# context encoding as victr_precompute_canonical_3cam_k1_job.sh, PLUS continuous
# action-interpolation extras (exp_lamda_distance, nearest_action) for
# Pi0VictrConfig.use_action_interpolation (see notes/action_interpolation.md and
# config.py's yor_icl_victr_vision_interp_canonical_ctx3cam comment block).
# vision-metric only -- continuous interpolation requires --metric vision
# (enforced by precompute_retrieval_context.py itself).
#
# Requires: assets/victr_icl_pool_canonical_3cam_224/ (already built) and its
# chunk.action_native30_truncated field.
#
# lamda=3.0, NOT the discrete FAST case's default of 10.0 -- real measurement
# (see config.py's yor_icl_victr_vision_interp_canonical_ctx3cam comment) found
# lamda=10.0 nearly inert on this dataset's real distance scale; lamda=3.0 gives
# a real, meaningful dynamic range (weight~=0.43 for the closest ~10% of real
# neighbors, ~=0.30 at the median, ~=0.05 for the farthest).
#
# Sharding/time budget mirrors victr_precompute_canonical_3cam_k1_job.sh (single
# metric here, so if anything this finishes faster).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_canonical_3cam_k1_interp_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=precompute-canonical-3cam-k1-interp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=250G
#SBATCH --time=05:45:00
#SBATCH --partition=cs
#SBATCH --output=logs/victr-precompute-canonical-3cam-k1-interp-%j.out
#SBATCH --error=logs/victr-precompute-canonical-3cam-k1-interp-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_canonical_3cam_k1_interp_job.sh <repo_dir>"}
cd "$REPO_DIR"

METRIC=vision

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME, metric=$METRIC (continuous interpolation)"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="third_party/openpi/assets/victr_icl_pool_canonical_3cam_224"
ALL_EPISODES_JSON="outputs/victr/icl_pool_canonical/all_episodes.json"
ICL_DATASET_ROOT="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs"
OUT_DIR="outputs/victr/retrieval_context_canonical_3cam_k1_interp"
FAST_TOKENIZER_PATH="$REPO_DIR/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical"
NORM_STATS_JSON="third_party/openpi/assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json"
CAMERA_KEYS="observation.images.zed,observation.images.fish0,observation.images.fish1"
LAMDA=3.0
NSHARDS=20

test -f "$ALL_EPISODES_JSON" || { echo "missing $ALL_EPISODES_JSON" >&2; exit 1; }
test -d "$POOL_DIR" || { echo "missing $POOL_DIR" >&2; exit 1; }
test -d "$FAST_TOKENIZER_PATH" || { echo "missing $FAST_TOKENIZER_PATH" >&2; exit 1; }
test -f "$NORM_STATS_JSON" || { echo "missing $NORM_STATS_JSON" >&2; exit 1; }

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

echo "--- metric=$METRIC: launching $NSHARDS parallel shards, $OMP_NUM_THREADS threads each ---"
pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir "$POOL_DIR" \
        --all-episodes-json "$ALL_EPISODES_JSON" \
        --icl-dataset-root "$ICL_DATASET_ROOT" \
        --out-dir "$OUT_DIR" \
        --metric "$METRIC" \
        --tasks all \
        --num-context-chunks 1 \
        --context-frames-per-chunk 3 \
        --context-text-max-length 256 \
        --context-camera-keys "$CAMERA_KEYS" \
        --fast-tokenizer-path "$FAST_TOKENIZER_PATH" \
        --state-norm-stats-json "$NORM_STATS_JSON" \
        --action-norm-stats-json "$NORM_STATS_JSON" \
        --use-continuous-action-interpolation \
        --lamda "$LAMDA" \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-canonical-3cam-k1-interp-${METRIC}-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=$METRIC: one or more shards failed -- check logs/precompute-canonical-3cam-k1-interp-${METRIC}-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=$METRIC: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================================
# Precompute per-frame VICTR retrieval context for the REDESIGNED context
# encoding (3 camera views per chunk, k=1, task+digitized-state+truncated-action
# combined into one token span via the real pi0.5 mechanism -- see
# openpi.policies.yor_retrieval.build_chunk_context_tokens and the "Redesigned
# VICTR context" plan). One job per metric -- submit 3 times (vision/value/
# vision_value), each independent and safe to run concurrently, since all 3 new
# arms (yor_icl_victr_{vision,value,vision_value}_canonical_ctx3cam) read this
# same OUT_DIR, keyed by their own metric subdirectory.
#
# Requires: assets/victr_icl_pool_canonical_3cam_224/ (scripts/build_pool_canonical_3cam.py,
# already built) and its chunk.action_native30_truncated field.
#
# --context-text-max-length 256: real exhaustive measurement (all 9,856 native
# 30-step windows, all 20 canonical pools) found mean=127.6, p99=197.5, max=248
# tokens for this exact combined encoding -- 256 covers the true worst case with
# margin, no silent per-frame truncation risk.
#
# Mirrors victr_precompute_canonical_interp_vision_job.sh's sharding (8 shards x
# 4 threads = 32 cpus) and time budget (that job, WITH extra action-interpolation
# work per frame, took under 5:45 for the ~512k-frame canonical set -- this path
# does strictly less work per frame, so the same budget is generous headroom).
#
# Submit (once per metric):
#   sbatch --account=<account> shells/slurm/victr_precompute_canonical_3cam_k1_job.sh <repo_dir> vision
#   sbatch --account=<account> shells/slurm/victr_precompute_canonical_3cam_k1_job.sh <repo_dir> value
#   sbatch --account=<account> shells/slurm/victr_precompute_canonical_3cam_k1_job.sh <repo_dir> vision_value
# ============================================================================
#SBATCH --job-name=precompute-canonical-3cam-k1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-canonical-3cam-k1-%j.out
#SBATCH --error=logs/victr-precompute-canonical-3cam-k1-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_canonical_3cam_k1_job.sh <repo_dir> <metric>"}
METRIC=${2:?"Usage: sbatch ... victr_precompute_canonical_3cam_k1_job.sh <repo_dir> <metric>  (metric: vision|value|vision_value)"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME, metric=$METRIC"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="third_party/openpi/assets/victr_icl_pool_canonical_3cam_224"
ALL_EPISODES_JSON="outputs/victr/icl_pool_canonical/all_episodes.json"
ICL_DATASET_ROOT="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs"
OUT_DIR="outputs/victr/retrieval_context_canonical_3cam_k1"
FAST_TOKENIZER_PATH="$REPO_DIR/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical"
NORM_STATS_JSON="third_party/openpi/assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json"
CAMERA_KEYS="observation.images.zed,observation.images.fish0,observation.images.fish1"
NSHARDS=8

test -f "$ALL_EPISODES_JSON" || { echo "missing $ALL_EPISODES_JSON" >&2; exit 1; }
test -d "$POOL_DIR" || { echo "missing $POOL_DIR" >&2; exit 1; }
test -d "$FAST_TOKENIZER_PATH" || { echo "missing $FAST_TOKENIZER_PATH" >&2; exit 1; }
test -f "$NORM_STATS_JSON" || { echo "missing $NORM_STATS_JSON" >&2; exit 1; }

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "--- metric=$METRIC: launching $NSHARDS parallel shards, 4 threads each ---"
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
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-canonical-3cam-k1-${METRIC}-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=$METRIC: one or more shards failed -- check logs/precompute-canonical-3cam-k1-${METRIC}-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=$METRIC: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

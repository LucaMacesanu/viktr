#!/bin/bash
# ============================================================================
# Precompute per-frame VICTR retrieval context (metric=vision) PLUS RICL
# action-interpolation extras (exp_lamda_distance/nearest_action_tokens/
# nearest_action_tokens_mask) for the ICRA canonical trimmed 20-task/1,186-
# episode set (assets/yor_icl_canonical_extended_episodes.json), so
# yor_icl_fast_victr_vision_interp_canonical (arm 6, RICL) can read an O(1)
# precomputed_context_dir lookup instead of live per-sample DINOv2 embed +
# pool search + tokenization -- job 17355272 averaged only 15.7% GPU
# utilization on the live path (DINOv2 runs CPU-only in the dataloader, see
# yor_retrieval.load_dinov2's docstring) and was killed for it.
#
# Requires the precompute_retrieval_context.py / yor_retrieval.py changes that
# add --use-action-interpolation support (previously use_action_interpolation
# required precomputed_dir=None -- see RetrievalContextInputs.canonical_actions'
# docstring for why the pool's dim-16:20 zeroing also needed a canonical-aware
# fix, not just this precompute extension).
#
# --context-text-max-length 1024: measured directly against this exact pool
# (victr_icl_pool_canonical_224) -- chunk_to_context_text's digitized Task/
# State/Action text ran 700-713 tokens across every task sampled (PaligemmaTokenizer),
# consistent with job 17355272's live truncation warnings (671-702, clipped to
# the old default of 64). 1024 leaves headroom above the measured range.
#
# Mirrors victr_precompute_expanded_vision_job.sh's sharding (8 shards x 4
# threads = 32 cpus, DINOv2's CPU forward pass is compute-bound and benefits
# from multithreading) -- this set is ~512k frames vs. the expanded set's
# ~1.02M, so expect roughly half the wall time (~1.8-2.5h incl. the extra
# action-interpolation work per frame).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_precompute_canonical_interp_vision_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=precompute-canonical-interp-vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:45:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-precompute-canonical-interp-vision-%j.out
#SBATCH --error=logs/victr-precompute-canonical-interp-vision-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_precompute_canonical_interp_vision_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

POOL_DIR="third_party/openpi/assets/victr_icl_pool_canonical_224"
ALL_EPISODES_JSON="outputs/victr/icl_pool_canonical/all_episodes.json"
ICL_DATASET_ROOT="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs"
OUT_DIR="outputs/victr/retrieval_context_canonical_interp"
FAST_TOKENIZER_PATH="$REPO_DIR/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical"
ACTION_NORM_STATS_JSON="third_party/openpi/assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json"
NSHARDS=8

test -f "$ALL_EPISODES_JSON" || { echo "missing $ALL_EPISODES_JSON" >&2; exit 1; }
test -d "$POOL_DIR" || { echo "missing $POOL_DIR" >&2; exit 1; }
test -d "$FAST_TOKENIZER_PATH" || { echo "missing $FAST_TOKENIZER_PATH -- run victr_fit_canonical_fast_tokenizer_job.sh first" >&2; exit 1; }
test -f "$ACTION_NORM_STATS_JSON" || { echo "missing $ACTION_NORM_STATS_JSON" >&2; exit 1; }

echo "--- metric=vision + action-interpolation: launching $NSHARDS parallel shards, 4 threads each ---"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pids=()
for ((i = 0; i < NSHARDS; i++)); do
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir "$POOL_DIR" \
        --all-episodes-json "$ALL_EPISODES_JSON" \
        --icl-dataset-root "$ICL_DATASET_ROOT" \
        --out-dir "$OUT_DIR" \
        --metric vision \
        --tasks all \
        --context-text-max-length 1024 \
        --use-action-interpolation \
        --lamda 10.0 \
        --max-action-tokens 224 \
        --action-horizon 30 \
        --canonical-actions \
        --fast-tokenizer-path "$FAST_TOKENIZER_PATH" \
        --action-norm-stats-json "$ACTION_NORM_STATS_JSON" \
        --num-shards "$NSHARDS" --shard-index "$i" \
        > "logs/precompute-canonical-interp-vision-${SLURM_JOB_ID}-shard${i}.out" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "metric=vision: one or more shards failed -- check logs/precompute-canonical-interp-vision-${SLURM_JOB_ID}-shard*.out" >&2
    exit 1
fi
echo "--- metric=vision + action-interpolation: all $NSHARDS shards completed ---"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

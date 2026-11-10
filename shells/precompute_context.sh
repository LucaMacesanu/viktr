#!/usr/bin/env bash
# Precomputes per-frame retrieval context for the VICTR training configs, so the
# dataloader does an O(1) lookup instead of a live DINOv2 + nearest-neighbour search.
# Needs one GPU (DINOv2 query embeddings). Restartable: finished episodes are skipped.
#
#   shells/precompute_context.sh ctx3cam <vision|value|vision_value>
#       -> yor_icl_victr_{vision,value,vision_value}_canonical_ctx3cam
#   shells/precompute_context.sh ctx3cam_interp
#       -> yor_icl_victr_vision_interp_canonical_ctx3cam (RICL-style baseline, pi0.5)
#   shells/precompute_context.sh fast_interp
#       -> yor_icl_fast_victr_vision_interp_canonical (RICL-style baseline, pi0-FAST)
#
# NUM_SHARDS (default 4) parallel worker processes split the work; lower it if you
# run out of host RAM (each worker holds one task's retrieval pool in memory).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VARIANT=${1:?"usage: precompute_context.sh <ctx3cam|ctx3cam_interp|fast_interp> [metric]"}
NUM_SHARDS=${NUM_SHARDS:-4}
CAMERAS="observation.images.zed,observation.images.fish0,observation.images.fish1"

case "$VARIANT" in
    ctx3cam)
        METRIC=${2:?"ctx3cam needs a metric: vision | value | vision_value"}
        POOL=victr_icl_pool_canonical_3cam_224
        OUT=canonical_3cam_k1
        ARGS=(--metric "$METRIC" --num-context-chunks 1 --context-frames-per-chunk 3
              --context-text-max-length 256 --context-camera-keys "$CAMERAS"
              --state-norm-stats-json "$NORM_STATS_JSON" --action-norm-stats-json "$NORM_STATS_JSON")
        ;;
    ctx3cam_interp)
        POOL=victr_icl_pool_canonical_3cam_224
        OUT=canonical_3cam_k1_interp
        ARGS=(--metric vision --num-context-chunks 1 --context-frames-per-chunk 3
              --context-text-max-length 256 --context-camera-keys "$CAMERAS"
              --state-norm-stats-json "$NORM_STATS_JSON" --action-norm-stats-json "$NORM_STATS_JSON"
              --use-continuous-action-interpolation --lamda 3.0)
        ;;
    fast_interp)
        POOL=victr_icl_pool_canonical_224
        OUT=canonical_interp
        ARGS=(--metric vision --context-text-max-length 1024 --use-action-interpolation --lamda 10.0
              --max-action-tokens 224 --action-horizon 30 --canonical-actions
              --action-norm-stats-json "$NORM_STATS_JSON")
        ;;
    *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

POOL_DIR="$OPENPI_DIR/assets/$POOL"
for f in "$POOL_DIR" "$VICTR_FAST_TOKENIZER" "$VICTR_DATA_ROOT"; do
    [ -e "$f" ] || { echo "missing $f -- run shells/download_data.sh first" >&2; exit 1; }
done

mkdir -p "$REPO_ROOT/logs"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    "$PY" "$OPENPI_DIR/scripts/precompute_retrieval_context.py" \
        --pool-dir "$POOL_DIR" \
        --all-episodes-json "$POOL_EPISODES_JSON" \
        --icl-dataset-root "$VICTR_DATA_ROOT" \
        --out-dir "$VICTR_RETRIEVAL_CONTEXT_DIR/$OUT" \
        --tasks all \
        --fast-tokenizer-path "$VICTR_FAST_TOKENIZER" \
        --balance-episodes-across-shards \
        --num-shards "$NUM_SHARDS" --shard-index "$i" \
        "${ARGS[@]}" \
        > "$REPO_ROOT/logs/precompute-$OUT-${METRIC:-vision}-shard$i.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" = 0 ] || { echo "a shard failed -- see logs/precompute-$OUT-*.log (rerun to resume)" >&2; exit 1; }
echo "done: $VICTR_RETRIEVAL_CONTEXT_DIR/$OUT"

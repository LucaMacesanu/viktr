#!/usr/bin/env bash
# Downloads the training dataset and the prebuilt VICTR assets from the Hugging Face Hub.
#
#   shells/download_data.sh            # dataset + tokenizer + 3-camera retrieval pool
#   shells/download_data.sh --fast     # also the single-camera pool used by the pi0-FAST RICL arm
#   shells/download_data.sh --stage1   # also the stage-1 checkpoint (skip training stage 1)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# TODO(release): set these to the published Hub repos.
DATASET_REPO="${VICTR_HF_DATASET_REPO:-TODO/icl-dataset-fixed-obs}"
ASSETS_REPO="${VICTR_HF_ASSETS_REPO:-TODO/victr-assets}"

includes=(--include "fast_tokenizer/*" --include "victr_icl_pool_canonical_3cam_224/*")
for arg in "$@"; do
    case "$arg" in
        --fast) includes+=(--include "victr_icl_pool_canonical_224/*") ;;
        --stage1) includes+=(--include "checkpoints/yor_icl_pi05_canonical_extended/*") ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

HF="$OPENPI_DIR/.venv/bin/huggingface-cli"
"$HF" download "$DATASET_REPO" --repo-type dataset --local-dir "$VICTR_DATA_ROOT"
"$HF" download "$ASSETS_REPO" "${includes[@]}" --local-dir "$OPENPI_DIR/assets"
if [ -d "$OPENPI_DIR/assets/checkpoints" ]; then
    mkdir -p "$OPENPI_DIR/checkpoints"
    cp -rn "$OPENPI_DIR/assets/checkpoints/." "$OPENPI_DIR/checkpoints/"
    rm -rf "$OPENPI_DIR/assets/checkpoints"
fi

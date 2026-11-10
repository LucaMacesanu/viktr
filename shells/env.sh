#!/usr/bin/env bash
# Sourced by the other shells/*.sh scripts. Sets up paths shared by every step.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$REPO_ROOT/third_party/openpi"
PY="$OPENPI_DIR/.venv/bin/python"

# Defaults match third_party/openpi/src/openpi/training/config.py; override to relocate.
export VICTR_DATA_ROOT="${VICTR_DATA_ROOT:-$OPENPI_DIR/data/icl-dataset-fixed-obs}"
export VICTR_FAST_TOKENIZER="${VICTR_FAST_TOKENIZER:-$OPENPI_DIR/assets/fast_tokenizer/yor-icl-canonical}"
export VICTR_RETRIEVAL_CONTEXT_DIR="${VICTR_RETRIEVAL_CONTEXT_DIR:-$OPENPI_DIR/assets/retrieval_context}"
export VICTR_WARMSTART_PARAMS="${VICTR_WARMSTART_PARAMS:-$OPENPI_DIR/checkpoints/yor_icl_pi05_canonical_extended/stage1/40000/params}"
NORM_STATS_JSON="$OPENPI_DIR/assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json"
POOL_EPISODES_JSON="$OPENPI_DIR/assets/victr_pool_all_episodes.json"

if [ ! -x "$PY" ]; then
    echo "openpi venv not found at $OPENPI_DIR/.venv -- run: (cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)" >&2
    exit 1
fi

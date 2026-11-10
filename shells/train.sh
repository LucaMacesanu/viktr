#!/usr/bin/env bash
# Trains one VICTR / baseline config with openpi (JAX). Uses every visible GPU
# (data-parallel); restrict with CUDA_VISIBLE_DEVICES.
#
#   shells/train.sh <config_name> <exp_name> [extra scripts/train.py args...]
#
# e.g. shells/train.sh yor_icl_pi05_canonical_extended stage1
#      shells/train.sh yor_icl_victr_vision_canonical_ctx3cam main --batch-size 64
#
# Checkpoints go to third_party/openpi/checkpoints/<config_name>/<exp_name>/.
# Pass --resume to continue a run or --overwrite to restart one.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG=${1:?"usage: train.sh <config_name> <exp_name> [extra args...]"}
EXP=${2:?"usage: train.sh <config_name> <exp_name> [extra args...]"}
shift 2

nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible" >&2; exit 1; }

cd "$OPENPI_DIR"
exec "$PY" scripts/train.py "$CONFIG" --exp-name "$EXP" "$@"

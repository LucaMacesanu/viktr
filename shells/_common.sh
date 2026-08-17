#!/usr/bin/env bash
# Sourced by every shells/*.sh wrapper. Not meant to be run directly.
#
# Resolves REPO_ROOT from this file's own location (not the caller's cwd) and
# cds there, so `uv run` always picks up this project's env regardless of
# where the wrapper was invoked from — matters on HPC, where a scheduler
# launches jobs from whatever cwd it feels like.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# HF_HOME defaults to ~/.cache/huggingface, which is where pi05/Robometer
# weights and the lerobot/libero dataset all land — easily 100GB+. On HPC,
# $HOME is usually small-quota/network-mounted; set HF_HOME to scratch or
# project storage before submitting (see notes/hpc.md). Left as a pure
# passthrough here: if the caller already exported HF_HOME, respect it.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# LIBERO's asset cache (~400MB of MuJoCo assets) is NOT configurable via env
# var upstream (hf_libero hardcodes ~/.cache/libero/assets) — documented in
# notes/hpc.md rather than worked around here.

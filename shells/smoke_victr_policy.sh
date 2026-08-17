#!/usr/bin/env bash
# Wraps scripts/smoke_test_victr_policy.py. No arguments. Loads real
# lerobot/pi05_base weights (~2.3B params) — needs a GPU with several GB free
# to run in reasonable time, though it will fall back to CPU.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
uv run python3 scripts/smoke_test_victr_policy.py "$@"

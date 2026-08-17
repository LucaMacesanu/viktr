#!/usr/bin/env bash
# Wraps scripts/smoke_test_robometer.py. No arguments.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
uv run python3 scripts/smoke_test_robometer.py "$@"

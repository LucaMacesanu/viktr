#!/usr/bin/env bash
# Wraps scripts/eval_libero_victr.py. All arguments pass through, e.g.:
#   shells/eval_libero.sh --suite libero_object --task-ids 0,1,2 \
#       --n-episodes 10 --retrieval-metrics vision,value,vision+value,none \
#       --output outputs/eval_libero_object.json
# See scripts/eval_libero_victr.py --help for the full flag list.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
uv run python3 scripts/eval_libero_victr.py "$@"

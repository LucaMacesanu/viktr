#!/usr/bin/env bash
# Submits a shells/slurm/*.slurm.sh job, supplying cluster-specific
# --partition/--account/--gres from shells/slurm/cluster.conf on the sbatch
# command line (these override whatever placeholder #SBATCH directives are
# in the job script itself, since sbatch CLI flags win over in-script
# directives).
#
# Setup (once per cluster/allocation):
#   cp shells/slurm/cluster.conf.example shells/slurm/cluster.conf
#   # then edit shells/slurm/cluster.conf
#
# Usage:
#   shells/slurm/submit.sh shells/slurm/run.slurm.sh shells/eval_libero.sh \
#       --suite libero_object --task-ids 0,1,2 --n-episodes 10 \
#       --retrieval-metrics vision,value,vision+value,none
#   shells/slurm/submit.sh --array=0-9 shells/slurm/eval_libero_array.slurm.sh libero_object
#
# Any leading flags starting with `--` are treated as extra sbatch flags
# (e.g. --array=..., --time=...) and forwarded before the job script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="$REPO_ROOT/shells/slurm/cluster.conf"

if [[ ! -f "$CONF" ]]; then
    echo "error: $CONF not found." >&2
    echo "  cp shells/slurm/cluster.conf.example shells/slurm/cluster.conf" >&2
    echo "  # then edit shells/slurm/cluster.conf" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

: "${SLURM_PARTITION:?SLURM_PARTITION is unset in $CONF}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is unset in $CONF}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"

sbatch_flags=()
while [[ $# -gt 0 && "$1" == --* ]]; do
    sbatch_flags+=("$1")
    shift
done

if [[ $# -lt 1 ]]; then
    echo "usage: shells/slurm/submit.sh [sbatch-flags...] <job-script> [job-args...]" >&2
    exit 1
fi

exec sbatch \
    --partition="$SLURM_PARTITION" \
    --account="$SLURM_ACCOUNT" \
    --gres="$SLURM_GRES" \
    "${sbatch_flags[@]}" \
    "$@"

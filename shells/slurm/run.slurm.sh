#!/usr/bin/env bash
# Generic SLURM launcher for any shells/*.sh wrapper.
#
# Cluster/account-specific settings (partition, account, scratch dir) live in
# shells/slurm/cluster.conf, not in this file — submit via
# shells/slurm/submit.sh, which reads cluster.conf and supplies
# --partition/--account/--gres to sbatch. See shells/slurm/cluster.conf.example
# and notes/hpc.md.
#
# Usage:
#   shells/slurm/submit.sh shells/slurm/run.slurm.sh shells/eval_libero.sh \
#       --suite libero_object --task-ids 0,1,2 --n-episodes 10 \
#       --retrieval-metrics vision,value,vision+value,none \
#       --output outputs/eval_libero_object.json
#
# The first argument is the shells/*.sh wrapper to run; everything after it
# is forwarded verbatim as that wrapper's own arguments.

#SBATCH --job-name=viktr
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
[[ -f shells/slurm/cluster.conf ]] && source shells/slurm/cluster.conf

if [[ $# -lt 1 ]]; then
    echo "usage: shells/slurm/submit.sh shells/slurm/run.slurm.sh <shells/*.sh wrapper> [args...]" >&2
    exit 1
fi

WRAPPER="$1"
shift

# Point large caches at scratch/project storage, not the (likely small-quota)
# home filesystem. SCRATCH_DIR comes from cluster.conf; falls back to
# /scratch/$USER if cluster.conf wasn't set up.
export HF_HOME="${HF_HOME:-${SCRATCH_DIR:-/scratch/$USER}/hf_cache}"

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST running $WRAPPER $*"
srun "$WRAPPER" "$@"

#!/usr/bin/env bash
# Generic SLURM launcher for any shells/*.sh wrapper. Cluster-specific
# directives below (#SBATCH partition/account/qos) are placeholders — fill
# them in for your allocation before submitting (see notes/hpc.md).
#
# Usage:
#   sbatch shells/slurm/run.slurm.sh shells/eval_libero.sh \
#       --suite libero_object --task-ids 0,1,2 --n-episodes 10 \
#       --retrieval-metrics vision,value,vision+value,none \
#       --output outputs/eval_libero_object.json
#
# The first argument is the shells/*.sh wrapper to run; everything after it
# is forwarded verbatim as that wrapper's own arguments.

#SBATCH --job-name=viktr
#SBATCH --partition=TODO_PARTITION
#SBATCH --account=TODO_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch shells/slurm/run.slurm.sh <shells/*.sh wrapper> [args...]" >&2
    exit 1
fi

WRAPPER="$1"
shift

# Point large caches at scratch/project storage, not the (likely small-quota)
# home filesystem — override these at submission time if your cluster uses
# different scratch variable names, e.g. `--export=HF_HOME=/scratch/$USER/hf_cache`.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST running $WRAPPER $*"
srun "$WRAPPER" "$@"

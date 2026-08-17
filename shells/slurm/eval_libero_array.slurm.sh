#!/usr/bin/env bash
# SLURM array template for the scaled LIBERO eval run noted in
# notes/progress.md's "What's next" (10 episodes/task across several tasks,
# all four retrieval-metric variants). Splits by LIBERO task id, one array
# task per index in --array, each running all retrieval metrics for that
# task and writing its own results file.
#
# Cluster/account-specific settings (partition, account, scratch dir) live in
# shells/slurm/cluster.conf, not in this file — submit via
# shells/slurm/submit.sh. See shells/slurm/cluster.conf.example and
# notes/hpc.md.
#
# Usage:
#   shells/slurm/submit.sh --array=0-9 shells/slurm/eval_libero_array.slurm.sh libero_object

#SBATCH --job-name=viktr-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs outputs

# shellcheck disable=SC1091
[[ -f shells/slurm/cluster.conf ]] && source shells/slurm/cluster.conf

SUITE="${1:-libero_object}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?run this via shells/slurm/submit.sh --array=..., not directly}"

export HF_HOME="${HF_HOME:-${SCRATCH_DIR:-/scratch/$USER}/hf_cache}"

echo "job $SLURM_JOB_ID array task $TASK_ID: suite=$SUITE"
srun shells/eval_libero.sh \
    --suite "$SUITE" \
    --task-ids "$TASK_ID" \
    --n-episodes 10 \
    --retrieval-metrics vision,value,vision+value,none \
    --output "outputs/eval_${SUITE}_task${TASK_ID}.json"

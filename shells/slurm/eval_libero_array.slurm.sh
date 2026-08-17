#!/usr/bin/env bash
# SLURM array template for the scaled LIBERO eval run noted in
# notes/progress.md's "What's next" (10 episodes/task across several tasks,
# all four retrieval-metric variants). Splits by LIBERO task id, one array
# task per index in --array, each running all retrieval metrics for that
# task and writing its own results file. Cluster-specific directives below
# are placeholders — fill in for your allocation before submitting (see
# notes/hpc.md).
#
# Usage:
#   sbatch --array=0-9 shells/slurm/eval_libero_array.slurm.sh libero_object

#SBATCH --job-name=viktr-eval
#SBATCH --partition=TODO_PARTITION
#SBATCH --account=TODO_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs outputs

SUITE="${1:-libero_object}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?run this via sbatch --array=..., not directly}"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"

echo "job $SLURM_JOB_ID array task $TASK_ID: suite=$SUITE"
srun shells/eval_libero.sh \
    --suite "$SUITE" \
    --task-ids "$TASK_ID" \
    --n-episodes 10 \
    --retrieval-metrics vision,value,vision+value,none \
    --output "outputs/eval_${SUITE}_task${TASK_ID}.json"

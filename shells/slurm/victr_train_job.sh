#!/bin/bash
# ============================================================================
# SLURM job script — one VictrPolicy retrieval-arm training run.
#
# DO NOT USE FOR REAL TRAINING RUNS. Runs scripts/train_victr.py, which trains
# pi05-based policies via lerobot's native PyTorch stack -- per project decision
# (2026-08-25), openpi is the sole training pipeline for pi05-based policies. See
# train_victr.py's module docstring. Kept only as the reference implementation of
# the vision+value fusion arm, not currently ported to openpi.
#
# Deliberately NOT submitted through this repo's own shells/slurm/submit.sh
# (which hard-requires an explicit --partition from cluster.conf): the
# nyu-finger-robot HPC work in this same project found that omitting
# --partition and letting the account+gres pair auto-infer partition/QOS is
# far more reliable on this cluster (see that repo's notes/hpc-sbatch-
# cheatsheet.md) than pinning one explicitly, which was observed to strand
# jobs in queue behind ReqNodeNotAvail. Submit directly instead:
#
#   sbatch --account=<account> --gres=gpu:h200:1 --time=24:00:00 \
#          shells/slurm/victr_train_job.sh <retrieval-metric> <repo_dir> \
#          <pool_dir> <output_dir> [extra train_victr.py args...]
#
# repo_dir is passed explicitly (not resolved from BASH_SOURCE[0]): slurmd
# copies this script into a spool dir before running it, so BASH_SOURCE[0]
# doesn't point at the real checkout once the job actually runs.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/victr-%j.out
#SBATCH --error=logs/victr-%j.err

set -euo pipefail

RETRIEVAL_METRIC=${1:?"Usage: sbatch ... victr_train_job.sh <metric> <repo_dir> <pool_dir> <output_dir> [extra args...]"}
REPO_DIR=${2:?"Usage: sbatch ... victr_train_job.sh <metric> <repo_dir> <pool_dir> <output_dir> [extra args...]"}
POOL_DIR=${3:?"Usage: sbatch ... victr_train_job.sh <metric> <repo_dir> <pool_dir> <output_dir> [extra args...]"}
OUTPUT_DIR=${4:?"Usage: sbatch ... victr_train_job.sh <metric> <repo_dir> <pool_dir> <output_dir> [extra args...]"}
shift 4

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Retrieval metric: $RETRIEVAL_METRIC"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "============================================================"

module purge

# sbatch inherits the submitting shell's env but does NOT source ~/.bashrc
# itself -- WANDB_API_KEY (added to ~/.bashrc during the earlier pi05 HPC
# setup) won't be here otherwise. set +u/-u around it: /etc/bashrc references
# unset vars that are harmless normally but fatal under this script's set -u.
set +u
[ -f ~/.bashrc ] && source ~/.bashrc
set -u

cd "$REPO_DIR"

# nvidia-smi, not $SLURM_GPUS_ON_NODE: more reliably reflects what's actually visible
# to this job across SLURM versions/configs (matches nyu-finger-robot/train.py's
# n_gpus>1 -> `accelerate launch --multi_gpu` pattern for the pi05 ablation jobs).
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "N_GPUS detected: $N_GPUS"

if [ "$N_GPUS" -gt 1 ]; then
    # Per-job port so co-located multi-GPU jobs on the same node don't clash.
    MAIN_PORT=$((29500 + SLURM_JOB_ID % 1000))
    "$REPO_DIR/.venv/bin/accelerate" launch \
        --multi_gpu \
        --num_processes="$N_GPUS" \
        --num_machines=1 \
        --mixed_precision=no \
        --main_process_port="$MAIN_PORT" \
        scripts/train_victr.py \
        --retrieval-metric "$RETRIEVAL_METRIC" \
        --pool-dir "$POOL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --wandb \
        "$@"
else
    "$REPO_DIR/.venv/bin/python3" scripts/train_victr.py \
        --retrieval-metric "$RETRIEVAL_METRIC" \
        --pool-dir "$POOL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --wandb \
        "$@"
fi

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

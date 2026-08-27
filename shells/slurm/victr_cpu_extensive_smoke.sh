#!/bin/bash
# ============================================================================
# SLURM job script — extensive CPU-only smoke test for all 3 VictrPolicy
# retrieval arms (scripts/train_victr.py, lerobot's native pi05 training stack).
#
# DO NOT USE FOR REAL TRAINING RUNS -- exercises train_victr.py's train-step/
# checkpoint code paths only. Per project decision (2026-08-25), openpi is the
# sole training pipeline for pi05-based policies; see train_victr.py's module
# docstring.
#
# Originally written to run while the (now-superseded, lerobot-based) 2-GPU
# training jobs 16080686/687/688 queued for H200s. Goes beyond the earlier 1-step
# forward-only dry run:
#
#   1. For each arm (vision, value, vision+value): 5 real train steps
#      (forward + backward + optimizer.step + lr_scheduler.step), confirming
#      loss is finite across multiple steps (not just step 1 -- catches
#      dataloader re-iteration / scheduler-state bugs), then a checkpoint
#      save (exercises accelerator.unwrap_model + save_pretrained on this
#      code path for the first time end-to-end).
#   2. eval_victr_offline.py loads all 3 just-saved checkpoints and runs a
#      held-out-split forward pass, exercising the eval script's checkpoint
#      loading (VictrPolicy.from_pretrained, ValueFusionMLP state dict) for
#      the first time against real checkpoints on disk.
#
# Runs on cpu_short (shares that QOS's 120G/user mem cap with any other
# concurrent CPU jobs under this account -- will simply queue if unavailable).
#
#   sbatch --partition=cpu_short --time=02:00:00 --mem=100G --cpus-per-task=8 \
#          shells/slurm/victr_cpu_extensive_smoke.sh <repo_dir> <pool_dir> <smoke_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/victr-cpusmoke-%j.out
#SBATCH --error=logs/victr-cpusmoke-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_cpu_extensive_smoke.sh <repo_dir> <pool_dir> <smoke_dir>"}
POOL_DIR=${2:?"Usage: sbatch ... victr_cpu_extensive_smoke.sh <repo_dir> <pool_dir> <smoke_dir>"}
SMOKE_DIR=${3:?"Usage: sbatch ... victr_cpu_extensive_smoke.sh <repo_dir> <pool_dir> <smoke_dir>"}

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "============================================================"

module purge
set +u
[ -f ~/.bashrc ] && source ~/.bashrc
set -u

cd "$REPO_DIR"
PY="$REPO_DIR/.venv/bin/python3"

CKPT_ARGS=()
for metric in vision value "vision+value"; do
    slug=$(echo "$metric" | tr '+' '_')
    out_dir="$SMOKE_DIR/$slug"
    echo "------------------------------------------------------------"
    echo "training arm: $metric  ->  $out_dir"
    echo "------------------------------------------------------------"
    "$PY" scripts/train_victr.py \
        --retrieval-metric "$metric" \
        --pool-dir "$POOL_DIR" \
        --output-dir "$out_dir" \
        --steps 5 \
        --batch-size 4 \
        --num-workers 2 \
        --log-freq 1 \
        --save-freq 5
    CKPT_ARGS+=("$metric=$out_dir/checkpoints/last")
done

echo "------------------------------------------------------------"
echo "offline eval over the 3 just-trained smoke checkpoints"
echo "------------------------------------------------------------"
"$PY" scripts/eval_victr_offline.py \
    --pool-dir "$POOL_DIR" \
    --checkpoints "${CKPT_ARGS[@]}" \
    --batch-size 4 \
    --out "$SMOKE_DIR/offline_eval.json"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

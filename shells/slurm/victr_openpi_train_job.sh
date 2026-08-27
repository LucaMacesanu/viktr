#!/bin/bash
# ============================================================================
# SLURM job script — openpi (JAX) pi05 fine-tune on icl-dataset's 4-task
# pick-and-place subset, reproducing nyu-finger-robot/configs/
# yor-icl-pi05-easy-pnp-v2.yaml on openpi's own implementation instead of
# lerobot's, to cross-check the lerobot pi05 training pipeline.
#
# Unlike shells/slurm/victr_train_job.sh (lerobot, needs `accelerate launch
# --multi_gpu` for >1 GPU), openpi's JAX training is single-process and
# shards across every locally-visible GPU on its own (training/sharding.py) —
# no launcher wrapper needed regardless of GPU count.
#
# Submit directly (matches victr_train_job.sh's reasoning for not going
# through shells/slurm/submit.sh — omitting --partition and letting
# account+gres auto-infer it is more reliable on this cluster). ALWAYS pass
# --job-name="$EXP_NAME" (or similarly distinguishing) explicitly -- this
# script has no #SBATCH --job-name of its own (can't reference $EXP_NAME from
# inside an #SBATCH comment), so every submission otherwise defaults to the
# same "victr_openpi_train_job.sh" name in squeue/sacct, indistinguishable
# from any other concurrent run of this script (bit us once: two jobs running
# at once with identical queue names, only distinguishable by JOBID):
#
#   sbatch --account=<account> --gres=gpu:h200:2 --time=24:00:00 \
#          --job-name="<exp_name>" \
#          shells/slurm/victr_openpi_train_job.sh <repo_dir> <config_name> <exp_name> \
#          [extra scripts/train.py args...]
#
# config_name is one of openpi/src/openpi/training/config.py's _CONFIGS entries, e.g.
# yor_icl_pi05_easy_pnp_v2 (plain pi05), yor_icl_victr_vision / yor_icl_victr_value
# (VICTR retrieval-conditioned arms).
#
# repo_dir is passed explicitly (not resolved from BASH_SOURCE[0]): slurmd
# copies this script into a spool dir before running it, so BASH_SOURCE[0]
# doesn't point at the real checkout once the job actually runs.
#
# --cpus-per-task/--mem below default to every real 2xH200 run of this exact
# script's ratio (16217307-310: 32 cpus/400G for 2 GPUs -- the CPU-heavy
# dataloader workers, num_workers=8 in every yor_icl_* TrainConfig, do video
# decode +, for VICTR arms, DINOv2 embedding). Two later submissions that
# omitted --cpus-per-task/--mem silently fell back to SLURM's bare-minimum
# default (1 cpu/2G) and were cancelled (16290517/519) -- override explicitly
# via `sbatch --cpus-per-task=... --mem=...` if requesting a different GPU
# count than 2 (sbatch CLI flags take precedence over #SBATCH defaults below).
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=logs/victr-openpi-%j.out
#SBATCH --error=logs/victr-openpi-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_openpi_train_job.sh <repo_dir> <config_name> <exp_name> [extra args...]"}
CONFIG_NAME=${2:?"Usage: sbatch ... victr_openpi_train_job.sh <repo_dir> <config_name> <exp_name> [extra args...]"}
EXP_NAME=${3:?"Usage: sbatch ... victr_openpi_train_job.sh <repo_dir> <config_name> <exp_name> [extra args...]"}
shift 3

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Config:    $CONFIG_NAME"
echo "Exp name:  $EXP_NAME"
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

cd "$REPO_DIR/third_party/openpi"

# lerobot[dataset] pulls in torchcodec unpinned, which uv previously resolved to
# a release built for torch>=2.11 -- incompatible with this project's pinned
# torch==2.7.1 (shared-lib load fails with "undefined symbol
# torch_dtype_float4_e2m1fn_x2"), so every video decode silently fell back to
# lerobot's much slower pure-Python pyav backend. This was a real GPU-utilization
# bottleneck: the CPU dataloader couldn't decode video fast enough to keep the
# GPU fed, even for the plain pi05 baseline with no VICTR overhead. Fixed at the
# dependency level (pyproject.toml now pins torchcodec==0.5, matching torch 2.7),
# but torchcodec 0.5 still needs real FFmpeg 7 shared libs on the loader path,
# which this cluster doesn't have installed system-wide -- pyav's own bundled
# FFmpeg 7 build (in av.libs/) has the right sonames, just hash-mangled filenames,
# so ffmpeg7_shim/ symlinks the plain sonames to pyav's mangled files.
"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "N_GPUS detected: $N_GPUS"

# Live GPU-utilization sampling (admins flagged jobs for sustained <80% GPU
# util): logs one line every 10s for the life of the job so utilization can
# be checked directly instead of inferred from step timing.
GPU_LOG="$REPO_DIR/logs/gpu-util-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 10 > "$GPU_LOG" 2>&1 &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT

.venv/bin/python3 scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    "$@"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

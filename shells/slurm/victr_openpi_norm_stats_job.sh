#!/bin/bash
# ============================================================================
# SLURM job script — computes openpi normalization stats (state/actions
# mean/std/quantiles) for a training config, via scripts/compute_norm_stats.py.
# CPU-only (video decode + a plain running-stats accumulation, no model
# forward pass) -- required once per new config name before its first
# training job (weight_loader/DataConfig.create() reads
# assets/<config_name>/icl-dataset/norm_stats.json, which doesn't exist for a
# newly-added config until this has run).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_openpi_norm_stats_job.sh <repo_dir> <config_name>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-openpi-norm-stats-%j.out
#SBATCH --error=logs/victr-openpi-norm-stats-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_openpi_norm_stats_job.sh <repo_dir> <config_name>"}
CONFIG_NAME=${2:?"Usage: sbatch ... victr_openpi_norm_stats_job.sh <repo_dir> <config_name>"}

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Config:    $CONFIG_NAME"
echo "Started:   $(date)"
echo "============================================================"

cd "$REPO_DIR/third_party/openpi"

# Same torchcodec/FFmpeg-7 fix as victr_openpi_train_job.sh -- this script also
# builds a LeRobotDataset backed by icl-dataset's videos, so it hits the same
# decode-backend issue if the shim isn't set up.
"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

.venv/bin/python3 scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

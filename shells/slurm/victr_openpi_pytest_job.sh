#!/bin/bash
# ============================================================================
# SLURM job script -- runs openpi's (JAX) pytest suite, or a subset of it.
# CPU-only: model-shape/loss-shape sanity tests (e.g. pi0_test.py,
# model_test.py) instantiate real gemma_2b + gemma_300m params and JIT-compile
# a forward+loss pass, which is real compute -- do NOT run this directly on
# the login node (see notes/hpc.md / the login-node-vs-sbatch lesson), always
# go through sbatch.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_openpi_pytest_job.sh \
#       <repo_dir> [pytest args...]
#
# Example (training-time RTC regression check, notes/training_time_rtc.md):
#   sbatch --account=<account> shells/slurm/victr_openpi_pytest_job.sh \
#       /scratch/lim2045/icl_ws/viktr \
#       src/openpi/models/model_test.py src/openpi/models/pi0_test.py -m "not manual"
# ============================================================================
#SBATCH --job-name=victr-openpi-pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-openpi-pytest-%j.out
#SBATCH --error=logs/victr-openpi-pytest-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_openpi_pytest_job.sh <repo_dir> [pytest args...]"}
shift

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Args:      $*"
echo "Started:   $(date)"
echo "============================================================"

cd "$REPO_DIR/third_party/openpi"

# Force JAX onto CPU explicitly -- this partition has no GPU, but being
# explicit avoids any surprise if this script is ever run on a GPU node.
export JAX_PLATFORMS=cpu

.venv/bin/python3 -m pytest -v "$@"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

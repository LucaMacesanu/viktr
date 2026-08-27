#!/bin/bash
# ============================================================================
# One-time resize of the expanded-set VICTR pool's chunk images to 224x224 --
# see scripts/resize_pool_images.py's module docstring for why (pools store
# native 480x640 images but only 224x224 is ever consumed; this was the real
# cause of scripts/precompute_retrieval_context.py's repeated OOMs, not just
# embed_frames/query_frames).
#
# Single process, one task's pool at a time (bounded peak memory = largest
# single pool, ~36GB raw + working copy) -- generous mem/time margin below.
# Must run under SLURM, not the login node (no cgroup memory accounting there).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_resize_pool_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-resize-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-resize-pool-%j.out
#SBATCH --error=logs/victr-resize-pool-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_resize_pool_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/resize_pool_images.py \
    --pool-dir third_party/openpi/assets/victr_icl_pool_expanded \
    --out-dir third_party/openpi/assets/victr_icl_pool_expanded_224 \
    --primary-camera observation.images.zed

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

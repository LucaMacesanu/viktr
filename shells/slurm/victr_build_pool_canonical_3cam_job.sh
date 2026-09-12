#!/bin/bash
# ============================================================================
# Builds victr_icl_pool_canonical_3cam_224/ (scripts/build_pool_canonical_3cam.py):
# extends the existing canonical retrieval pool with all 3 camera views per
# chunk + the redesigned context encoding's native-30-step-window, frequency-
# truncated, FAST-tokenized action field. CPU-only (video decode + DCT/BPE,
# no GPU needed) -- see that script's docstring for the full rationale.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_build_pool_canonical_3cam_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=build-pool-canonical-3cam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/build-pool-canonical-3cam-%j.out
#SBATCH --error=logs/build-pool-canonical-3cam-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_build_pool_canonical_3cam_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/openpi/ffmpeg7_shim:$REPO_DIR/third_party/openpi/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

"$REPO_DIR/third_party/openpi/.venv/bin/python3" scripts/build_pool_canonical_3cam.py

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

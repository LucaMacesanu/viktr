#!/bin/bash
# ============================================================================
# Fits scripts/fit_expanded_fast_tokenizer.py's FAST tokenizer on the 31-task
# expanded set (1,784 episodes) for the yor_icl_ki_expanded_subtask openpi
# config. CPU-only, single process (state/action columns only, no bulk video
# decode needed downstream of dataset construction).
#
# An earlier attempt run directly on the login node (not through SLURM) died
# silently ~1700/1784 episodes in, no traceback -- most likely an OOM kill on
# the shared login node, though it also ran before viktr's own venv had its
# ffmpeg7 shim (was silently falling back to pyav; see shells/
# setup_ffmpeg7_shim.sh's venv_dir param). Running properly resourced under
# SLURM this time regardless.
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_fit_expanded_fast_tokenizer_job.sh <repo_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-fit-expanded-fast-tokenizer-%j.out
#SBATCH --error=logs/victr-fit-expanded-fast-tokenizer-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_fit_expanded_fast_tokenizer_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR" "$REPO_DIR/.venv"
export LD_LIBRARY_PATH="$REPO_DIR/ffmpeg7_shim:$REPO_DIR/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

.venv/bin/python3 scripts/fit_expanded_fast_tokenizer.py \
    --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

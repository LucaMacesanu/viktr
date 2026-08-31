#!/bin/bash
# ============================================================================
# Fits scripts/fit_combined_fast_tokenizer.py's FAST tokenizer jointly over
# icl-dataset's 31-task expanded training set (1,784 episodes, same set
# victr_fit_expanded_fast_tokenizer_job.sh fits alone) and
# adityx23/icl-demo-dataset's kept+successful episodes (269 of 285). CPU-only,
# single process -- state/action columns only, no bulk video decode needed
# (mirrors victr_fit_expanded_fast_tokenizer_job.sh's own sizing: the
# expanded-only fit took 4min for 1,784 episodes on this same 4cpu/32G
# budget, so the ~270 extra episodes here comfortably fit the same shape).
#
# Prerequisite: /scratch/lim2045/icl_ws/icl-demo-dataset must already be
# downloaded (third_party/nyu-finger-robot/download_dataset.py
# adityx23/icl-demo-dataset), converted from v2.1 to v3.0
# (lerobot.scripts.convert_dataset_v21_to_v30, local/--push-to-hub=false),
# and locally patched to add a unified `action` [20] column + fix
# action.lift_cmd's list/scalar type mismatch (both are local-copy-only
# fixes, not pushed back to the Hub -- see scripts/fit_combined_fast_tokenizer.py's
# docstring for why).
#
# Submit:
#   sbatch --account=<account> shells/slurm/victr_fit_combined_fast_tokenizer_job.sh <repo_dir>
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-fit-combined-fast-tokenizer-%j.out
#SBATCH --error=logs/victr-fit-combined-fast-tokenizer-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_fit_combined_fast_tokenizer_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

"$REPO_DIR/shells/setup_ffmpeg7_shim.sh" "$REPO_DIR" "$REPO_DIR/.venv"
export LD_LIBRARY_PATH="$REPO_DIR/ffmpeg7_shim:$REPO_DIR/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"

.venv/bin/python3 scripts/fit_combined_fast_tokenizer.py \
    --icl-demo-root /scratch/lim2045/icl_ws/icl-demo-dataset \
    --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded-plus-demo

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

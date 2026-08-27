#!/bin/bash
# ============================================================================
# Extracts the DINOv2 key_embeddings (+ key_values + chunk metadata) out of every
# task's ChunkDictionary pool pickle under outputs/victr/icl_pool_expanded/, and
# zips them up. Those pool pickles are 1-36GB each (176GB total, mostly per-chunk
# image stacks) -- unpickling even the smallest one (11GB) OOM-kills on the login
# node, so this runs as a CPU job with real memory headroom instead.
#
# Uses the repo's own .venv (plain numpy/pickle, no DINOv2 model load needed --
# the embeddings are already computed and stored in the pool).
#
# Submit (CPU account/QOS, matching the precompute jobs):
#   sbatch --account=torch_pr_564_tandon_advanced --qos=cpu48 \
#     shells/slurm/victr_extract_dino_embeddings_job.sh <repo_dir>
# ============================================================================
#SBATCH --job-name=victr-extract-dino-embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=logs/victr-extract-dino-embeddings-%j.out
#SBATCH --error=logs/victr-extract-dino-embeddings-%j.err

set -euo pipefail

REPO_DIR=${1:?"Usage: sbatch ... victr_extract_dino_embeddings_job.sh <repo_dir>"}
cd "$REPO_DIR"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

.venv/bin/python3 scripts/extract_dino_embeddings.py \
    --pool-dir outputs/victr/icl_pool_expanded \
    --out-dir outputs/victr/dino_embeddings_expanded \
    --zip-path outputs/victr/dino_embeddings_expanded.zip

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

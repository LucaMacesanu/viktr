# VICTR

VICTR (Value-Informed Context Retrieval) is a bi-level in-context-learning
policy for VLAs: retrieve a demonstration context chunk using a fused
visual-similarity + value-alignment score, then feed that chunk into a
context-conditioned VLA policy (IC-VLA) that predicts the next action
chunk — no fine-tuning at deployment. It extends RICL (vision-only
retrieval) with an in-context value signal.

Built on lerobot's native PyTorch `pi05` (π0.5) as the IC-VLA base, and
Robometer as the value estimator. See `notes/progress.md` for the full
design-decision log, milestones, and what's built so far; `notes/hpc.md` for
running this on a cluster.

## Layout

- `src/viktr/retrieval/` — chunk dictionary construction, DINOv2 embeddings,
  the three retrieval metrics (vision / value / vision+value fusion).
- `src/viktr/value/` — `ValueEstimator` interface and the Robometer wrapper.
- `src/viktr/policy/` — `VictrPolicy`, a context-chunk-conditioned subclass
  of lerobot's `pi05`.
- `src/viktr/data/` — `lerobot/libero` dataset adapter.
- `scripts/` — Python entrypoints (smoke tests, LIBERO eval harness).
- `shells/` — `.sh` wrappers around `scripts/*.py` for local/HPC runs, plus
  SLURM job templates under `shells/slurm/`. See `shells/README.md`.
- `third_party/` — `lerobot` (editable submodule, modified for context
  conditioning) and `ricl_openpi` (reference only).

## Setup

```bash
git clone --recurse-submodules <repo-url> viktr   # or: git submodule update --init --recursive
uv sync
```

Requires Python 3.12 (`.python-version`) and a CUDA GPU for anything beyond
the cheapest smoke tests (`pi05` is a 2B+300M-param model).

## Running things

```bash
shells/smoke_chunk_retrieval.sh     # retrieval sanity check, no GPU-heavy models
shells/smoke_victr_policy.sh        # loads real pi05_base weights, checks context conditioning
shells/eval_libero.sh --suite libero_object --task-ids 0 --n-episodes 2 \
    --retrieval-metrics vision,none
```

See `shells/README.md` for the full wrapper list, and `notes/hpc.md` for
SLURM submission.

## Status

MVP retrieval + context-conditioned policy + LIBERO closed-loop eval harness
are built and verified end-to-end (zero-shot, not yet fine-tuned — see
`notes/progress.md`'s "What's next" for the training loop that's still
needed before eval success rates are meaningful).

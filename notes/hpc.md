# Moving `viktr` to HPC

Task 7 in the build plan. This documents what's already true, what to set up
on a new cluster, and how the `shells/` wrappers / `shells/slurm/` templates
fit together. Written without a specific target cluster in mind (none was
specified) — SLURM directives are placeholders to fill in per-allocation.

## What's already relocatable

- No hardcoded absolute paths in `src/viktr` or `scripts/` (checked via
  `grep -rn "/home/lairlab"` — zero hits). This was a standing design
  constraint from the start (design decision #4 in `progress.md`), not
  something added for this task.
- The whole env is `uv`-managed (`pyproject.toml` + `uv.lock`), with
  `third_party/lerobot` and `third_party/ricl_openpi` as pinned git
  submodules rather than loose local checkouts — `uv sync` reproduces the
  exact env anywhere.
- `shells/*.sh` wrappers resolve their own location (`$(dirname
  "${BASH_SOURCE[0]}")`) rather than assuming a cwd, so they work identically
  whether launched interactively, from `sbatch`, or from a different checkout
  path entirely.

## What to do on a fresh cluster checkout

1. **Clone with submodules**, or the `third_party/` dirs will be empty:
   ```bash
   git clone --recurse-submodules <repo-url> viktr
   # or, if already cloned flat:
   git submodule update --init --recursive
   ```
2. **Python 3.12 + uv.** Load whatever module gives Python 3.12
   (`.python-version` pins this), then install `uv` per its own instructions
   (pipx / standalone installer — no cluster-specific step known). Then:
   ```bash
   uv sync
   ```
   This builds the venv from `uv.lock`, including the editable
   `third_party/lerobot` path dependency — no separate `pip install -e` step.
3. **Redirect the Hugging Face cache off `$HOME`.** `HF_HOME` (default
   `~/.cache/huggingface`) is where `pi05_base` weights, `Robometer-4B`
   weights, and the `lerobot/libero` dataset all land. On this dev machine
   that cache is currently **155GB** — almost certainly larger than a
   cluster home-directory quota. `shells/_common.sh` exports `HF_HOME` (as a
   passthrough — respects whatever's already set) and the `slurm/` templates
   default it to `/scratch/$USER/hf_cache`. Adjust that path to match your
   cluster's actual scratch/project filesystem convention before submitting.
4. **LIBERO's asset cache is NOT relocatable.** `hf_libero`'s
   `download_utils.py` hardcodes `~/.cache/libero/assets` — no env var
   override exists upstream. It's small (~400MB on this machine), so it
   should fit under most home quotas even when everything else is redirected
   to scratch; just be aware it's the one cache that stays on `$HOME`
   regardless of `HF_HOME`.
5. **GPU.** Verified working on an RTX PRO 6000 Blackwell (98GB VRAM) with
   ~21GB in use by other processes at the time; VictrPolicy (PaliGemma 2B +
   Gemma-300M action expert) plus Robometer-4B loaded and ran comfortably in
   that headroom. No lower-bound VRAM figure has been measured — a single
   forward pass plus a rollout hasn't been profiled for peak memory, so
   don't assume a specific `--gres=gpu:1` flavor is sufficient without
   checking your cluster's smallest GPU option first.
6. **`logs/` and `outputs/`** are created on demand by the `slurm/`
   templates (`mkdir -p logs outputs`) and are already `.gitignore`d
   (anchored `/outputs/`, see `.gitignore` — not `data/`/`checkpoints/`,
   which are also ignored the same way).

## Running things

Interactively (login node or an interactive allocation with a GPU):
```bash
shells/eval_libero.sh --suite libero_object --task-ids 0 --n-episodes 2 \
    --retrieval-metrics vision,none
```

Via SLURM, generic wrapper launcher:
```bash
sbatch shells/slurm/run.slurm.sh shells/eval_libero.sh \
    --suite libero_object --task-ids 0,1,2 --n-episodes 10 \
    --retrieval-metrics vision,value,vision+value,none \
    --output outputs/eval_libero_object.json
```

Via SLURM, one task-id per array index (matches the "scaled eval run" noted
as not-yet-done in `progress.md`):
```bash
sbatch --array=0-9 shells/slurm/eval_libero_array.slurm.sh libero_object
```

See `shells/README.md` for the full wrapper list and what each one takes.

## Explicitly not covered here

- **Multi-GPU / multi-node.** Nothing in this codebase has been tested
  beyond a single GPU, single process. The `srun` calls in the SLURM
  templates are single-task; distributed training/eval would need real
  changes to `eval_libero_victr.py` and the (not-yet-built) training script,
  not just SLURM directives.
- **Cluster-specific SLURM config** (partition names, account/QOS strings,
  module names for CUDA/Python, whether `/scratch/$USER` is the right
  convention). Left as `TODO_PARTITION` / `TODO_ACCOUNT` placeholders in
  `shells/slurm/*.slurm.sh` and a default-guessed `HF_HOME` path — fill in
  once the actual target cluster is known.
- **Container/Apptainer packaging.** Not attempted; `uv sync` on the compute
  node is the only setup path documented here. If the target cluster
  requires containers instead of bare-metal module loads, that's a separate
  piece of work.

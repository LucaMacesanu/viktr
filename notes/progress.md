# VICTR — design decisions, milestones, and progress

Living doc. Update this alongside any significant implementation decision or
completed milestone — don't let it drift out of date. `notes/explanation.md`
and `notes/vktr.pdf` are the original problem statement / paper draft and stay
as-is; this file tracks what we actually decided and built in response.

## Design decisions

1. **Build fresh, not forked from RICL.** `third_party/ricl_openpi` (vendored
   as a submodule, `ricl-vla/ricl_openpi`) is reference-only — we study and
   selectively port logic (its DINOv2 embedding convention, its offline
   preprocessing pattern) rather than building on top of its codebase.
2. **Value estimation uses Robometer, not a trained IC-VFE**, for the MVP.
   Robometer ships with `lerobot>=0.6` (`lerobot[robometer]` extra). This
   substitutes for the paper's Gemma3-270M IC-VFE, which isn't available as
   open-source code. A custom IC-VFE remains a possible future upgrade behind
   the same `ValueEstimator` interface (`src/viktr/value/base.py`).
3. **Vision-similarity retrieval mirrors RICL's actual mechanism**: DINOv2
   ViT-B/14 (`facebookresearch/dinov2`, `dinov2_vitb14`), CLS token embedding,
   L2 distance. Note RICL itself does *per-frame* retrieval with block-diagonal
   attention over independently-retrieved frames — architecturally different
   from VICTR's *chunk-based* retrieval (paper Eq. 6). We use RICL's embedding
   convention but rebuild the chunking/dictionary logic from the paper, not
   ported from RICL's model code.
4. **`uv` for env/dependency management**; python entrypoints in `scripts/`;
   high-level orchestration in `shells/` (`.sh` files, not yet created). The
   codebase is expected to move to HPC later, so paths/config stay relocatable
   (no hardcoded absolute paths in library code).
5. **IC-VLA's base model is π0.5 via lerobot's native PyTorch implementation,
   not openpi/JAX.** Real π0.5 weights (`lerobot/pi05_base`) and a full
   training/inference stack already exist in `lerobot`. This was a pivot from
   the original plan (build fresh on top of `openpi`) after confirming real
   openpi lacks Gemma3/π0.5-FAST. `lerobot` is vendored as an **editable git
   submodule** (`third_party/lerobot`, pinned `v0.6.1`) rather than a plain
   PyPI dependency, because task 5 requires modifying `pi05`'s model internals
   directly (context-chunk conditioning), not just importing it as a black box.
   `openpi` stays reference-only for retrieval/context-injection wiring
   patterns.
6. **Consequence of #5 — single PyTorch stack.** With both IC-VLA (`pi05`) and
   Robometer on PyTorch, there's no JAX↔PyTorch process-isolation problem to
   design around. Value scoring and policy inference can run in one process,
   which simplifies rollout scripts considerably versus the openpi-based plan.
7. **Validate on LIBERO-100 in sim before the real bimanual dataset.** lerobot
   0.6 has native LIBERO/LIBERO-plus benchmark support to build on
   (`docs/source/libero.mdx` in the vendored submodule).
8. **MVP demo data source: `lerobot/libero`** (HF dataset, 4 standard
   reproducibility suites, 40 tasks, 1.9GB, v3.0 format) rather than the full
   raw LIBERO-90+10 benchmark (100 tasks, matches the paper's exact benchmark)
   or the larger `HuggingFaceVLA/libero` PNG variant (69.9GB). Chosen to keep
   iteration fast while building retrieval/value logic; swapping to the full
   100-task benchmark or the real bimanual dataset later is a data-source
   swap, not a logic change — same `load_episode_arrays` / chunking / retrieval
   code path either way. Not separately confirmed with the user; a pragmatic
   call made to keep forward momentum, worth revisiting before final eval.
9. **Never commit without being asked** was the standing rule during early
   implementation; the user has since asked for regular commits going forward
   (2026-08-15), so from this point on we commit as each meaningful chunk of
   work lands rather than batching everything into one commit.

## Repo layout (current)

```
viktr/
  pyproject.toml / uv.lock        # uv-managed env, PyTorch/lerobot stack only
  third_party/
    ricl_openpi/                  # submodule, reference only
    lerobot/                      # submodule, editable, v0.6.1 — modified for context conditioning (task 5)
  src/viktr/
    retrieval/
      embeddings.py                # DINOv2 frame embeddings
      chunk_dictionary.py          # Chunk / ChunkDictionary, 50%-stride splitting (paper Eq. 6)
      metrics.py                   # f_retrieve: vision / value / vision+value dispatch
      fusion.py                    # ValueFusionMLP g_psi(d_vis, d_val) (Sec III-G)
    value/
      base.py                      # ValueEstimator interface
      robometer.py                 # MVP estimator, wraps lerobot.rewards.robometer
      annotate.py                  # offline per-episode value annotation
    data/
      libero.py                    # lerobot/libero HF dataset -> per-episode array adapter
    policy/                        # empty so far — task 5
  scripts/
    smoke_test_chunk_retrieval.py
    smoke_test_robometer.py
    smoke_test_fused_retrieval.py
  shells/                          # not yet created — task 7
  notes/
    explanation.md, vktr.pdf       # original problem statement / paper draft
    progress.md                    # this file
```

## Milestones / progress log

- **2026-08-14 — Task 1, environment setup.** `uv` project created; `lerobot`
  and `ricl_openpi` vendored as git submodules; `lerobot[libero,robometer]`
  wired as an editable path dependency. Confirmed `pi05_base` and
  `Robometer-4B` both load from the HF cache inside this env.
- **2026-08-14 — Task 2, chunk dictionary + vision retrieval.**
  `retrieval/embeddings.py`, `retrieval/chunk_dictionary.py` written.
  Verified against real `lerobot/libero` data
  (`scripts/smoke_test_chunk_retrieval.py`): chunk dictionary size 46 from 3
  episodes of one task; self-retrieval sanity check passes (score ≈ 0 for the
  query chunk itself; meaningful nearest neighbors across episodes).
- **2026-08-14 — Task 3, Robometer value wrapper.** `value/base.py`,
  `value/robometer.py` written; calling convention (K=4 subsampled window,
  `TransitionKey` import path) worked out against the real vendored source.
  Verified against one real cached episode
  (`scripts/smoke_test_robometer.py`): progress trends 0.25 → 0.44 → 0.57
  over 214 frames — monotonic but noisy (expected, zero-shot, no
  task-specific tuning).
- **2026-08-14 — Task 4, fused retrieval.** `retrieval/metrics.py` rewritten
  to dispatch across all three modes; `retrieval/fusion.py` (`ValueFusionMLP`)
  and `value/annotate.py` added. Verified with a synthetic linear-progress
  estimator (`scripts/smoke_test_fused_retrieval.py`) so the wiring test
  doesn't depend on loading the (slow) real Robometer model.
- **2026-08-15 — Repo committed to git** for the first time; this progress
  doc added. Task 5 (context-conditioned IC-VLA) not yet started.

## Known bugs fixed along the way (for context, not action items)

- `hf-libero`'s importable module name is `libero`, not `hf_libero`.
- Dropped a redundant `third_party/LIBERO` submodule once we confirmed
  lerobot's `[libero]` extra already pulls in `hf-libero` as a pip dependency.
- `episodes_for_task` (`src/viktr/data/libero.py`) originally assumed a
  `"tasks"` column on `meta.episodes`; that column doesn't exist. Task
  membership only lives on the per-frame `task_index` column, read via
  `download_videos=False` to avoid an expensive full video download.
- `load_episode_arrays` originally indexed with the *global*
  `dataset_from_index`/`dataset_to_index` from `meta.episodes`, but
  `LeRobotDataset(..., episodes=[...])` reindexes to local contiguous indices
  over just the selected episodes — fixed by computing offsets from a
  cumulative sum of per-episode lengths in sorted order.
- `lerobot.types.TransitionKey` (used by an external reference
  implementation written against a newer lerobot) doesn't exist in our pinned
  v0.6.1; the real path is `lerobot.lerobot_types.TransitionKey`.

## What's next

- **Task 5 (not started, architecturally the hardest piece)**: extend
  `third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py` so
  `PI05Policy` accepts `k` retrieved `Chunk`s as in-context conditioning —
  neighbor blocks ordered nearest-to-farthest, each with a learned
  neighbor-rank embedding, prepended to the query (paper Sec III-E). New file:
  `src/viktr/policy/pi05_context.py`, plus `configuration_viktr.py` to wire
  the retrieval-metric choice through config. First milestone: one LIBERO-100
  forward pass + rollout with vision-only retrieval, before adding value.
- **Task 6**: LIBERO-100 rollout/eval across all three retrieval-metric
  variants (vision / value / vision+value) plus a no-retrieval baseline,
  10 rollouts/task on a small task subset, single PyTorch process.
- **Task 7**: `shells/` — thin `.sh` wrappers per script for local runs, kept
  relocatable for the eventual HPC/SLURM move.
- **Deferred, post-MVP**: port to the real bimanual dataset; compare against
  ICRT (`projects/icl_baselines/icrt`) and RECAP/TOPReward/SARM
  (`projects/value-estimation`) baselines.

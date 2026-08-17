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

10. **Task 5 built as a subclass of pi05, not an edit to the vendored submodule.**
    `src/viktr/policy/pi05_context.py` (`VictrPI05`, `VictrPolicy`) subclasses
    `PI05Pytorch`/`PI05Policy` and composes on top of `embed_prefix` rather than
    editing `third_party/lerobot/.../modeling_pi05.py` in place. Reasons: (a)
    keeps the submodule a clean, diffable upstream checkout that's easy to bump
    later; (b) `embed_prefix` already does everything a context block needs
    (SigLIP image embedding + tokenized text, concatenated with block-boundary
    attention flags) — reusing it per chunk avoids re-deriving that logic.
    Context blocks are ordered farthest-to-nearest and prepended to the query's
    own prefix, each starting a fresh causal block (`att_masks[:, 0] = True`)
    and tagged with a learned `neighbor_rank_embedding` added to every token in
    that block. Net effect under pi05's existing block-causal mask
    (`make_att_2d_masks`): each chunk sees itself + every chunk before it, the
    query prefix sees every chunk, the query's action-expert suffix sees
    everything, and no chunk ever sees the query (see the module docstring for
    the full argument). `forward`/`sample_actions` are full overrides (not
    monkey-patches) that duplicate the small amount of surrounding plumbing
    from the base class so there's one linear code path to read.
11. **Retrieved chunks are described to the model as text, not just images.**
    Each chunk block is `K` subsampled frames (`linspace`, matching Robometer's
    convention) embedded through the same SigLIP tower as the query's own
    images, *plus* a "Task: …, State: …; Action: …" text summary using pi05's
    own 256-bin digitization convention (`Pi05PrepareStateTokenizerProcessorStep`)
    applied to the chunk's first-frame proprio and a subsampled action
    sequence. This lets the model read what the retrieved chunk *did*
    (its actions), not just what it *looked like* — closer to genuine
    in-context conditioning than image similarity alone. Known gap: chunk
    proprio/actions are clipped to `[-1, 1]` assuming they're already
    normalized the way the base model's own `observation.state` is; nothing
    in the current wiring actually normalizes them first (they come straight
    from `viktr.data.libero` in raw dataset units). Fine for a forward-pass
    smoke test; needs real normalization stats wired through before training.
12. **Task 6's eval harness drives `LiberoEnv` directly, unvectorized, instead
    of going through `lerobot-eval`/`gym.vector`.** `lerobot-eval` resolves
    policies by a hardcoded type-name switch that doesn't know about `"victr"`,
    and its `rollout()` calls `policy.select_action(observation)` with no hook
    for injecting retrieved context chunks per step. Since we need to retrieve
    a fresh chunk at every action-chunk refill (not just pick an action), we
    own the loop: `scripts/eval_libero_victr.py` builds one raw
    `lerobot.envs.libero.LiberoEnv` per task (bypassing `create_libero_envs`'s
    `gym.vector` wrapping), reuses lerobot's own `LiberoProcessorStep` /
    `make_pi05_pre_post_processors` for observation formatting, and calls
    `VictrPolicy.predict_action_chunk(..., context_chunks=...)` directly.
    One `VictrPolicy` (loaded from `lerobot/pi05_base`), one DINOv2 embedder,
    and (if needed) one `RobometerValueEstimator` are constructed once and
    reused across every task/metric in a run, since reloading a 2B+300M-param
    model per variant would dominate runtime.

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
    policy/
      configuration_victr.py       # VictrConfig(PI05Config): num_context_chunks, context_chunk_size, etc.
      pi05_context.py              # VictrPI05(PI05Pytorch) / VictrPolicy(PI05Policy): context-chunk conditioning
  scripts/
    smoke_test_chunk_retrieval.py
    smoke_test_robometer.py
    smoke_test_fused_retrieval.py
    smoke_test_victr_policy.py
    eval_libero_victr.py          # task 6: closed-loop LIBERO rollout/eval harness
  shells/                          # task 7: .sh wrappers per script + SLURM templates
    _common.sh, smoke_*.sh, eval_libero.sh
    slurm/run.slurm.sh, slurm/eval_libero_array.slurm.sh
  notes/
    explanation.md, vktr.pdf       # original problem statement / paper draft
    progress.md                    # this file
    hpc.md                        # task 7: HPC migration writeup
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
  doc added.
- **2026-08-15 — Task 5, context-conditioned IC-VLA.** `configuration_victr.py`
  and `pi05_context.py` written (see design decisions #10, #11 above).
  Verified end-to-end (`scripts/smoke_test_victr_policy.py`) against real
  `lerobot/pi05_base` weights and real retrieved `lerobot/libero` data: loaded
  the pretrained checkpoint with exactly one missing key
  (`neighbor_rank_embedding.weight`, expected — it's new), retrieved the
  vision-nearest chunk for a query frame, ran `predict_action_chunk` with that
  chunk as context, got a correctly-shaped `(1, 50, 7)` action chunk, and
  confirmed the context-conditioned output differs from a no-context call on
  the same observation (context is actually influencing the prediction, not
  silently dropped). Not yet validated: actual rollout quality (both the base
  model and the new context-conditioning weights are untrained/zero-shot on
  LIBERO right now — that's expected and is what task 6's training loop is
  for), and the `forward()` (training-loss) path's context branch specifically
  (it reuses the same `embed_prefix_with_context` machinery `sample_actions`
  uses, so risk is low, but it hasn't been smoke-tested on its own).
- **2026-08-17 — Task 6, LIBERO rollout/eval harness.** `scripts/eval_libero_victr.py`
  written (see design decision #12 above): builds real LIBERO sim envs, a
  per-task chunk pool from `lerobot/libero` demos, and drives closed-loop
  rollouts calling `VictrPolicy.predict_action_chunk` with freshly retrieved
  context at every action-chunk refill. All four variants — `vision`,
  `value`, `vision+value`, and a `none` (no-retrieval) baseline — verified
  running end-to-end on a real `libero_object` task-0 sim rollout without
  crashing (short `--max-steps` smoke runs; 0% success is expected at this
  horizon/without LIBERO finetuning, not a bug). Two real bugs found and
  fixed along the way:
  - `viktr/retrieval/metrics.py`'s `fused_retrieve` didn't move the
    `d_vis`/`d_val` tensors onto the fusion module's device — worked in the
    task-4 smoke test only because that fusion module happened to stay on
    CPU; crashed here once `fusion` was moved to CUDA. Now moves both onto
    `next(fusion.parameters()).device` before calling it.
  - LIBERO's raw (unvectorized) `gym.Env.reset()`/`step()` return unbatched
    leaf arrays, including inside the nested `robot_state` dict — but
    `LiberoProcessorStep._quat2axisangle` requires its input already batched
    and raises instead of adding the batch dim itself (unlike the final
    concatenated state, which does get a `dim()==1: unsqueeze(0)` fallback).
    Fixed by adding a local `_add_batch_dim` helper in the eval script that
    recursively batches every leaf before handing the observation to
    `preprocess_observation`/`LiberoProcessorStep`.
  Not yet done: an actual scaled run (10 episodes/task across several tasks
  and all four variants) to compare success rates — the harness supports it
  via `--n-episodes`/`--task-ids`/`--retrieval-metrics`, but a real run is
  slow (2B+300M-param model, up to ~280 sim steps/episode for libero_object)
  and neither `pi05_base` nor the new context-conditioning weights are
  LIBERO-finetuned yet, so success rates would be near zero regardless of
  retrieval quality — that requires the training loop this harness doesn't
  build (fine-tuning `VictrPolicy` on `lerobot/libero`, not just running it
  zero-shot). A harmless `transformers` warning ("Kwargs passed to
  `processor.__call__`...") prints repeatedly during tokenization; traced to
  vendored lerobot processor code, not our code, and doesn't affect output —
  left alone.
- **2026-08-17 — Task 7, `shells/` + HPC readiness.** Added one `.sh` wrapper
  per `scripts/*.py` entrypoint (`shells/smoke_*.sh`, `shells/eval_libero.sh`),
  all sourcing a shared `shells/_common.sh` that resolves the repo root from
  the wrapper's own path (not the caller's cwd) and cds there before `uv run`
  — verified this actually matters by running `shells/smoke_chunk_retrieval.sh`
  and `shells/eval_libero.sh` from `/tmp`, both worked. Added
  `shells/slurm/run.slurm.sh` (generic launcher: `sbatch
  shells/slurm/run.slurm.sh <wrapper> [args...]`) and
  `shells/slurm/eval_libero_array.slurm.sh` (one LIBERO task id per array
  index, for the still-pending scaled eval run). Cluster-specific `#SBATCH`
  fields (`partition`, `account`) are left as `TODO_*` placeholders since no
  target cluster was specified. Wrote `notes/hpc.md`: documents that
  `src`/`scripts` already have zero hardcoded absolute paths (checked via
  grep), how to clone-with-submodules and `uv sync` on a fresh checkout, and
  the one real cache-relocation gotcha found while writing it — `HF_HOME`
  (currently 155GB on this dev machine: pi05/Robometer weights +
  `lerobot/libero` dataset) is env-overridable and should point at
  scratch/project storage on a cluster, but LIBERO's own asset cache
  (~400MB) is hardcoded to `~/.cache/libero/assets` by `hf_libero` with no
  env var override — that one stays on `$HOME` regardless. Also noted no
  VRAM lower bound has been measured (only tested on a 98GB card with ~21GB
  already in use by other processes) and multi-GPU/multi-node is untested
  and unsupported by the current single-process eval script.
  `shells/README.md` documents the wrapper/template list. Root `README.md`
  (previously empty) now gives a repo overview pointing at `notes/progress.md`
  and `notes/hpc.md`.

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

- **Training loop (not started, not yet a numbered task)**: fine-tune
  `VictrPolicy` on `lerobot/libero` (the `forward()` / loss path exists from
  task 5 but has never been run in a training loop) — needed before a scaled
  `eval_libero_victr.py` run would show anything but near-zero success rates.
  This is also where the `vision+value` fusion MLP (`ValueFusionMLP`,
  currently random-initialized) would actually get trained (paper Sec III-G:
  jointly with the IC-VLA via soft top-k).
- **Deferred, post-MVP**: port to the real bimanual dataset; compare against
  ICRT (`projects/icl_baselines/icrt`) and RECAP/TOPReward/SARM
  (`projects/value-estimation`) baselines.

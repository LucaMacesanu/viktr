# RICL action interpolation (baseline port)

Source: `third_party/ricl_openpi/src/openpi/models/pi0_fast_ricl.py`
(`Pi0FASTRicl.interpolate_actions`, `use_action_interpolation`/`lamda`) and
`src/openpi/policies/policy.py` (`RiclPolicy.retrieve`'s `exp_lamda_distances`).

Not to be confused with `notes/training_time_rtc.md`'s "action interpolation" --
that's real-time chunking (smoothing across re-plan boundaries during closed-loop
execution). This is a completely different mechanism: interpolating between the
VLA's own predicted action and the top-1 retrieved neighbor's actual action,
weighted by retrieval similarity. Implemented here as a baseline to compare
against, not as a VICTR design choice -- see "What this is for" below.

## 1. What RICL actually does

At every FAST decode step (autoregressive, one action token at a time), RICL
blends the model's own predicted token distribution with the top-1 retrieved
neighbor's ground-truth action token, weighted by how visually similar that
neighbor is to the current query:

```
exp_lamda_distance = exp(-lamda * normalized_vision_distance(query, nearest_neighbor))
new_probs = exp_lamda_distance * onehot(neighbor_action_token) + (1 - exp_lamda_distance) * softmax(model_logits)
```

Close visual match -> weight near 1 -> the model is pulled toward literally copying
the neighbor's action. Far match -> weight near 0 -> the model's own prediction
dominates. Applied identically during training (`compute_loss`) and inference
(`sample_actions`'s KV-cache decode loop), so it's a real intervention on what
gets generated, not just a training-time regularizer.

This only works cleanly in RICL's own code because their tokenizer
(`FASTTokenizerRicl`) pads the prefix and postfix (action) token spans to a fixed
50/50 split of `max_token_len` -- so "the neighbor's postfix tokens" and "the
query's own postfix tokens" always occupy the identical fixed-size array slice,
making index-for-index blending well-defined by construction.

## 2. Why it doesn't port onto our own tokenizer as-is

`Pi0FastVictrConfig`/`yor_icl_fast_victr_vision_expanded` uses the plain
`openpi.models.tokenizer.FASTTokenizer`, not RICL's Ricl variant: prefix and
postfix are concatenated at their own natural (data-dependent) lengths and only
the *whole* sequence is padded at the end. The postfix's actual start offset
therefore varies per example -- there's no fixed slice to blend against.

Resolved (2026-08-28, user decision -- see AskUserQuestion in this session)
without introducing a new tokenizer convention:

- **Query side**: at both training (`compute_loss`) and inference
  (`sample_actions`) time, the postfix start is either looked up dynamically via
  `token_loss_mask` (training: `argmax` per example, batched) or known for free
  (inference: decode always starts exactly at the postfix boundary, so decode
  step index `i` *is* postfix position `i`, no lookup needed). A fixed-width
  window (`max_action_tokens - 1`) is then extracted/written back via
  `jax.lax.dynamic_slice`/`dynamic_update_slice` -- see
  `Pi0FastVictr._dynamic_window`, `interpolate_actions`, `_blend_decode_step` in
  `third_party/openpi/src/openpi/models/pi0_fast_victr.py`.
- **Neighbor side**: the nearest chunk's actions are tokenized with
  `FASTTokenizer.tokenize_action_only` (already existed, previously only used by
  Knowledge Insulation's auxiliary loss -- `openpi.policies.yor_ki.KiTargetInputs`)
  -- `[bos, "Action: ", <FAST tokens>, "|", eos]`, padded to a fixed
  `max_action_tokens`. Dropping the leading bos gives exactly the same
  `["Action: ", <FAST tokens>, "|", eos, 0-pad...]` layout the query's own postfix
  window has, so the two align token-for-token within that fixed window without
  needing RICL's own 50/50 split.

The alternative considered and rejected (for now): actually adopt RICL's fixed
50/50 prefix/postfix tokenizer convention. More literally faithful, but would
have required a new tokenizer wrapper and re-validating `max_token_len` budgets
for our action space -- more invasive for the same algorithmic result.

## 3. A second gap: chunk length vs. action_horizon

RICL retrieves a native `action_horizon`-length action window directly from an
unchunked per-frame index. Our retrieval pool
(`assets/victr_icl_pool_expanded_224/*.pkl`) is built from fixed-length chunks
(`context_chunk_size`, default 10 frames) via
`viktr.retrieval.chunk_dictionary`/`scripts/build_retrieval_pool.py` -- a
pool-build-time property, not something a training config can change without
rebuilding the pool (viktr's own venv, DINOv2 re-embedding all 1,784 episodes).

Rather than requiring a new pool built at `context_chunk_size == action_horizon`,
`RetrievalContextInputs.__call__` resamples the nearest chunk's
`(context_chunk_size, action_dim)` actions to `(action_horizon, action_dim)` via
per-dimension linear interpolation over the chunk's own time axis before
FAST-tokenizing. This is a documented approximation (a real action_horizon-length
window would show what the neighbor did over that literal span; this shows a
time-stretched version of a shorter span) -- acceptable for a first baseline pass,
worth revisiting (rebuild the pool at `context_chunk_size=action_horizon`) if the
approximation turns out to matter empirically.

## 4. Normalization constant

RICL normalizes by a single global max-distance constant computed once across
its entire DROID+collected-demos corpus (`assets/max_distance.json`). We instead
normalize per-task, by the max pairwise DINOv2 distance within that task's own
pool (`yor_retrieval._pool_max_vision_distance`, cached like the pool itself) --
icl-dataset's per-task pools vary widely in visual variety, so a single global
constant would over- or under-normalize depending on the task. Deviation from
RICL, not expected to change the qualitative behavior.

## 5. What's implemented vs. not

**Implemented**: `Pi0FastVictrConfig.use_action_interpolation`/`lamda`/
`max_action_tokens`; `Pi0FastVictr.interpolate_actions`/`_blend_decode_step`
wired into both `compute_loss` and `sample_actions`;
`RetrievalContextInputs.use_action_interpolation` computing
`exp_lamda_distance`/`nearest_action_tokens`/`nearest_action_tokens_mask` at
data-loading time; new `Observation` fields (`model.py`) threaded through
`preprocess_observation`. New training config:
`yor_icl_fast_victr_vision_interp_expanded` (same episodes/tokenizer/pool as
`yor_icl_fast_victr_vision_expanded`, `use_action_interpolation=True`,
`lamda=10.0`, `max_action_tokens=224`, live retrieval only --
`precomputed_context_dir` must stay unset).

**Not yet done**: an actual training run that gets past startup (see job history
in `notes/training_runs.md` -- 16520341 crashed 8min in on a real bug, fixed
2026-08-28, see below) and any eval/comparison against
`yor_icl_fast_victr_vision_expanded` without interpolation. `max_action_tokens=224`
is carried over from `yor_icl_fast_victr_vision_expanded`'s own config comment
(measured/estimated ~191-token ceiling for this action space), not independently
re-measured for the action-only (`tokenize_action_only`) path specifically --
worth checking against real data before trusting it doesn't truncate.

**Bug found and fixed (2026-08-28), job 16520341:** `RetrievalContextInputs`
looked up the DINOv2 pool file with `task = str(data["prompt"])` -- but by the
time this transform runs, `data_loader.py`'s `PromptFromLeRobotTask` has already
replaced `data["prompt"]` with the *overridden* display prompt
(`YOR_EXPANDED_TASK_PROMPT_OVERRIDES`), while pool `.pkl` files are named by
`task_slug()` of the *raw* lerobot task string. All 8 override entries map to a
different string than their key, so every one of them slug-mismatched its pool
file (crash: looked for a pool file that doesn't exist). This is the *only*
config that hits this code path live -- every other retrieval arm sets
`precomputed_context_dir`, which takes an O(1) episode-index array-lookup branch
instead and never calls the task-string pool lookup at all; the serve-only twin
(`yor_icl_fast_victr_vision_expanded_serve`) also has `precomputed_context_dir=
None` and shares the same latent bug, just never yet exercised with an
overridden task's prompt. Fixed by threading the inverse of
`task_prompt_overrides` into `RetrievalContextInputs.prompt_to_task_overrides`
(`config.py`'s `LeRobotYorVictrDataConfig.create()`), so the pool lookup maps
the display prompt back to the raw task string first. Verified against the real
pool directory: all 8 overridden tasks now resolve to their existing `.pkl`
files.

## What this is for

This is a baseline to compare VICTR against, not a mechanism VICTR's own design
adopts -- the user asked to reproduce RICL's actual method (arXiv:2508.02062) on
icl-dataset to have a fair comparison point, distinct from VICTR's own
chunk-based retrieval (paper Eq. 6) and value-fusion retrieval, neither of which
use action interpolation.

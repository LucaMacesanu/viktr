# Training-Time RTC (Real-Time Chunking)

Source: `third_party/real-time-chunking-kinetix` (git submodule-style vendored repo,
reference implementation for two papers by the same group):

- **"Real-Time Execution of Action Chunking Flow Policies"** (arXiv 2506.07339) — the
  original, *inference-time* RTC.
- **"Training-Time Action Conditioning for Efficient Real-Time Chunking"**
  (arXiv 2512.05964) — what this note and the `Pi0Config.simulated_delay` /
  `Pi0.compute_loss` code actually implement.

Both operate on flow-matching action-chunking policies (openpi's `Pi0`/pi05 is exactly
this). This is an implemented-and-running feature, not a pure design doc — see §4 for
what's actually in the code and what's deferred.

## 1. The problem RTC solves

A chunking policy re-plans every `H` steps by producing a fresh action chunk. But
inference takes real wall-clock time, so by the time the new chunk is ready, the robot
has already started executing the *previous* chunk's next few actions — there's an
"inference delay" `d` between "start planning" and "chunk is usable." A new chunk that
ignores this and gets blended in naively produces a jerky discontinuity at every
re-plan boundary, because the model was never trained to be consistent with anything
already committed.

## 2. Two ways to fix it

**Inference-time RTC** (2506.07339, `third_party/real-time-chunking-kinetix/src/
model.py:FlowPolicy.realtime_action`/`pinv_corrected_velocity`): no retraining. During
the flow-matching ODE integration, compute a pseudo-inverse-corrected velocity via
`jax.vjp` that pulls the sample toward the previous chunk's committed prefix, weighted
by a decaying "prefix attention" schedule (`get_prefix_weights`) and a guidance-weight
schedule derived from flow-matching SDE theory. Works with an unmodified model, but
costs a VJP every integration step (or best-of-N rejection sampling in the repo's
`bid_action` variant) and is only an approximate soft correction.

**Training-time RTC** (2512.05964, `model.py:FlowPolicy.loss`, gated by
`simulated_delay`): bake the "part of this chunk is already fixed" scenario into
training instead. Per training example, sample a delay `d` from `[0, simulated_delay)`,
weighted toward small values (`w = exp(arange(delay)[::-1])`, normalized — small delays
much more likely than large ones, matching what actually happens at inference). Pin the
first `d` action-chunk positions' local flow time to "fully resolved" (`x_t = action`)
instead of the globally-sampled time, and mask the loss to only the remaining noisy
suffix positions. This trains the velocity field to be a correct **conditional
denoiser**: "given the first `d` actions of this chunk are already fixed to
ground-truth-like values, predict the flow for the rest." At inference you then just
directly clamp the prefix to the previous chunk's committed actions and do one plain
forward pass — no VJP, no rejection sampling, exact rather than approximate, and much
cheaper. This is strictly the more attractive of the two for a policy we're training
from scratch (or fine-tuning) ourselves, which is why it's the one implemented here.

## 3. The architecture gap in pi05 (and how it's closed)

Kinetix's toy `FlowPolicy` conditions each action-chunk position on its own time value
independently (`model.py:140-158`: `time` has shape `(b, chunk)`, gets its own
per-position embedding, modulates each chunk position independently via per-token
adaLN in an MLP-Mixer). openpi's pi05 (`Pi0.embed_suffix`, `pi0.py`) did not support
this: it computes one scalar timestep per example and injects it as a single adaRMS
`(b, emb)` vector that uniformly scales/shifts *every* action token
(`gemma.RMSNorm`, `modulation[:, None, :]` broadcasts one vector across the whole
sequence). There was no way to tell the model "this chunk position is done, that one
is still noisy" — exactly what training-time RTC needs.

Closed with three small, backward-compatible changes (all in
`third_party/openpi/src/openpi/models/`):

- **`gemma.py:RMSNorm`** — `cond` may now be `(b, d)` (broadcast over the sequence, as
  before) *or* `(b, s, d)` (already one modulation vector per token); branch on
  `modulation.ndim` instead of always inserting a broadcast axis. `Module.__call__`'s
  `adarms_cond` type hint widened to accept both shapes.
- **`pi0.py:Pi0.embed_suffix`** — `timestep` may now be rank-1 `(b,)` (existing
  behavior, unchanged bit-for-bit) or rank-2 `(b, ah)` (one timestep per action-chunk
  position). The sincos embedding is computed by flattening `(b, ah) -> (b*ah,)`,
  embedding pointwise, and reshaping back — cheaper than and equivalent to vmapping
  over the chunk axis. `time_mlp_in`/`time_mlp_out` (nnx.Linear) apply unchanged
  regardless of rank, since they only act on the trailing embedding dim. Only the pi05
  path (`adaRMS`) exercises the rank-2 case; the non-pi05 concat path is untouched and
  in practice never receives rank-2 input (see below).
- **`pi0_config.py:Pi0Config`** — new `simulated_delay: int | None = None` field,
  validated to require `pi05=True` in `__post_init__` (training-time RTC needs the
  per-token adaRMS path above; pi0's non-pi05 concat-based time conditioning was never
  extended to support it, since there's no use case for it yet).

## 4. What's implemented vs. deferred

**Implemented** (`Pi0.compute_loss`, `pi0.py`): when `simulated_delay` is set, sample
per-example delay `d`, build `resolved = arange(action_horizon) < d` over the chunk
axis, set the per-position flow time to `0` (this repo's convention: `t=0` is the clean
action, `t=1` is pure noise — opposite of the reference repo's `t=1`-is-clean
convention, so "resolved" means pinned to `t=0`, not `t=1`) at resolved positions and
the globally-sampled time elsewhere, and forward with this per-position time through
the now-per-token adaRMS path. The per-position squared-error loss is rescaled
(`action_horizon / num_suffix_positions` on unresolved positions, `0` on resolved
ones) so that a **plain mean over the action-horizon axis downstream is already the
correct masked mean** — this means `scripts/train.py`'s `loss_fn` and the RA-BC
per-item weighting on top of it (`notes/reward_aligned_bc.md`) both compose with
training-time RTC with **zero changes**, since both already do
`jnp.mean(chunked_loss, axis=-1)` or an item-weighted variant of it as their first step.
When `simulated_delay is None` (every config except `yor_icl_pi05_rtc`), the RNG split
and computation path are bit-identical to before this change — a strict no-op.

**Deferred** (§5's phase 3, not built yet): the actual real-time inference-time
execution path — a `Pi0` method mirroring `realtime_action` that clamps `x_t`'s prefix
positions to a previous chunk's committed actions during `sample_actions`'s ODE
integration, *and* the closed-loop rollout harness change needed to exercise it
(re-planning before a chunk finishes and stitching in the new suffix, instead of the
current open-loop "execute `execute_horizon` steps then fully re-plan"). Training a
`simulated_delay` policy without this doesn't get you anything on its own — the point
of training-time RTC is what it enables at inference. This is intentionally out of
scope for the current training run; revisit once `yor_icl_pi05_rtc` has a checkpoint
worth evaluating this way.

## 5. Config

`yor_icl_pi05_rtc` (`third_party/openpi/src/openpi/training/config.py`) reuses
`yor_icl_pi05_easy_pnp_v2`'s exact data/hyperparameters (same 196-episode 4-task
pick-and-place `episodes_path`, `norm_stats.json` copied not recomputed, same
`batch_size`/LR schedule/optimizer/50k steps), changing only
`model=pi0_config.Pi0Config(pi05=True, action_horizon=30, simulated_delay=19)` — no
data-transform changes, since the mechanism lives entirely in the model/loss. `19` is
`round(action_horizon * 5/8)`, matching the reference repo's `simulated_delay=5` out of
`action_chunk_size=8` ratio (~62%); the delay actually sampled during training skews
well below that due to the exponential weighting. Trained from `pi05_base` for the
full 50k steps (not fine-tuned from an existing checkpoint) so it stays directly
comparable to the other pi05 arms on this dataset, per
`[[feedback-no-lerobot-pi05-training]]`'s openpi-only training constraint.

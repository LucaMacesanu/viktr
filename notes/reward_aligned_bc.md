# Reward-Aligned Behavior Cloning (RA-BC)

Source: `third_party/opensarm/2509.25358` (`.pdf`, no extension — arxiv 2509.25358,
"SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation", ICLR 2026;
Chen, Yu, Schwager, Abbeel, Shentu, Wu). RA-BC is Sec. 3.2 of that paper, built on
top of SARM/SARM2, their stage-aware progress reward model (the `third_party/opensarm`
repo itself). This note explains the method and proposes how to add it to VICTR —
it is a design doc, nothing here has been implemented yet.

## 1. What it is

RA-BC is a one-line change to the behavior-cloning loss: instead of averaging the
per-sample loss uniformly over a (possibly mixed-quality) demonstration dataset, each
training sample is weighted by how much *task progress* the demonstration makes over
that sample's action-chunk window, as estimated by a reward/progress model. Samples
that visibly move the task forward are weighted near 1; samples that stall or regress
are weighted near 0. It's a soft, learned-signal alternative to hand-filtering
demonstrations (e.g. VICTR's own `D_2min` duration-filtering ablation, see below) that
needs no extra human curation once a progress estimator exists.

The paper's motivation is directly analogous to ours: large real-world teleop datasets
are heterogeneous in quality (misgrasps, recovery struggles, back-and-forth motion),
naive BC on the union of everything underperforms BC on a small hand-filtered subset,
but hand-filtering (by duration, in their ablation) is a crude proxy that still misses
tasks needing fine visual judgment (their T-shirt-folding "is it flattened enough to
fold" example). RA-BC's reward-based weighting outperforms both.

## 2. The math (paper Sec. 3.2, Eq. 5–9)

Vanilla BC minimizes the uniform-average loss over dataset items `i` (state–action
pairs or, as used in their actual policy training, action-chunk windows):

```
L_BC(θ) = (1/N) Σ_i ℓ(π_θ(o_i), a_i)                                    (Eq. 5)
```

RA-BC replaces the uniform prior with a per-item weight `w_i`. For each training item
i, take a **progress delta** between the item's anchor observation and the observation
one action-chunk stride later:

```
b_i = φ(o_{t+Δ}) - φ(o_t)                                                (Eq. 6)
```

where `φ(·) ∈ [0,1]` is the progress/value model's estimate at a frame, `t` is the
anchor frame, and `Δ` is the action-chunk stride (they use Δ = 25 to match their
policy's action-chunking; ours would be the model's action horizon, see §5). `b_i` is
a raw, unnormalized signal of "how much progress did this chunk make."

The weighted objective:

```
L_RA-BC(θ) = [ Σ_i w_i · ℓ(π_θ(o_i), a_i) ] / [ Σ_i w_i + ε ]            (Eq. 7)
```

`w_i` is derived from `b_i` via **running dataset statistics**, not a fixed threshold.
They maintain an online (Welford) running mean `μ` and std `σ` of the raw deltas
`{b_j}` seen so far (clamping `μ ← max(μ, 0)` so weights don't get centered around
negative progress early in training), and map each delta to a soft weight by a linear
ramp between `μ - 2σ` and `μ + 2σ`:

```
w̃_i = clip( (b_i - (μ - 2σ)) / (4σ + ε), 0, 1 )                          (Eq. 8)
```

Then apply a hard override so clearly-good/clearly-bad items are decisive rather than
softly ramped, using one absolute threshold `κ > 0`:

```
w_i = 1{b_i > κ} + 1{0 ≤ b_i ≤ κ} · w̃_i                                  (Eq. 9)
```

i.e.: `b_i > κ` → weight 1 (unambiguously good progress); `b_i < 0` → weight 0
(regression/no progress, since the indicator terms are both false); `0 ≤ b_i ≤ κ` →
the soft ramped weight. Losses are first averaged over the action-chunk's own temporal
extent to get one scalar per item, *then* Eqs. 8–9 produce `w_i`, which is what makes
this a drop-in replacement for Eq. 5 rather than a per-timestep reweighting.

**Their hyperparameters** (T-shirt folding, Pi0 LoRA fine-tune, 40k steps, batch 32,
30fps data): `κ = 0.01`, `ε = 1e-6`, `Δ = 25` actions. They note `κ = 0.01` happens to
correspond to ~1.5 minutes of task duration at their frame rate, i.e. the top ~5% of
demonstrations by speed — a sanity-check post hoc, not how `κ` was actually chosen (it
was picked directly as a progress-delta threshold, dataset-agnostic in form even if the
specific value is dataset-calibrated).

## 3. Why it's worth doing (their results)

On T-shirt folding (flattened→folded = "medium" task, crumpled→folded = "hard" task),
fine-tuning Pi0 at 40k steps:

| Method | Medium SR | Hard SR |
|---|---|---|
| BC on full 200hr dataset (`D_all`) | 1/12 | 0/12 |
| BC on duration-filtered subset (`D_2min`) | 7/12 | 0/12 |
| RA-BC with ReWiND reward model | 6/12 | 3/12 |
| RA-BC with SARM reward model | **10/12 (83%)** | **8/12 (67%)** |

Two things stand out for us: (1) plain duration-filtering (their closest analogue to
VICTR's own oracle `t/(T-1)` progress signal being used only for *retrieval*, not
*training weighting*) helps on medium but is useless on hard — RA-BC's continuous,
per-chunk reward signal is what unlocks the hard task; (2) the ablation swapping in a
worse reward model (ReWiND) at the same κ/Δ nearly erases the gain — **weight quality
is bottlenecked by value-model quality**, which matters directly for us since our own
value sources (Robometer / RoboDopamine, see below) are not SARM-grade and this should
be treated as an open risk, not a guaranteed win.

## 4. What VICTR already has that RA-BC needs

The good news: the one hard prerequisite for RA-BC — a per-frame progress function
`φ(o) ∈ [0,1]` — already exists in this codebase in two independent forms, for two
different data sources:

- **`viktr.value.base.ValueEstimator`** (`src/viktr/value/base.py`) is exactly this
  interface (`estimate`/`estimate_episode`, `[0,1]` progress). Its MVP implementation,
  **`RobometerValueEstimator`** (`src/viktr/value/robometer.py`), is what
  `viktr.value.annotate.annotate_episode_values` uses for the LIBERO track.
- For the icl-dataset/openpi track (the one that actually matters — see the
  training-pipeline note in §5), **`viktr.data.icl_dataset`** already exposes two `φ`
  sources: `robodopamine_value_at(episode_index, frame_index)` (real per-frame
  estimates from a separately-run RoboDopamine value model, read from
  `meta/value_estimates/*.json`, interpolated onto the frame grid) and
  `oracle_value_at(frame_index, episode_length)` = `t/(T-1)` (Eq. 7 of *our own* paper,
  `notes/vktr.pdf`, used as a fallback where RoboDopamine coverage is missing). The
  openpi side has its own vendored copy of the RoboDopamine reader,
  `openpi/src/openpi/policies/yor_retrieval.py`'s `robodopamine_value_at` /
  `_robodopamine_curve` (kept manually in sync with the viktr-side original per that
  file's module docstring).
- Today `φ` is consumed for exactly one purpose: **retrieval**, not training-loss
  weighting. `viktr.retrieval.metrics.value_retrieve` picks context chunks by
  `|query_value - chunk_value|`, and `openpi/policies/yor_retrieval.py`'s
  `RetrievalContextInputs` transform does the same live/precomputed lookup at
  data-loading time to build `Pi0Victr`'s context blocks. RA-BC would be the first use
  of `φ` for *loss shaping* rather than *what to retrieve* — a new, orthogonal
  consumer of a signal that's already being computed and precomputed at scale.

**Important scoping constraint** (`[[feedback-no-lerobot-pi05-training]]` in memory):
real training only ever happens through **openpi** (`third_party/openpi`, JAX) —
`scripts/train_victr.py` (lerobot/PyTorch) is training-banned. So RA-BC has to be
built as an openpi data transform + training-loop change, not a lerobot-side change,
even though `src/viktr/value/*` (PyTorch/lerobot-side) is where `φ` sources are
originally implemented for LIBERO.

## 5. Design decisions for porting RA-BC into this codebase

**(a) Δ (chunk stride).** The paper's Δ=25 matches their policy's own action-chunk
length. Our direct analogue is the model's action horizon —
`Pi0VictrConfig(pi05=True, action_horizon=30, ...)` (`training/config.py`, the
existing VICTR train configs) / `icl_dataset.ACTION_CHUNK_SIZE = 30`. Recommend
`Δ = action_horizon` by default, exposed as an override in case a shorter/longer
progress-delta window turns out to work better than the action-prediction window
itself (they're conceptually distinct even though the paper reuses one value for both).

**(b) Running stats: offline instead of online.** The paper maintains `μ`/`σ` as an
*online* Welford estimator updated during training — appropriate for their setup,
where the reward model or the loss stream is not necessarily fully materializable in
advance. In our setup `φ` is either fully precomputed offline (RoboDopamine curves,
`meta/value_estimates/*.json`) or trivially precomputable (oracle `t/(T-1)`, or a
Robometer pass), and the training episode set is fixed ahead of time
(`episodes_path` JSON, e.g. `assets/yor_icl_pi05_easy_pnp_v2_episodes.json`). This
means `μ`, `σ`, and a `κ` derived from a percentile of the delta distribution (matching
how the paper explains their `κ=0.01` choice — top ~5% of items) can all be computed
**once, offline**, exactly like `openpi/scripts/compute_norm_stats.py` already computes
action/state `q01`/`q99` into `norm_stats.json` for QUANTILES normalization. This is
strictly simpler than online stats: no new mutable state to add to
`training_utils.TrainState`/checkpointing/EMA (`scripts/train.py`'s `train_step`
already threads `ema_params` through — avoid extending that machinery for this), and
fully deterministic/reproducible across resumed runs. Recommended unless an online
value model (e.g. one being jointly fine-tuned) makes precomputation impossible later.

**(c) Weight storage: precomputed per-episode curves, not runtime math.** Given (a)+(b),
the entire RA-BC weight for every `(episode_index, frame_index)` in the training set is
a fixed number computable in one offline pass. Store it exactly like a value curve —
one `(T,)` float32 array per episode — so the data-loading transform is an O(1) lookup,
mirroring `yor_retrieval.py`'s existing `_load_precomputed_episode` /
`_robodopamine_curve` convention, not new-runtime-math. Boundary case: for `t` within
`Δ` of an episode's end, `t + Δ ≥ T`; clamp to `T - 1` (same convention chunk-window
code elsewhere in this repo already uses for near-end indices), which just makes those
frames' deltas measure "progress over a shorter-than-Δ window" rather than being
undefined.

**(d) Orthogonal to VICTR's retrieval-context conditioning.** RA-BC only touches loss
weighting; it says nothing about what's in the model's input context. It should be
implementable as a data transform + training-loop change that composes with *any*
`LeRobotYorDataConfig`-derived config — plain pi05 fine-tuning (`YorInputs` only), KI
(`LeRobotYorKiDataConfig`), or VICTR-with-retrieval (`LeRobotYorVictrDataConfig`) — not
bolted onto `Pi0Victr`/`Pi0VictrConfig` specifically. Concretely: a new
`RabcWeightInputs` transform (mirroring `RetrievalContextInputs`'s shape) can sit in
any config's `data_transforms.inputs` list, and the `train.py` loss change only needs
to check "does this batch have a rabc weight field," independent of which model class
is training.

## 6. Concrete implementation plan

1. **Offline stats/weight precompute script** (viktr-side, PyTorch venv, since that's
   where `φ` sources already live): a new `scripts/precompute_rabc_weights.py`.
   Inputs: an episodes list (same JSON convention as `episodes_path`), a `φ` source
   selector (`"oracle"` / `"robodopamine"` / `"robometer"`), `Δ`. For every episode:
   compute the `(T,)` `φ` curve (reuse `viktr.data.icl_dataset.robodopamine_values` /
   `oracle_values`, or `viktr.value.robometer.RobometerValueEstimator.estimate_episode`
   for the LIBERO track), then the `(T,)` raw-delta curve `b_t = φ(clamp(t+Δ, T-1)) -
   φ(t)`. Pool all `b_t` across every episode in the training set to get one global
   `μ = max(mean(b), 0)`, `σ = std(b)`, and `κ` (default: a percentile of `b`, e.g.
   `np.quantile(b, 0.95)`, overridable). Apply Eq. 8–9 per-frame with those global
   constants to get the final `(T,)` weight curve per episode. Write one
   `assets/rabc_weights/<config-name>/<episode_index>.npy` per episode plus a
   `stats.json` (`{mu, sigma, kappa, epsilon, delta, phi_source}`) for provenance —
   same directory-per-run-config convention as `assets/victr_icl_pool*`.
2. **openpi-side data transform**: `openpi/src/openpi/policies/yor_rabc.py` (new file,
   or added to `yor_retrieval.py` since it can share the RoboDopamine-curve-reading
   helpers already vendored there) — `RabcWeightInputs(transforms.DataTransformFn)`
   with a `weights_dir: str` field. `__call__` reads `data["episode_index"]`/
   `data["frame_index"]`, memory-maps that episode's precomputed `.npy` (cached per
   dataloader worker, same `functools.lru_cache` pattern as `_load_precomputed_episode`
   / `_load_pool_cached`), and adds `data["rabc_weight"] = float(curve[frame_index])`.
   Must run before `yor_policy.YorInputs()` (needs the raw `episode_index`/
   `frame_index` keys `YorInputs` consumes/replaces), same ordering constraint as
   `RetrievalContextInputs`.
3. **`Observation` field**: add `rabc_weight: at.Float[ArrayT, "*b"] | None = None` to
   `openpi/src/openpi/models/model.py`'s `Observation` (`from_dict`/`to_dict` need the
   matching `data.get("rabc_weight")` plumbing) — same no-op-everywhere-else pattern as
   `context_images`/`subtask_tokens`: every model that doesn't know about it just
   ignores it, only the training loop reads it.
4. **`TrainConfig`/`DataConfig` wiring**: extend the relevant `DataConfigFactory`
   subclass(es) in `openpi/src/openpi/training/config.py` with a `rabc_weights_dir:
   str | None = None` field; when set, prepend `RabcWeightInputs(weights_dir=...)` to
   `data_transforms.inputs` in `create()`. Given (§5d), do this as a small mixin/shared
   helper usable from `LeRobotYorDataConfig`, `LeRobotYorKiDataConfig`, and
   `LeRobotYorVictrDataConfig` alike, rather than only on the Victr config.
5. **Loss change**: `openpi/scripts/train.py`'s `train_step.loss_fn` (currently
   `chunked_loss = model.compute_loss(...); return jnp.mean(chunked_loss)`,
   `scripts/train.py:150-151`). `compute_loss` returns per-`(batch, action-horizon)`
   MSE (`Pi0Victr.compute_loss` / `pi0.Pi0.compute_loss`, shape `[*b, ah]`). Change to:
   average over the action-horizon axis first to get one scalar per batch item
   (`per_item = jnp.mean(chunked_loss, axis=-1)`), then, if
   `observation.rabc_weight is not None`, return the Eq. 7 weighted mean
   (`jnp.sum(w * per_item) / (jnp.sum(w) + eps)`) instead of the plain
   `jnp.mean(per_item)`. `eps` as a small constant (config field or hardcoded `1e-6`,
   matching the paper). No TrainState/checkpoint changes needed (per §5b).
6. **New named `TrainConfig` entries**: alongside the existing vision/value-retrieval
   VICTR configs (`training/config.py:1048`, `:1075`, `:1116`), add one or more
   RA-BC-enabled configs — both a plain-pi05 + RA-BC arm (isolates RA-BC's effect from
   retrieval) and a VICTR-retrieval + RA-BC arm (paper's own headline setup: reward
   model drives both retrieval *and* loss weighting). Mirrors the paper's own
   `RA-BC-ReWiND` vs `RA-BC-SARM` ablation structure — see §7.
7. **SLURM/shells wrapper**: one `shells/slurm/victr_precompute_rabc_weights_job.sh`
   (mirrors the existing `victr_precompute_expanded_*_job.sh` pattern) for step 1's
   offline precompute on HPC, since `meta/value_estimates` (RoboDopamine) coverage
   likely needs to be checked/backfilled first — per `icl_dataset.py`'s docstring,
   RoboDopamine is not guaranteed to cover every episode, oracle is the fallback.

## 7. Validation plan (mirrors the paper's Q1–Q3)

Given `[[feedback-no-lerobot-pi05-training]]`, all of this runs as openpi `TrainConfig`
jobs, evaluated the same way existing VICTR arms are (`scripts/eval_victr_offline.py`
/ closed-loop rollout harness), not via `scripts/train_victr.py`.

- **RA-BC vs. plain BC, same data.** Fine-tune identical configs on the same
  `episodes_path` with and without `RabcWeightInputs` — isolates whether reweighting
  helps at all before touching retrieval.
- **RA-BC value-source ablation** (their Q3 finding: reward-model quality gates
  RA-BC's benefit). Compare RA-BC weighted by `oracle_value_at` (cheap, always
  available, but naive linear-in-time — closer to their duration-filtering baseline
  than a real reward signal) vs. `robodopamine_value_at` (real learned progress
  estimates) vs. (LIBERO track only) `RobometerValueEstimator`. Expect oracle-weighted
  RA-BC to underperform RoboDopamine-weighted RA-BC on tasks with meaningfully
  nonlinear progress (their T-shirt case; probably less pronounced on our own
  pick-and-place tasks, which are shorter/simpler — worth checking whether RA-BC's
  benefit is even visible at our task complexity before investing further).
- **RA-BC + retrieval vs. RA-BC alone vs. retrieval alone** — a 2x2 over
  {plain pi05, VICTR-retrieval} x {no RA-BC, RA-BC}, to see whether the two mechanisms
  (what to retrieve into context vs. how to weight the loss) are additive, as the
  framing in §5d assumes but which isn't validated by anything in the SARM paper
  (they never combine RA-BC with in-context retrieval — that combination is novel to
  us, not reproduced from the paper).

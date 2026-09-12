# ICRA training plan: canonical EE+rot6d action/observation space

Status as of 2026-09-10 (late): both in-flight "aligned" extended jobs
(`yor_icl_pi05_aligned_q_extended` job 17299822, `yor_icl_pi05_aligned_deltarot6d_extended`
job 17326328) were killed early on — neither used the action/observation space below, and
both predate the gripper/lift/base decisions made in this plan.
`yor_icl_pi05_aligned_q_sanity15k` and `yor_icl_pi05_aligned_deltarot6d_sanity15k`
checkpoints already uploaded to `lair-nyu` remain as historical artifacts (the
aligned-q arm in particular — quat-based aligned-deltarot6d state, no gripper in
aligned-q's action) but are **not** continued under this plan; all 7 arms below
restart from a single new, shared action/observation space.

**`yor_icl_pi05_aligned_deltarot6d_sanity15k` was deployed and confirmed 0% SR, wrist
rotating sporadically as soon as a rollout starts** (vs. 85% SR for
`yor_icl_pi05_easy_pnp_v2`, whose state has no quaternion at all — joint-space only).
Root-caused, not just suspected: that arm's state is a raw quaternion per arm
(`YorInputsAlignedDeltaRot6D`), and `icl-dataset-fixed-obs`'s quaternion hemisphere
convention is resolved via temporal continuity/unwrapping across each recorded
episode (confirmed empirically: qw negative in 42% of all frames, sign-flips
mid-episode in 16% of episodes) — a property a live rollout's per-timestep FK can't
reproduce without explicitly replicating that unwrap. Quantified on the actual
196-episode sanity-training population (not just the full dataset): only 6.1% of
episodes (left arm only, right arm clean) actually cross the ambiguous zone, but when
they do it's bursty, not a single clean flip — the wrist dwells with `|qw|<0.05` for
20-40% of the episode once it enters that band, producing clusters of 2-7 spurious
sign-flip events within a couple seconds, concentrated ~70-90% through the episode (the
place/release phase for these 4 tasks). This is a real, demonstrated contributor to
the 0% SR, though probably not provably the *sole* cause (94% of demo episodes never
cross the boundary at all) — plots and methodology in the scratchpad from that session.
This is exactly the fragility rot6d state (§1) is structurally immune to
(`R(q) == R(-q)` for any rotation matrix construction), and is the strongest concrete
evidence yet that switching state to rot6d (not just action) is worth the churn below,
beyond the representational-consistency argument in §1.

**Progress on the 7 arms** (see the per-arm status table in §2 and the corrected
precompute analysis in §3 for detail):
- Arm 1 (`yor_icl_pi05_canonical_sanity15k`, job 17333295): running since ~16:11,
  15k steps, no errors. Confirmed via a full smoke test: state 18-dim/action 20-dim,
  rot6d blocks unit-norm and orthogonal. 5000-step checkpoint uploaded to
  `lair-nyu/yor_icl_pi05_canonical_sanity15k-step5000` (private, `params`+`assets`
  only, `train_state` dropped, matching the existing checkpoint-repo convention).
- Arm 2 (`yor_icl_pi05_canonical_extended`): episode set trimmed from the original
  31-task/1,784-episode expanded set down to **20 tasks / 1,186 episodes** (11 tasks
  dropped — see the new §1b below for the list and rationale) after this plan was
  first written; norm_stats recomputed on the trimmed population (genuinely needed —
  same transform, different episode population, so different empirical quantiles).
  Job 17354108 (this plan's 3rd submission of this arm — the first ran on the old
  31-task set and was cancelled mid-run when the trim was decided; the 2nd crashed
  instantly on a stale `checkpoints/` dir left by that cancellation; the 3rd crashed
  instantly on a `repo_dir` submission typo). Currently pending on GPU availability.
- Arms 3-5 (VICTR vision/value/vision_value canonical): registered and submitted —
  jobs 17354321/17354322/17354323, pending on GPU availability, `batch_size=256` per
  explicit direction (overriding the initial submission at 128, the existing VICTR
  precedent — see §2's flag; **this is genuinely untested at 256 for a VICTR arm**,
  real OOM risk on a 47h job, not yet de-risked by a short smoke run). Turned out to
  need less new precompute than §3 originally assumed — see the correction there —
  but DID need a new, smaller thing §3 didn't anticipate: the retrieval pool's own
  `proprio`/`actions` fields (baked into the retrieved neighbor's `context_tokens`
  text description) were still in the old raw representation, so they were patched to
  canonical (§1a addendum below) before submitting.
- Arms 6-7 (RICL, pi0_fast canonical): not started. Still blocked on §3.2's FAST
  tokenizer refit, which hasn't been run yet.

## 1. Canonical action/observation space (applies to every arm below)

**Observation**: 3 cameras (`base_0_rgb`/zed, `left_wrist_0_rgb`/fish0,
`right_wrist_0_rgb`/fish1, unchanged from every existing Yor config) + proprioception
= **absolute EE position + absolute rot6d orientation, per arm, nothing else**.
9-dim per arm (pos 3 + rot6d 6), 18-dim total. No gripper, no lift, no base in state.

**Action**: per arm, **delta EE position (relative to the query window's own first
frame) + absolute rot6d orientation + gripper** = 3 + 6 + 1 = 10, both arms = 20-dim
total. No lift, no base.

This reuses `YorInputsDeltaRot6D`'s existing action math exactly (already computes
delta-pos + rot6d + gripper from the raw 20-dim `action` column's
`_LEFT_POS`/`_RIGHT_POS`/`_LEFT_QUAT`/`_RIGHT_QUAT`/`_LEFT_GRIP`/`_RIGHT_GRIP` slices,
never touches `action`'s dims 16:20 where base_vel/lift_cmd live — so the action side
needs **no new code**, only a new class name/registration). The only genuinely new
piece is proprioception: today's closest arm
(`YorInputsAlignedDeltaRot6D`/`icl-dataset-fixed-obs`) uses raw quaternion
(`observation.left_ee`/`observation.right_ee`, quat+xyz, 14-dim) for state, not rot6d —
this plan converts that to rot6d via `yor_rotation.quat_wxyz_to_rot6d` (already used by
the action side of every DeltaRot6D arm, so no new rotation-conversion code either,
just applying it to state instead of only to action).

### 1a. New code (small — mostly wiring, not new math)

- `yor_policy.py`: `YorInputsCanonical` / `YorOutputsCanonical` —
  copy `YorInputsAlignedDeltaRot6D`'s camera/action logic verbatim, but build state as
  `concatenate([pos_L(3), quat_wxyz_to_rot6d(quat_L)(6), pos_R(3), quat_wxyz_to_rot6d(quat_R)(6)])`
  from `observation.left_ee`/`observation.right_ee` instead of passing the raw
  quat+xyz through. `YorOutputsCanonical` is identical to `YorOutputsAlignedDeltaRot6D`
  (action-side unpad/rot6d→quat is unchanged).
- `config.py`: `LeRobotYorCanonicalDataConfig(LeRobotYorDeltaRot6DDataConfig)` —
  same shape as `LeRobotYorAlignedDeltaRot6DDataConfig`, swap in the two classes above.
  `root` points at `icl-dataset-fixed-obs` (already built this session at
  `/scratch/lim2045/icl_ws/icl-dataset-fixed-obs`, no new dataset merge needed).
- `config.py`: `LeRobotYorVictrCanonicalDataConfig(LeRobotYorVictrDataConfig)` — for the
  3 VICTR arms + RICL, override `create()` to swap `YorInputs`/`YorOutputs` for
  `YorInputsCanonical`/`YorOutputsCanonical` (everything else — `RetrievalContextInputs`
  wiring, `use_action_interpolation` plumbing — is unchanged, it's agnostic to the
  query's own action/state encoding).

Naming below (`yor_icl_*_canonical*`) is a proposal, not a commitment — flag before
implementing if you want different names.

### 1a addendum. Retrieval pool proprio/actions patch (added after arms 3-5 were built)

The retrieval pool's chunks (`assets/victr_icl_pool_expanded_224/*.pkl`) carry their
own `proprio`/`actions` arrays per chunk, independent of the query's own state/action
encoding — `yor_retrieval.chunk_to_context_text` digitizes these directly into the
`context_tokens` text block ("Task: ...; State: ...; Action: ...") every VICTR arm's
model actually reads. Checked one chunk directly: `proprio` was `(10, 15)` (old
joint-space `observation.state`), `actions` was `(10, 20)` (old absolute
quat+xyz+grip+base/lift) — neither matches canonical. This is **not** the same
dependency §3.3 originally worried about (`RetrievalContextInputs`' `action_norm_stats`
wiring, which only fires for RICL's action-interpolation path) — it's a separate,
previously-unnoticed one: the pool's own stored arrays, baked into the context text
regardless of query encoding.

Fixed by rebuilding just `proprio`/`actions` (not images, not `key_embeddings`/
`key_values` — those are DINOv2-embedding/RoboDopamine-value-based and
representation-agnostic) into a new pool dir, `assets/victr_icl_pool_canonical_224`,
using the same math as `YorInputsCanonical` (state: pos+rot6d per arm; action: delta
position relative to the *chunk's own* first frame, generalizing "query window's own
first frame" to a chunk since chunks have no other window reference) — computed from
`icl-dataset-fixed-obs` for each chunk's exact `(episode_index, start_frame:end_frame)`
window. Only the 20 tasks arm 2's trimmed episode set (§1b) can actually query were
rebuilt (the other 11 tasks' pools are never loaded by these arms regardless, since
`yor_retrieval._load_pool_cached` loads per-task on demand — no need to touch them).
Verified: all 20 rebuilt files load cleanly, correct shapes (proprio 18-dim, actions
20-dim), rot6d blocks unit-norm/orthogonal to float32 precision, `key_embeddings`/
`key_values` byte-identical to the source pool. The original
`victr_icl_pool_expanded_224` (used by the older, now-superseded expanded VICTR arms)
was left untouched.

### 1b. Extended episode set trimmed to 20 tasks (added after arm 2's row was written)

Arm 2 and up were originally scoped to the full 31-task/1,784-episode expanded set
(`assets/yor_icl_expanded_episodes.json`). After checking frame-volume impact, 11
tasks were dropped, producing `assets/yor_icl_canonical_extended_episodes.json`
(1,186 episodes, -33.5%; 512,102 frames, -49.8% — two of the 11,
`sort_the_items_into_their_containers`/`stack_the_three_cups_into_a_tower`, alone
account for ~62% of the removed frame volume despite being only ~17% of the removed
episodes — by far the longest-horizon tasks in the set). This applies to every arm
2-7 above (all share the trimmed episode set), not just arm 2.

**Removed (11)**: `hit the yellow cube with the mallet using the left arm`,
`open the gatorade bottle`, `pass the red chilli from the right arm to the left arm`,
`pass the salt shaker from the right arm to the left arm`,
`pass the yellow cube from the right arm to the left arm`,
`pick up the orange cube and place it on the plate with the left arm on an uncluttered table`,
`pick up the orange cube and place it on the plate with the right arm on an uncluttered table`,
`put the bottle on the plate` (left-arm variant), `put the bottle on the plate with the right arm`,
`sort the items into their containers`, `stack the three cups into a tower`.

**Kept (20)**: `clean the plate`,
`hit the yellow cube with the mallet using the right arm`,
`pass the red chilli from the left arm to the right arm`,
`pass the salt shaker from the left arm to the right arm`,
`pass the yellow cube from the left arm to the right arm`,
`pick up the eggplant and place it on the plate with the left arm`,
`pick up the eggplant and place it on the plate with the right arm`,
`pick up the orange cube and place it on the plate with the left arm on a cluttered table`,
`pick up the orange cube and place it on the plate with the right arm on a cluttered table`,
`pick up the pumpkin and place it on the plate with the left arm`,
`pick up the pumpkin and place it on the plate with the right arm`,
`pick up the red chilli and place it on the plate with the left arm`,
`pick up the red chilli and place it on the plate with the right arm`,
`place the red chilli in the cup`, `place the red chilli in the cup with the left arm`,
`put the circle on the peg`, `put the eggplant into the box with the left arm`,
`put the eggplant into the box with the right arm`, `uncap the blue marker`,
`uncap the red marker`.

Norm_stats were recomputed on the trimmed population (not reused) — same transform as
before the trim, but a different episode population changes the empirical q01/q99
quantiles, unlike a pure axis-drop or identical-episode-set reuse.

## 2. The 7 arms

| # | Name (proposed) | Steps | Model config | Data config | Precedent this reuses | Status (2026-09-10) |
|---|---|---|---|---|---|---|
| 1 | `yor_icl_pi05_canonical_sanity15k` | 15,000 | `pi0_config.Pi0Config(pi05=True, action_horizon=30)` | `LeRobotYorCanonicalDataConfig`, 4-task pnp episodes | `yor_icl_pi05_aligned_deltarot6d_sanity15k` (today's killed arm) | **Running** — job 17333295 |
| 2 | `yor_icl_pi05_canonical_extended` | 50,000 | same | same, **20-task/1,186-episode trimmed set** (§1b — was 31-task/1,784 when this row was first written) | `yor_icl_pi05_expanded` | **Pending** (GPUs) — job 17354108 |
| 3 | `yor_icl_victr_vision_canonical` | 50,000 | `pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision")` | `LeRobotYorVictrCanonicalDataConfig`, `retrieval_metric="vision"`, same trimmed episode set as arm 2, `pool_dir="assets/victr_icl_pool_canonical_224"` (§1a addendum) | `yor_icl_victr_vision_expanded` | **Pending** (GPUs) — job 17354321 |
| 4 | `yor_icl_victr_value_canonical` | 50,000 | same, `retrieval_metric="value"` | same, `retrieval_metric="value"` | `yor_icl_victr_value_expanded` | **Pending** (GPUs) — job 17354322 |
| 5 | `yor_icl_victr_vision_value_canonical` | 50,000 | same, `retrieval_metric="vision_value"` | same, `retrieval_metric="vision_value"` | `yor_icl_victr_vision_value_expanded` | **Pending** (GPUs) — job 17354323 |
| 6 | `yor_icl_fast_victr_vision_interp_canonical` (**RICL**) | 50,000 | `pi0_fast_victr.Pi0FastVictrConfig(action_dim=20, action_horizon=30, use_action_interpolation=True, lamda=10.0, max_action_tokens=224, retrieval_metric="vision", fast_model_tokenizer_kwargs=<new canonical tokenizer, see §3>)` | `LeRobotYorVictrCanonicalDataConfig`, live retrieval only (`precomputed_context_dir=None`) | `yor_icl_fast_victr_vision_interp_expanded` — this arm already exists and already implements RICL's action-token-blending method (see `notes/action_interpolation.md`); nothing algorithmically new needed, just re-pointing at the canonical space + tokenizer | **Not started** — blocked on §3.2 |
| 7 | `yor_icl_pi0_fast_canonical` | 50,000 | `pi0_fast.Pi0FASTConfig(action_dim=20, action_horizon=30, max_token_len=256, fast_model_tokenizer_kwargs=<new canonical tokenizer>)` | `LeRobotYorCanonicalDataConfig` | `yor_icl_pi0_fast_expanded` | **Not started** — blocked on §3.2 |

All 7: `batch_size=256`, `num_workers=8`, `CosineDecaySchedule(warmup_steps=2_000,
peak_lr=5e-5, decay_steps=<num_train_steps>, decay_lr=5e-6)`,
`AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0)`,
`ema_decay=None`, `save_interval=5_000`, seed=42. Arms 1–2 use
`weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`;
arms 3–5 use `weight_loaders.VictrCheckpointWeightLoader(<same pi05_base path>)`; arm 6
uses `VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params")`;
arm 7 uses plain `CheckpointWeightLoader` on the same pi0_fast_base path.

**⚠️ batch_size=256 flag — arms 3–5 submitted at 256 by explicit direction, still
unvalidated**: every existing VICTR/RICL precedent (arms 3–6's ancestors) in this
codebase was run at `batch_size=128`, not 256 — I found no documented reason (no OOM
note in `notes/training_runs.md`), but the retrieval-context arms carry extra
per-sample tokens (context chunks/text) that the plain pi05/pi0-fast arms don't, so
256 may not fit in H200 memory for arms 3–6 where it fits fine for arms 1, 2, 7. Arms
3–5 were first submitted at the safer precedent value (128), then re-submitted at 256
per explicit direction, without a smoke test to de-risk it first — genuine OOM risk on
a 47h job if it doesn't fit; worth checking the first few steps' memory/loss once one
of these actually starts. Still open for arm 6 (RICL) whenever it's built.

## 3. Preprocessing that must happen before training, per arm

1. **Fresh norm_stats** (arms 1–7, all): action/state distributions are new (rot6d
   state, delta+rot6d+gripper action) — `scripts/compute_norm_stats.py` for each of
   arm 1 and arm 2's configs; arms 3–7 copy arm 2's `norm_stats.json` (same convention
   already used across every existing expanded-set VICTR/FAST arm — same episodes,
   same action/state encoding once the canonical `DataConfig` is shared).
2. **Fresh FAST tokenizer fit** (arms 6, 7 only): the existing
   `outputs/fast_tokenizer/yor-icl-expanded` tokenizer was fit on the OLD 20-dim action
   (quat+xyz+grip+base+lift) — a different distribution from canonical's
   delta+rot6d+grip, even though both are 20-dim. Must re-run
   `scripts/fit_expanded_fast_tokenizer.py` (or `fit_combined_fast_tokenizer.py`,
   whichever matches the 31-task expanded set) against the canonical action encoding
   before arms 6/7 can train. Existing SLURM wrapper:
   `shells/slurm/victr_fit_expanded_fast_tokenizer_job.sh`.
3. **Retrieval context re-precompute — CORRECTED, turned out NOT needed for arms 3-5**:
   this section originally claimed `RetrievalContextInputs`' `action_norm_stats`
   wiring (`action_norm_stats=base.norm_stats["actions"]`) meant the precomputed
   context depended on the query's action encoding for all of arms 3-6. Wrong for
   arms 3-5: read `RetrievalContextInputs.__call__` directly — that `action_norm_stats`
   codepath only fires when `use_action_interpolation=True`, which is asserted `False`
   for every `Pi0VictrConfig` arm (arms 3-5 all use `Pi0VictrConfig`, not
   `Pi0FastVictrConfig`) — it's RICL-only (arm 6). For arms 3-5, the precomputed
   context (`build_context_block`) depends only on the query's own DINOv2 image
   embedding (vision metric) or RoboDopamine progress value (value metric), both
   computed from the query's own image/episode — never from its action/state encoding.
   Confirmed by checking coverage directly: the existing
   `outputs/victr/retrieval_context_expanded/{vision,value,vision_value}` (precomputed
   over the original 1,784-episode expanded set) covers all 1,186 episodes of the
   trimmed canonical set (§1b) with 0 missing — reused unchanged for arms 3-5, no
   re-precompute run. Still genuinely needed for arm 6 (RICL) whenever it's built —
   that arm's `use_action_interpolation=True` does hit the `action_norm_stats`
   codepath, and it uses live retrieval anyway (`precomputed_context_dir=None`).

   What the pool/precompute assets DID actually need (not anticipated by this section
   originally): see the §1a addendum above — the pool's own `proprio`/`actions`
   fields, baked into `context_tokens`' digitized text description, needed patching to
   canonical regardless of query encoding or the `action_norm_stats` question. The
   retrieval **pool**'s images/`key_embeddings`/`key_values` (DINOv2 embeddings +
   oracle values) did NOT need rebuilding — those are keyed on images/rewards, not on
   the query's or the pool's own action encoding.
4. **Value function**: `retrieval_metric="value"`/`"vision_value"` (arms 4, 5) read a
   precomputed oracle value per `(episode_index, frame_index)`
   (`robodopamine_value_at`, `yor_retrieval.py`) — this is a reward/outcome signal, not
   an action-encoding-dependent one, so it should not need retraining. ✅ Checked: all
   3,149 episodes have a `meta/value_estimates/*.json` entry, including all 1,186 of
   the trimmed canonical set — nothing missing, no work needed.

## 4. Execution order

1. ✅ Built `YorInputsCanonical`/`YorOutputsCanonical` +
   `LeRobotYorCanonicalDataConfig`/`LeRobotYorVictrCanonicalDataConfig`, registered arm 1
   (`yor_icl_pi05_canonical_sanity15k`).
2. ✅ Smoke-tested the data pipeline: state 18-dim, action 20-dim confirmed, rot6d
   blocks verified unit-norm and orthogonal.
3. ✅ Computed norm_stats for arm 1, submitted the 15k sanity job (17333295, running,
   no errors). Step-0 loss sanity not yet explicitly re-confirmed this round (only
   checked via wandb, not pulled into this doc).
4. ✅ (partial) Registered + computed norm-stats for arm 2 (`_extended`) — on a
   **trimmed** 20-task/1,186-episode set, not the original 31-task/1,784 (§1b, decided
   after this plan's step 4 was first written). Registered + submitted arms 3-5
   (vision/value/vision_value canonical) — turned out `precompute_retrieval_context.py`
   did NOT need re-running (§3 correction), but the retrieval pool's own
   `proprio`/`actions` did need a separate patch (§1a addendum) that this step didn't
   originally anticipate. **Not yet done**: the FAST tokenizer refit (§3.2) and arms
   6-7 (RICL, pi0_fast canonical) — both still blocked on that refit.
5. ✅ (by direction, not validation) batch_size=256 was requested explicitly for arms
   3-5 rather than resolved by testing — see §2's flag. **Still open**: no smoke run
   has actually confirmed 256 fits in H200 memory for a VICTR arm; worth checking once
   one of jobs 17354321/17354322/17354323 actually starts.

## 5. Open questions

- ~~Naming convention~~ — resolved: `_canonical` suffix used for all registered arms.
- ~~Whether `batch_size=256` is a hard requirement for the VICTR arms~~ — resolved by
  direction: yes, use 256 for arms 3-5 (§2, §4 step 5) — but this is a decision, not a
  validated-safe result; still genuinely unconfirmed whether it fits in H200 memory
  for a VICTR arm. Watch the first steps of 17354321/17354322/17354323 once they
  start.
- Whether arm 6 (RICL) should also get a `_serve` twin now (existing
  `yor_icl_fast_victr_vision_expanded_serve` pattern) or only once results look good —
  not included above, out of scope unless requested. Still open, and arm 6 itself
  hasn't been started yet (blocked on the FAST tokenizer refit, §3.2).
- New: should the FAST tokenizer refit (§3.2, arms 6-7) be scoped to the trimmed
  20-task set (§1b) or the original 31-task set? Not yet decided — needs resolving
  before arms 6-7 can be built.

# openpi training runs — status tracker

Living doc, one row per `openpi` `TrainConfig` (`third_party/openpi/src/openpi/training/config.py`).
Update the **Status** / **Latest step** columns as jobs progress; update **Notes**
when something changes materially (crash, fix, resume). Checkpoints live at
`third_party/openpi/checkpoints/<config_name>/<exp_name>/`; wandb project is
`luca-m-new-york-university/openpi`.

Per [[feedback_no_lerobot_pi05_training]]: openpi/JAX is the sole training
pipeline now — `scripts/train_victr.py` (lerobot/PyTorch) is training-banned.
`notes/progress.md` predates this pivot and is stale on that point.

## Original 4-task subset (left-arm-only pick-and-place)

All of these use the un-overridden default `episodes_path` (not
`assets/yor_icl_expanded_episodes.json`) — the original left-arm-only 4-task
pick-and-place subset. Action dims 15–19 (right gripper, base vel, lift_cmd)
are exactly 0 for every episode in this subset (confirmed via norm_stats:
`std=0.0` exactly), so none of these were ever exposed to the base/lift
normalization bug below, regardless of code path.

| config | model | retrieval | latest checkpoint | target steps | SLURM | notes |
|---|---|---|---|---|---|---|
| `yor_icl_pi05_easy_pnp_v2` | `Pi0Config` (plain pi05) | – | 14,999 (`..._sanity15k`) | 50,000 | not queued | multiple exp-name attempts (`dryrun*`, `_full`, `_full2`, `_sanity15k`); wandb history for `_full2` only shows 200 steps, crashed early — not investigated further, loss looked healthy (0.08–0.41) while it ran |
| `yor_icl_pi05_rabc` | `Pi0Config` + RA-BC weighting | – | 28,600 (wandb) / 25,000 (checkpoint) | 50,000 | not queued | wandb state `crashed`; loss healthy throughout (0.0009–0.41) |
| `yor_icl_pi05_rtc` | `Pi0Config`, `simulated_delay=19` (training-time RTC) | – | 45,000 | 50,000 | **running**, job 16371057 | healthy (loss 0.0005–0.4 range) |
| `yor_icl_victr_vision` | `Pi0VictrConfig` | vision | dryrun-only (no `_full` checkpoint) | 50,000 | not queued | wandb `_full2` only 200 steps, crashed early; loss healthy while it ran |
| `yor_icl_victr_value` | `Pi0VictrConfig` | value | dryrun-only (no `_full` checkpoint) | 50,000 | not queued | same as above |
| `yor_icl_ki_subtask` | `Pi0KiConfig` (insulated flow + FAST-action/subtask aux losses) | – | dryrun-only | 50,000 | not queued | wandb only 400 steps, crashed early; loss healthy (3.4–31.2, discrete aux loss included) |
| `yor_icl_ki_subtask_rabc` | `Pi0KiConfig` + RA-BC | – | 20,000 | 50,000 | not queued | wandb `crashed` at step 21,200; loss healthy (0.03–31.6) |

## Expanded set (1,784-episode bimanual, `assets/yor_icl_expanded_episodes.json`)

**Bug (fixed 2026-08-27):** action dims 16:20 (`base_vel.vx/vy/omega`,
`lift_cmd`) are unused-but-not-exactly-zero in this set (std between `0.0`
and `9e-4`). `openpi/transforms.py`'s `Normalize` (`(x-mean)/(std+1e-6)`, by
design unclipped for continuous consumers) amplified that residual jitter by
up to ~1e6x, producing per-item loss spikes up to ~6.3e5 in every arm whose
primary loss is continuous flow-matching. Fixed in
`openpi/policies/yor_policy.py`'s `YorInputs` (shared by every `yor_icl_*`
config) by zeroing those 4 dims before `Normalize` ever sees them — no model
shape change needed, pi0 already zero-pads the action space to 32 regardless
of real action_dim. Discrete FAST-token consumers were never affected: both
`tokenizer.py`'s `FASTTokenizer.tokenize` and `yor_ki.py`'s `KiTargetInputs`
already clip to `[-1, 1]` before quantizing (an earlier fix, predating this
session, for the `chr()` range crash below).

| config | model | retrieval | latest checkpoint (pre-fix) | target steps | SLURM | notes |
|---|---|---|---|---|---|---|
| `yor_icl_pi05_expanded` | `Pi0Config` | – | ~~49,999 (corrupted)~~ | 50,000 | **pending**, job 16468973 (`--overwrite`) | **was broken** — median loss ~46k, max 6.3e5 over the full run. Requeued fresh post-fix. |
| `yor_icl_ki_expanded_subtask` | `Pi0KiConfig` | – | ~~20,000 (corrupted)~~ | 50,000 | **pending**, job 16468974 (`--overwrite`) | **was broken** on its primary flow-matching branch (median loss ~45k). Its discrete FAST-action aux target already had the `np.clip` crash-fix (`yor_ki.py:154`, comment cites job 16351040/step 319 — `ValueError: chr() arg not in range(0x110000)`), but that only stopped the crash, not the flow-loss explosion. Requeued fresh post-fix. |
| `yor_icl_victr_value_expanded` | `Pi0VictrConfig` | value | ~~30,000 (corrupted)~~ | 50,000 | **pending**, job 16468975 (`--overwrite`) | **was broken** — same bug, discovered while investigating the two above (median loss ~46k). Was running live when found; cancelled and requeued. |
| `yor_icl_victr_vision_expanded` | `Pi0VictrConfig` | vision | ~~20,000 (corrupted)~~ | 50,000 | **running**, job 16468976 (`--overwrite`) | **was broken** — same bug (median loss ~45k). Was running live when found; cancelled and requeued. |
| `yor_icl_victr_vision_value_expanded` | `Pi0VictrConfig` | vision_value | never run | 50,000 | **pending**, job 16469185 | never trained before this session — launched fresh post-fix, so never actually hit the bug, just would have |
| `yor_icl_fast_victr_vision_expanded` | `Pi0FastVictrConfig` (discrete FAST, autoregressive) | vision | 15,000 | 50,000 | **running**, job 16429151 | **not affected** — entire loss is discrete FAST-token cross-entropy, already protected by the pre-existing tokenizer clip (see above). Untouched by this session's fix/requeue. |
| `yor_icl_pi05_expanded_frozen_vision` | `Pi0Config`, `freeze_filter=PathRegex(".*img.*")` (SigLip vision tower frozen, Gemma LM + action expert hot) | – | never run | 70,000 | **pending**, job 16476252 (`QOSGrpGRES`) | new config added post-fix, never hit the bug. 4×H200 (`gres=gpu:h200:4`, 64cpu/800G), global batch 256, `lr_schedule` decay_steps=70,000 to match num_train_steps (user-confirmed choice — see [[feedback_no_lerobot_pi05_training]] sibling doc for the annealing discussion that motivated matching decay_steps to the actual run length). Reuses `yor_icl_pi05_expanded`'s norm_stats.json (copied, same convention as the VICTR expanded arms). Auto-scheduled onto reserved node `gh125` (reservation `jz4725-h200`, MAGNETIC, no explicit `--reservation` flag needed). |

## Job IDs (as of 2026-08-27, post-fix requeue)

- 16371057 — `yor_icl_pi05_rtc` — running
- 16429151 — `yor_icl_fast_victr_vision_expanded_full` — running
- 16468973 — `yor_icl_pi05_expanded_full` — pending (QOSGrpGRES / node availability)
- 16468974 — `yor_icl_ki_expanded_subtask_full` — pending
- 16468975 — `yor_icl_victr_value_expanded_full` — pending
- 16468976 — `yor_icl_victr_vision_expanded_full` — running
- 16469185 — `yor_icl_victr_vision_value_expanded_full` — pending
- 16476252 — `yor_icl_pi05_expanded_frozen_vision_full` — pending (QOSGrpGRES, 4xH200 on gh125)

All 4-day-and-under jobs share the account's GPU-hour cap for jobs requesting
>48h wall time (`QOSGrpGRES`, observed empirically, not documented anywhere
official found so far) — pending jobs above will start as running ones free
GPUs, no action needed.

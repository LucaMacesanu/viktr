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
| `yor_icl_victr_vision_value_expanded` | `Pi0VictrConfig` | vision_value | never run | 50,000 | job 16469185 **FAILED** in 11min (2026-08-27) | crashed on `FileNotFoundError: .../vision_value/1190_images.npy` — the `vision_value` retrieval-context precompute (job 16442521) was incomplete: shard0 hit its 5:45:00 time limit mid-episode, leaving 19/1,784 episodes (1154–1195, with gaps) unprecomputed. Fixed by resubmitting `shells/slurm/victr_precompute_expanded_vision_value_job.sh` (job 16509888, 2026-08-28) — it skips/resumes already-done episodes so only needed to redo the 19. Once `outputs/victr/retrieval_context_expanded/vision_value/` has 7,136 files (1,784×4), re-launch training. **Note:** the failed run already created `checkpoints/yor_icl_victr_vision_value_expanded/.../` (just `wandb_id.txt`, no real step) — per `openpi/training/checkpoints.py:initialize_checkpoint_dir`, resubmitting without `--overwrite` or `--resume` raises `FileExistsError`; use `--overwrite` (as the other requeued arms above do) since nothing was actually checkpointed. |
| `yor_icl_fast_victr_vision_expanded` | `Pi0FastVictrConfig` (discrete FAST, autoregressive) | vision | 20,000 | 50,000 | **pending** (`QOSGrpGRES`), job 16511278 (`--resume`) | **not affected** by the normalization bug — entire loss is discrete FAST-token cross-entropy, already protected by the pre-existing tokenizer clip (see above). Original job 16429151 reached ~21.9k/50k steps (last checkpoint saved: 20,000) then was cancelled 2026-08-27T18:12 (`CANCELLED by 4539934`, not a crash) — same node (gh125) as `yor_icl_pi05_expanded_frozen_vision`'s 4×H200 reservation, which started ~4h later; looks like it was killed to free node capacity, not for a bug. Resubmitted 2026-08-28 with `--resume` (not `--overwrite` — real checkpoints exist, unlike the vision_value case above) to continue from step 20,000 to the original 50,000 target. |
| `yor_icl_pi05_expanded_frozen_vision` | `Pi0Config`, `freeze_filter=PathRegex(".*img.*")` (SigLip vision tower frozen, Gemma LM + action expert hot) | – | never run | 70,000 | **pending**, job 16476252 (`QOSGrpGRES`) | new config added post-fix, never hit the bug. 4×H200 (`gres=gpu:h200:4`, 64cpu/800G), global batch 256, `lr_schedule` decay_steps=70,000 to match num_train_steps (user-confirmed choice — see [[feedback_no_lerobot_pi05_training]] sibling doc for the annealing discussion that motivated matching decay_steps to the actual run length). Reuses `yor_icl_pi05_expanded`'s norm_stats.json (copied, same convention as the VICTR expanded arms). Auto-scheduled onto reserved node `gh125` (reservation `jz4725-h200`, MAGNETIC, no explicit `--reservation` flag needed). |
| `yor_icl_fast_victr_vision_interp_expanded` | `Pi0FastVictrConfig`, `use_action_interpolation=True` (RICL baseline, `notes/action_interpolation.md`) | vision | never run | 50,000 | job 16520341 **FAILED**, requeued as **16526464** (`exp_name=full`, `--overwrite`) | New arm, first ever real (non-synthetic) exercise of this config. Pre-launch validation found and fixed a real bug in `yor_retrieval.py`'s `_pool_max_vision_distance`: it materialized a full `N x N x 768` elementwise-difference tensor to find the per-task max pairwise DINOv2 distance — ~49GB for the largest task pool (`sort_the_items_into_their_containers`, ~4k chunks) for zero benefit. Rewrote via `||a-b||^2=||a||^2+||b||^2-2a.b` (an `N x N` matrix instead), confirmed fixed against real pool data. Also fixed a missing `norm_stats.json` (copied from `yor_icl_fast_victr_vision_expanded`'s, same dataset/action space, same convention as the other VICTR expanded arms). **Actually FAILED in 8min (2026-08-28T16:32-16:40)**: `RetrievalContextInputs` looked up the pool file by `data["prompt"]`, which by then holds the *overridden* display prompt (`YOR_EXPANDED_TASK_PROMPT_OVERRIDES`) rather than the raw task string the pool `.pkl` files are named from — every one of the 8 overridden tasks slug-mismatched. Only this config's live-retrieval path (`precomputed_context_dir=None`) ever hits that lookup; every other retrieval arm takes the precomputed episode-index array-lookup branch instead. Fixed (2026-08-28) by threading the inverse of `task_prompt_overrides` into `RetrievalContextInputs.prompt_to_task_overrides`; verified all 8 overridden tasks now resolve to their real pool files. See `notes/action_interpolation.md` for the full writeup. **Still open, not yet validated by a real run**: `yor_retrieval.py`'s `_load_pool_cached` caches every distinct task's full pool (images included) forever per dataloader worker (`lru_cache(maxsize=None)`) — total pool corpus is 29GB across 31 tasks, so 8 workers each eventually touching most tasks could reach ~232GB, tight against this job's 400G allocation. The prior job crashed before ever reaching this risk (8min in, dataloader workers barely warmed up), so it remains untested. Since the checkpoint dir has no real steps (`wandb_id.txt` only), a requeue needs `--overwrite`. |

## Job IDs (as of 2026-08-27, post-fix requeue)

- 16371057 — `yor_icl_pi05_rtc` — running
- 16429151 — `yor_icl_fast_victr_vision_expanded_full` — running
- 16468973 — `yor_icl_pi05_expanded_full` — pending (QOSGrpGRES / node availability)
- 16468974 — `yor_icl_ki_expanded_subtask_full` — pending
- 16468975 — `yor_icl_victr_value_expanded_full` — pending
- 16468976 — `yor_icl_victr_vision_expanded_full` — running
- 16469185 — `yor_icl_victr_vision_value_expanded_full` — pending
- 16476252 — `yor_icl_pi05_expanded_frozen_vision_full` — pending (QOSGrpGRES, 4xH200 on gh125)
- 16518901 — `yor_icl_fast_victr_vision_interp_expanded_dryrun` — **FAILED** in 2:30 (missing norm_stats.json), fixed, superseded by 16520333
- 16520333 — `yor_icl_fast_victr_vision_interp_expanded_dryrun` — cancelled by user before starting (skipped dry run, went straight to full run)
- 16520341 — `yor_icl_fast_victr_vision_interp_expanded_full` — **FAILED** in 8min (2026-08-28), pool-lookup bug (prompt override vs. raw task string), fixed same day
- 16526464 — `yor_icl_fast_victr_vision_interp_expanded_full` — requeued 2026-08-28 with `--overwrite` post-fix, pool-cache memory risk still untested going into this 48h allocation (user chose to skip a dry run again)

All 4-day-and-under jobs share the account's GPU-hour cap for jobs requesting
>48h wall time (`QOSGrpGRES`, observed empirically, not documented anywhere
official found so far) — pending jobs above will start as running ones free
GPUs, no action needed.

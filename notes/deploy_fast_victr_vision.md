# Deploying pi0-FAST-VICTR-vision (`yor_icl_fast_victr_vision_expanded`)

There was no existing deploy doc for any `yor_icl_*` policy — this covers the
FAST+VICTR "vision" arm specifically (openpi's discrete-token,
retrieval-conditioned pi0-FAST backend, retrieval_metric="vision"). See
`notes/training_runs.md` for how this checkpoint was trained and how it
compares to the other arms.

## What this policy needs at inference time

`openpi.models.pi0_fast_victr.Pi0FastVictrConfig` conditions each query on
`num_context_chunks` (=1) demonstration chunks retrieved by DINOv2 visual
similarity (`openpi.policies.yor_retrieval`), in addition to the usual
image/state/prompt inputs. Two things fall out of that which don't apply to a
plain pi05/pi0-FAST policy:

1. **A local retrieval pool.** `assets/victr_icl_pool_expanded_224/` — one
   `.pkl` per task (31 files, one per task in the expanded training set).
   Retrieval looks the pool file up by `task_slug(prompt)` (alnum-lowercased,
   truncated to 80 chars), so **the `prompt` you send must exactly match one
   of the 31 on-disk task strings** (list them with
   `ls third_party/openpi/assets/victr_icl_pool_expanded_224/` and reverse
   the slug, or see `viktr.data.icl_dataset`/`meta/tasks.parquet`), not the
   possibly-different text `YOR_EXPANDED_TASK_PROMPT_OVERRIDES` substitutes
   during training replay. See "Known limitation" below for the 8 tasks
   where those two strings actually differ.
2. **DINOv2 weights cached locally.** `yor_retrieval.load_dinov2()` calls
   `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`, expected to
   already be cached under `~/.cache/torch/hub` (same assumption training
   makes) — no network fetch expected on a compute node lacking outbound
   internet.

**Do not serve directly from the `yor_icl_fast_victr_vision_expanded`
training config** — its `precomputed_context_dir` makes retrieval do an O(1)
lookup keyed on `(episode_index, frame_index)`, fields that only exist on a
replayed dataset row. A live observation has neither, so
`RetrievalContextInputs.__call__` would `KeyError` on the very first
`policy.infer(...)` call. Use the sibling **`yor_icl_fast_victr_vision_expanded_serve`**
config instead (added alongside this doc) — identical model/checkpoint,
`precomputed_context_dir=None` so retrieval falls back to its live path
(DINOv2-embed the current camera frame, then nearest-neighbor search the
pool) which only needs `prompt` + the primary camera image, both present on
a real observation.

## 1. Start the policy server

From `third_party/openpi/` (needs its own `.venv`, not viktr's):

```bash
.venv/bin/python3 scripts/serve_policy.py policy:checkpoint \
    --policy.config=yor_icl_fast_victr_vision_expanded_serve \
    --policy.dir=checkpoints/yor_icl_fast_victr_vision_expanded/yor_icl_fast_victr_vision_expanded_full/15000 \
    --port=8000
```

`--policy.dir` is one of the saved step subdirs under
`checkpoints/yor_icl_fast_victr_vision_expanded/yor_icl_fast_victr_vision_expanded_full/`
(currently `5000`/`10000`/`15000` — training (job 16429151) is still running
toward 50,000 steps, saving every 5,000; check
`ls checkpoints/yor_icl_fast_victr_vision_expanded/yor_icl_fast_victr_vision_expanded_full/`
for the latest before deploying). Norm stats are loaded from the checkpoint's
own `assets/icl-dataset/norm_stats.json` (replicated at save time), not
recomputed, so serving always matches what that checkpoint was actually
trained with.

This starts a websocket server on `0.0.0.0:8000` (see
`third_party/openpi/docs/remote_inference.md` for the generic pattern this
follows). Run it on a GPU node — DINOv2 retrieval runs CPU-side but the
policy forward pass needs a GPU.

## 2. Query it from robot code

Install the client package once (minimal deps, meant to be embedded in
robot-side code separate from openpi's own env):

```bash
cd third_party/openpi/packages/openpi-client && pip install -e .
```

Then, per inference step:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="<server-host>", port=8000)

observation = {
    "observation.state": state,                    # (15,) float32, unnormalized -- lj0..lj6, rj0..rj6, lift
    "observation.images.zed": zed_image,            # (H, W, 3) uint8 -- primary camera, used for retrieval too
    "observation.images.fish0": left_wrist_image,   # (H, W, 3) uint8
    "observation.images.fish1": right_wrist_image,  # (H, W, 3) uint8
    "prompt": "clean the plate",                    # must match a assets/victr_icl_pool_expanded_224/*.pkl slug
}
action_chunk = client.infer(observation)["actions"]  # (30, 20): left_ee(7) | right_ee(7) | gripper(2) | base_vel(3) | lift(1)
```

Note the `observation.*` dotted keys (matching icl-dataset's own schema,
`openpi.policies.yor_policy.YorInputs`) rather than the `observation/foo`
slash convention other openpi example configs (DROID/LIBERO) use — this is
config-specific, not a generic openpi convention. Images can be `(H,W,C)`
uint8 or `(C,H,W)` float32 in `[0,1]`; resizing to 224x224 happens
server-side. As with any pi0 policy, call `infer` roughly every
`action_horizon` (30) steps and execute the chunk open-loop in between,
rather than every control tick.

Action dims 16:20 (`base_vel.vx/vy/omega`, `lift_cmd`) will come back ≈0 —
this embodiment never actually commands them (see `yor_policy.YorInputs`'s
zeroing fix in `notes/training_runs.md`) — safe to ignore/drop them
downstream.

## Known limitation: 8 tasks' prompt text is ambiguous

`YOR_EXPANDED_TASK_PROMPT_OVERRIDES` (`training/config.py`) remaps 8 of the
31 tasks' on-disk text to different prompt text before the model ever sees
it (e.g. raw `"clean the plate"` → `"Grab the plate and the rag, and use the
rag to wipe the plate."`). During training, that override is applied
*before* `RetrievalContextInputs` runs (`PromptFromLeRobotTask` in
`training/data_loader.py`), so retrieval for those 8 tasks looks up the pool
by the *overridden* text — which doesn't match any of the 31 on-disk pool
filenames (verified: none of the 8 overrides' slugs match an existing
`.pkl`). That should raise `FileNotFoundError` in `yor_retrieval.load_pool`
the first time one of those 8 tasks' episodes is sampled — but the
currently-running training job has been active for 20+ hours across 19k+
steps (all 31 tasks are in its 1,784-episode set, so it should have hit this
by now) without crashing. This is flagged but unresolved — before relying on
inference for these 8 tasks specifically (`clean the plate`,
`sort the items into their containers`, both `hit the yellow cube...`
tasks, and all four `pick up the orange cube...` variants), verify locally
which prompt string (raw vs. override) actually retrieves a sane context at
serve time. The other 23 tasks have no override and are unaffected.

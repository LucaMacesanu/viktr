# Deploying a VICTR retrieval-augmented policy (pi0.5 backend)

Runbook for standing up serving for a `yor_icl_victr_{vision,value,vision_value}_*`
arm (`openpi.models.pi0_victr.Pi0Victr`) on a machine that has never touched
this project before. Scope: pi0.5-backend VICTR only — pi0-FAST/RICL arms
are not pursued in this project.

## What you're deploying

`Pi0Victr` conditions each query on `num_context_chunks` demonstration
chunks retrieved by nearest-neighbor search (`openpi.policies.yor_retrieval`)
against a precomputed per-task pool, on top of the usual image/state/prompt
inputs a plain pi0.5 policy takes.

## Prerequisites

| Needs | Why |
|---|---|
| GPU + CUDA driver compatible with `jax[cuda12]` | policy forward pass |
| Python 3.12, `uv` | openpi's venv |
| Outbound internet, or a pre-populated cache (step 5) | DINOv2 weights (~380MB) |
| rsync/scp from the training machine | the retrieval pool (step 4, 15-34GB) |

**Not needed at serve time**: `ffmpeg7_shim`, video decode. Serving reads
live camera frames as plain numpy and the pool's images are already-decoded
pickled arrays — nothing touches a video file.

## 1. Clone

```bash
git clone --recurse-submodules <viktr-repo-url>
# or: git submodule update --init --recursive
```

Check `.gitmodules` for `third_party/openpi`'s URL — this project vendors
local model/policy code that only exists on its own fork, not upstream
`Physical-Intelligence/openpi`.

## 2. Set up openpi's venv

```bash
cd third_party/openpi && uv sync
```

## 3. Get the checkpoint

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="lair-nyu/<config-name>-step<N>")
```

Only `params/` + `assets/` are uploaded (no `train_state/`) — this is
exactly the layout `--policy.dir` expects, no reshuffling needed. Norm stats
load from the checkpoint's own `assets/` automatically — no separate step.

## 4. Copy the retrieval pool

**Not in git, not in the HF checkpoint** — budget real transfer time.
Per-task `.pkl` files under `assets/<pool_dir>/`; check the target arm's
`TrainConfig.data.pool_dir` in `config.py` (all three vision/value/
vision_value arms currently share `assets/victr_icl_pool_canonical_224`,
~15GB).

```bash
rsync -avP <training-host>:/path/to/viktr/third_party/openpi/assets/<pool_dir>/ \
    third_party/openpi/assets/<pool_dir>/
```

If transfer isn't possible, it can be rebuilt via
`scripts/build_retrieval_pool.py` + `resize_pool_images.py`, but that needs
the full `icl-dataset` with videos (hundreds of GB) and hours of DINOv2
embedding — a last resort, not routine.

## 5. DINOv2 weights

`yor_retrieval.load_dinov2()` calls `torch.hub.load("facebookresearch/dinov2",
"dinov2_vitb14")`, cached under `~/.cache/torch/hub`. With internet, this
happens automatically on first use. Air-gapped: copy that cache dir (repo +
`checkpoints/dinov2_vitb14_pretrain.pth`) from a machine that has internet.

## 6. Use (or add) the arm's `_serve` twin config

**Never serve directly from a config with `data.precomputed_context_dir`
set** — that does an O(1) lookup keyed on `(episode_index, frame_index)`,
which only exists on a replayed dataset row, not a live observation; a real
`policy.infer(...)` call will `KeyError`. Use/add a `<name>_serve` twin:
identical `model`/`weight_loader`, `precomputed_context_dir=None` so
retrieval falls back to its live path (DINOv2-embed the current frame, then
search the pool from step 4).

## 7. Start the server

```bash
cd third_party/openpi
.venv/bin/python3 scripts/serve_policy.py policy:checkpoint \
    --policy.config=<name>_serve \
    --policy.dir=<path from step 3> \
    --port=8000
```

## 8. Query it

```bash
cd third_party/openpi/packages/openpi-client && pip install -e .
```

```python
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy(host="<server-host>", port=8000)
action_chunk = client.infer(observation)["actions"]
```

Observation shape/keys are arm-specific — check the target config's `data=`
block and the matching `YorInputs*.__call__` in `yor_policy.py` rather than
assuming. Constant across arms: dotted `observation.*` keys (icl-dataset's
own schema), images as `(H,W,C)` uint8 or `(C,H,W)` float32 `[0,1]`
(resized to 224x224 server-side), `prompt` as plain text. Call `infer`
roughly every `action_horizon` (30) steps, execute open-loop in between.

## Gotchas

- **Prompt must match a pool filename slug.** Retrieval looks up the pool
  by `task_slug(prompt)` — must exactly match an on-disk `.pkl` name
  (`ls assets/<pool_dir>/`), not necessarily the training-time prompt text.
- **`task_prompt_overrides` can break that match.** Some tasks' on-disk
  text gets remapped to different prompt text before the model sees it
  (`YOR_EXPANDED_TASK_PROMPT_OVERRIDES` in `config.py`) — if the override
  doesn't correspond to any on-disk pool filename, retrieval fails for that
  task. Verify locally with the exact prompt string before relying on it.
- **`value`/`vision_value` retrieval needs `key_values`** in the pool
  (`pool["key_values"] is not None`) — `vision`-only pools may lack it.
- **GPU is for the policy forward pass only.** DINOv2 embedding and
  nearest-neighbor search both run CPU-side in the request path.

## What's not deployable yet

`notes/context_encoding_redesign.md` describes a redesigned per-chunk
context encoding (state via `state_proj`, action via truncated FAST tokens,
3 cameras/chunk) that's measured and verified feasible but **not
implemented** — no arm trains or serves with it yet.
`assets/victr_icl_pool_canonical_3cam_224/` already carries the extra
camera views and truncated action tokens, but `pi0_victr.py`/
`yor_retrieval.py` don't read those fields yet — pointing a `_serve` config
at that pool today just reads it the same as the existing 1-camera pools.

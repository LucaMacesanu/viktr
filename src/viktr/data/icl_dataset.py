"""Adapts `icl-dataset` (the real bimanual YOR data, vendored setup in
`third_party/nyu-finger-robot`) into the per-episode arrays
`viktr.retrieval.chunk_dictionary.build_chunk_dictionary` expects, plus
progress-value annotation and the train/pool/test episode split.

Progress-value annotation defaults to the RoboDopamine value model's real,
per-frame estimates (`robodopamine_value_at`/`robodopamine_values`, reading
`meta/value_estimates/*.json`) rather than the paper's oracle t/(T-1) formula
(`oracle_value_at`/`oracle_values`, Eq. 7) -- the latter remain available as a
fallback/ablation for episodes RoboDopamine doesn't cover.

Mirrors `viktr.data.libero`'s two functions (`episodes_for_task`,
`load_episode_arrays`), but icl-dataset carries a per-episode `tasks` column
directly in `meta/episodes` (a 1-element list), unlike LIBERO's per-frame-only
`task_index`, so task resolution here is a plain parquet filter rather than a
per-frame scan. Episode exclusion reads `icl_dataset_v2_excluded_episodes.json`
(packaged alongside this module -- see `_excluded_episode_indices`), the Hub
dataset's curated `keep` field, which is a strict superset of the on-disk
meta/excluded_episodes.json that `nyu-finger-robot/train.py`'s `filter_excluded`
reads: it also drops excess *successful* episodes per task to balance per-task
episode counts. So the episode set used here can be a strict subset of what an
unpatched nyu-finger-robot config with the same filter_tasks would see, for
tasks the Hub update rebalanced.
"""

from __future__ import annotations

import functools
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_ID = "icl-dataset"
DEFAULT_ROOT = Path("/scratch/lim2045/icl_ws/icl-dataset")

# The 4-task left-arm pnp subset this whole VICTR-on-icl-dataset experiment is
# scoped to (matches nyu-finger-robot/configs/yor-pi05-ablation-*.yaml's
# filter_tasks exactly, so the pi05 baseline and the 3 VictrPolicy arms are
# trained on the same underlying episode pool). Shared by build_retrieval_pool.py,
# train_victr.py, and eval_victr_offline.py.
TASKS = [
    "pick up the eggplant and place it on the plate with the left arm",
    "pick up the pumpkin and place it on the plate with the left arm",
    "pick up the red chilli and place it on the plate with the left arm",
    "pick up the orange cube and place it on the plate with the left arm on a cluttered table",
]
# Static head camera used to key the DINOv2 retrieval index and as the sole
# camera view in retrieved context chunks (pi05_context.py's
# _prepare_context_images only ever reads one camera per chunk). The query's
# own multi-view input (all 3 cameras) is unaffected -- this only controls
# retrieval keying and what context frames look like.
PRIMARY_CAMERA = "observation.images.zed"
CONTEXT_CHUNK_SIZE = 10  # L in the paper's Eq. 6 -- VictrConfig.context_chunk_size default
# pi0.5's own flow-matching action horizon (VictrConfig.chunk_size/n_action_steps), NOT
# the retrieval context chunk size above -- unrelated concepts that happen to both be
# called "chunk". Used to build the `delta_timestamps` LeRobotDataset needs to return a
# full ACTION_CHUNK_SIZE-length action sequence per sample (PI05Config.action_delta_indices
# is list(range(chunk_size)); without this, embed_suffix's att_masks (hardcoded to
# config.chunk_size) mismatches the actual per-sample action length and crashes
# make_att_2d_masks).
ACTION_CHUNK_SIZE = 30


def _episode_task(value) -> str | None:
    """Normalize the episode `tasks` cell (a 1-element array/list) to a string."""
    if isinstance(value, str):
        return value
    if value is None or len(value) == 0:
        return None
    return str(value[0])


def _read_episode_meta(root: Path) -> pd.DataFrame:
    """Concatenates every `meta/episodes/chunk-*/file-*.parquet` shard (lerobot v3
    splits episode metadata across multiple shards; icl-dataset has several)."""
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no meta/episodes/chunk-*/file-*.parquet under {root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _excluded_episode_indices(root: Path) -> set[int]:
    """Reads icl_dataset_v2_excluded_episodes.json (packaged alongside this module,
    not read from `root`) -- extracted from the icl-dataset Hub repo's `keep` field
    (huggingface.co/datasets/adityx23/icl-dataset, commit 5948ba4), which supersedes
    the on-disk meta/excluded_episodes.json that `nyu-finger-robot/train.py`'s
    `filter_excluded` reads (fail/invalid only, 434 episodes). `keep=False` is a
    strict superset: it additionally drops excess *successful* episodes per task to
    balance per-task episode counts, trimming the longest-duration outliers first
    (e.g. 'sort the items into their containers' 100->50 valid episodes). See that
    file's `note` field for details. This means viktr's episode set can be a strict
    subset of what an unpatched nyu-finger-robot config with the same filter_tasks
    would see, for tasks the Hub update rebalanced."""
    excl_file = Path(__file__).parent / "icl_dataset_v2_excluded_episodes.json"
    return set(json.loads(excl_file.read_text())["excluded_episodes"])


def episodes_for_task(task: str, root: Path = DEFAULT_ROOT) -> list[int]:
    """Successful episode indices for `task` (exact string match against
    meta/episodes' `tasks[0]`), with excluded_episodes.json's fail/invalid
    episodes dropped -- cheap: parquet metadata only, no video decode."""
    eps = _read_episode_meta(root)
    excluded = _excluded_episode_indices(root)
    task0 = eps["tasks"].apply(_episode_task)
    matches = eps.loc[(task0 == task) & ~eps["episode_index"].isin(excluded), "episode_index"]
    if matches.empty:
        valid = sorted(t for t in eps["tasks"].apply(_episode_task).unique() if t)
        raise ValueError(f"task {task!r} not found (or has 0 surviving episodes); valid tasks include: {valid[:10]}...")
    return sorted(int(e) for e in matches)


def all_tasks(root: Path = DEFAULT_ROOT) -> list[str]:
    """Every task string with >=1 surviving (non-excluded) episode in the full
    dataset, not just the 4-task TASKS subset above -- same filtering
    episodes_for_task applies, just without the single-task restriction. Used by
    build_retrieval_pool.py's --tasks=all and scripts/precompute_retrieval_context.py
    to cover the whole icl-dataset rather than the pnp subset."""
    eps = _read_episode_meta(root)
    excluded = _excluded_episode_indices(root)
    task0 = eps.loc[~eps["episode_index"].isin(excluded), "tasks"].apply(_episode_task)
    return sorted(t for t in task0.unique() if t)


# Tasks excluded from the "expanded" dataset -- viktr's second, broader VICTR
# experiment beyond the original 4-task pnp subset (TASKS) above: assembling
# gears, opening the notebook, anything going into grocery bags, and octagon
# stacking.
EXPANDED_TASKS_EXCLUDED = {
    "assemble the two gears",
    "open the notebook",
    "put the green pepper into the grocery bag",
    "put the red chilli into the grocery bag",
    "stack the colored octagons",
}


def expanded_tasks(root: Path = DEFAULT_ROOT) -> list[str]:
    """all_tasks() minus EXPANDED_TASKS_EXCLUDED -- the broader task set for the
    second VICTR experiment (31 of the full dataset's 36 tasks). See
    EXPANDED_PROMPT_OVERRIDES for the language-prompt rewrites that go with this
    set (clutter mention removed, long-horizon sort task's prompt replaced)."""
    return [t for t in all_tasks(root=root) if t not in EXPANDED_TASKS_EXCLUDED]


# Language-prompt overrides for expanded_tasks(): keyed by the *raw* task string
# (as stored in meta/episodes / meta/tasks.parquet, and what retrieval pools and
# train/pool/test splits stay keyed on) -> the prompt text actually shown to the
# policy. Episode selection and retrieval pools are unaffected -- only the
# language-conditioning text differs. Four kinds of rewrite:
#   - "orange cube ... cluttered table" is corrected, not merged: the icl-dataset
#     Hub repo (adityx23/icl-dataset, commit 5948ba4) fixed a mislabel -- these
#     episodes' actual destination is the box, not the plate (the on-disk raw
#     task string here still reads "...place it on the plate...", since we don't
#     mutate the dataset files themselves; only the shown-to-the-policy prompt is
#     corrected here, same mechanism used for every other override in this dict).
#     Earlier this override merged the cluttered/uncluttered pair into one
#     clutter-free prompt on the assumption they were the same skill differing
#     only by an irrelevant visual attribute -- now known wrong, since cluttered
#     and uncluttered have genuinely different destinations (box vs. plate), so
#     they're left as two distinct prompts instead of collapsed into one (doing
#     so would teach the same prompt two different target behaviors).
#   - "orange cube ... uncluttered table" just drops the now-irrelevant clutter
#     mention (that variable never changes the desired action, so leaving it in
#     just splits an already-small demo pool across two labels for identical
#     behavior).
#   - "hit the yellow cube..." tasks are reworded (tool via "using", arm via
#     "with") so every task ends in the same "with the left/right arm" arm-
#     qualifier template instead of a one-off "using the left/right arm" --
#     consistent-language-conditioned-IL literature (e.g. CALVIN's ablations)
#     finds a single syntactic template per semantic attribute grounds better
#     than multiple surface forms for the same fact.
#   - "sort the items..." (long-horizon) and "clean the plate" (originally
#     underspecified -- no object/tool named) are replaced with explicit,
#     fully-specified instructions.
EXPANDED_PROMPT_OVERRIDES: dict[str, str] = {
    "pick up the orange cube and place it on the plate with the left arm on a cluttered table": (
        "pick up the orange cube and place it in the box with the left arm"
    ),
    "pick up the orange cube and place it on the plate with the left arm on an uncluttered table": (
        "pick up the orange cube and place it on the plate with the left arm"
    ),
    "pick up the orange cube and place it on the plate with the right arm on a cluttered table": (
        "pick up the orange cube and place it in the box with the right arm"
    ),
    "pick up the orange cube and place it on the plate with the right arm on an uncluttered table": (
        "pick up the orange cube and place it on the plate with the right arm"
    ),
    "hit the yellow cube with the mallet using the left arm": (
        "hit the yellow cube using the mallet with the left arm"
    ),
    "hit the yellow cube with the mallet using the right arm": (
        "hit the yellow cube using the mallet with the right arm"
    ),
    "clean the plate": ("Grab the plate and the rag, and use the rag to wipe the plate."),
    "sort the items into their containers": (
        "Place the markers in the cup, the cubes in the box, and the food items on the plate."
    ),
}


def episode_length(episode_index: int, root: Path = DEFAULT_ROOT) -> int:
    return episode_lengths(root)[episode_index]


def episode_lengths(root: Path = DEFAULT_ROOT) -> dict[int, int]:
    """episode_index -> length for every episode in the dataset. Bulk version of
    episode_length -- call this once and reuse the dict rather than re-reading
    meta/episodes per lookup (train_victr.py needs this per training example)."""
    eps = _read_episode_meta(root)
    return dict(zip(eps["episode_index"].astype(int), eps["length"].astype(int), strict=True))


def oracle_value_at(frame_index: int, episode_length: int) -> float:
    """Scalar version of oracle_values -- frame_index/(episode_length-1) in [0,1],
    Eq. 7 of the VICTR paper (see oracle_values' docstring for the [-1,0]->[0,1]
    remap note). Used per-query-frame in the training loop; oracle_values below is
    the vectorized per-episode version used when building the retrieval pool."""
    if episode_length <= 1:
        return 1.0
    return float(frame_index) / (episode_length - 1)


def oracle_values(episode_length: int) -> np.ndarray:
    """Linear task-progress value per frame, Eq. 7 of the VICTR paper (notes/vktr.pdf,
    Sec III-C): frame t of a successful T-frame episode -> t/(T-1). Remapped from the
    paper's literal [-1, 0] to [0, 1] to match this codebase's existing convention
    (viktr.value.base.ValueEstimator's docstring, Chunk.value) -- a harmless affine
    reparameterization, since retrieval only ever consumes |value difference|.
    Every episode this is called on is already known-successful (icl_dataset.
    episodes_for_task / build_splits only select the dataset's `success` episodes),
    matching the paper's oracle definition, which is only defined for successes."""
    if episode_length <= 1:
        return np.ones(max(episode_length, 1), dtype=np.float32)
    return (np.arange(episode_length, dtype=np.float32) / (episode_length - 1)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _value_estimate_index(root: str) -> dict[int, Path]:
    """episode_index -> RoboDopamine JSON path (meta/value_estimates/*.json).
    Filenames don't encode the global episode_index for most files (they're keyed
    by episode_uid instead), so the index is built by reading each file's
    "episode_index" field; a couple of files instead only carry `episode_<N>.json`
    naming with no such field, handled via that filename pattern as a fallback."""
    ve_dir = Path(root) / "meta/value_estimates"
    index: dict[int, Path] = {}
    for p in sorted(ve_dir.glob("*.json")):
        d = json.loads(p.read_text())
        ei = d.get("episode_index")
        if ei is None:
            m = re.match(r"episode_(\d+)\.json$", p.name)
            ei = int(m.group(1)) if m else None
        if ei is not None:
            index[int(ei)] = p
    return index


@functools.lru_cache(maxsize=None)
def _robodopamine_curve(root: str, episode_index: int) -> np.ndarray:
    """Per-frame [0,1] progress curve for one episode, linearly interpolated from
    the RoboDopamine value model's sparse (frame_index, progress in [0,100]) points
    (meta/value_estimates/*.json) -- a higher-fidelity substitute for oracle_values'
    linear t/(T-1) interpolation, used when real value annotations are available."""
    index = _value_estimate_index(root)
    if episode_index not in index:
        raise KeyError(f"no RoboDopamine value estimate for episode_index={episode_index} under {root}")
    d = json.loads(index[episode_index].read_text())
    points = sorted(d["points"], key=lambda pt: pt["frame_index"])
    xs = np.array([pt["frame_index"] for pt in points], dtype=np.float64)
    ys = np.array([pt["progress"] for pt in points], dtype=np.float64) / 100.0
    length = episode_length(episode_index, root=Path(root))
    frames = np.arange(length, dtype=np.float64)
    curve = np.interp(frames, xs, ys, left=ys[0], right=ys[-1])
    return curve.astype(np.float32)


def robodopamine_values(episode_index: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    """Vectorized per-episode version of robodopamine_value_at -- used when
    building the retrieval pool (load_episode_arrays' "values" key)."""
    return _robodopamine_curve(str(root), episode_index)


def robodopamine_value_at(episode_index: int, frame_index: int, root: Path = DEFAULT_ROOT) -> float:
    """Scalar per-query-frame RoboDopamine value, mirroring oracle_value_at's
    signature shape but keyed by episode identity too (the curve is per-episode,
    not a fixed formula) -- used for the query's own value at train/eval time."""
    return float(_robodopamine_curve(str(root), episode_index)[frame_index])


def load_episode_arrays(
    episode_indices: list[int],
    camera_keys: list[str],
    repo_id: str = DEFAULT_REPO_ID,
    root: Path = DEFAULT_ROOT,
) -> list[dict]:
    """Returns one dict per episode: {"episode_index", "images": {cam: (T,H,W,3)
    uint8}, "proprio": (T,15) float32, "actions": (T,20) float32, "values": (T,)
    float32}. `camera_keys` is required (not defaulted to all cameras, unlike
    viktr.data.libero's version) -- callers should pass just the retrieval
    `primary_camera` when building a chunk dictionary, since
    VictrPolicy._prepare_context_images only ever reads one camera per chunk
    (pi05_context.py) and storing all 3 views here would triple memory for
    nothing. Eagerly loads every requested frame into memory -- fine for the
    small pool/test splits this is meant for (tens of episodes), NOT for
    streaming full training data (use lerobot's own DataLoader for that)."""
    episode_indices = sorted(episode_indices)
    # A handful of icl-dataset episodes have a query/loaded-frame timestamp gap that
    # lands right on lerobot's default tolerance_s=1e-4 boundary (e.g. 0.0001000...1s,
    # a float-rounding hair over 1e-4), which raises FrameTimestampError under the
    # default. Doubling the tolerance absorbs that boundary noise without materially
    # loosening synchronization (still sub-millisecond).
    dataset = LeRobotDataset(repo_id, root=root, episodes=episode_indices, tolerance_s=2e-4)

    eps = _read_episode_meta(root)
    lengths = dict(zip(eps["episode_index"].astype(int), eps["length"].astype(int), strict=True))

    out = []
    offset = 0
    for ep_idx in episode_indices:
        length = lengths[ep_idx]
        frm, to = offset, offset + length
        offset = to
        images = {cam: [] for cam in camera_keys}
        proprio, actions = [], []
        for i in range(frm, to):
            item = dataset[i]
            for cam in camera_keys:
                frame = item[cam]
                arr = frame.permute(1, 2, 0).numpy() if hasattr(frame, "permute") else np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                images[cam].append(arr)
            proprio.append(np.asarray(item["observation.state"], dtype=np.float32))
            actions.append(np.asarray(item["action"], dtype=np.float32))
        out.append(
            {
                "episode_index": ep_idx,
                "images": {cam: np.stack(frames, axis=0) for cam, frames in images.items()},
                "proprio": np.stack(proprio, axis=0),
                "actions": np.stack(actions, axis=0),
                "values": robodopamine_values(ep_idx, root=root),
            }
        )
    return out


def build_splits(
    tasks: list[str],
    seed: int,
    root: Path = DEFAULT_ROOT,
    train_frac: float = 0.85,
    pool_frac: float = 0.10,
    test_frac: float = 0.05,
) -> dict[str, dict[str, list[int]]]:
    """Deterministic, per-task-stratified 85/10/5 train/pool/test split (paper Sec
    III-A(a): "we hold out 10% of demonstrations as a shared context pool... the
    remaining 90% supplying query pairs"; the 90% is further split 85/5 here into
    train-queries and a held-out offline-eval set, per the user's own choice, since
    the paper doesn't need a held-out set within meta-training itself -- it evaluates
    on a wholly separate 24-task suite instead (Sec IV-B), which we don't have)."""
    if abs(train_frac + pool_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {train_frac}+{pool_frac}+{test_frac}")
    rng = random.Random(seed)
    splits = {}
    for task in tasks:
        eps = episodes_for_task(task, root=root)
        rng.shuffle(eps)
        n = len(eps)
        n_pool = max(1, round(n * pool_frac))
        n_test = max(1, round(n * test_frac))
        n_train = n - n_pool - n_test
        if n_train < 1:
            raise ValueError(f"task {task!r} has only {n} episodes; too few for an 85/10/5 split")
        splits[task] = {
            "train": sorted(eps[:n_train]),
            "pool": sorted(eps[n_train : n_train + n_pool]),
            "test": sorted(eps[n_train + n_pool :]),
        }
    return splits


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def task_slug(task: str) -> str:
    """Filesystem-safe stem for a task string -- shared by build_retrieval_pool.py
    (writes one ChunkDictionary pickle per task) and train_victr.py (reads them)."""
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]

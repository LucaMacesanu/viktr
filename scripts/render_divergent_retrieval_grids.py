"""Shard 2 (single process, run after every scripts/find_divergent_retrievals.py shard
has finished) of the vision-vs-vision+value retrieval deep-dive: merges every shard's
per-frame diff_score, groups it by task, takes the top --top-k-per-task frames WITHIN
EACH TASK (not a single global top-K, which a first pass found just gets dominated by
whichever one task has the largest-magnitude disagreements), and for each one renders
a 3-row (top/zed, left/fish0, right/fish1 camera views) comparison grid PNG -- so a
human can eyeball exactly what each retrieval metric pulled in at the frames where
"vision" and "vision_value" disagreed most, with every one of the 20 canonical tasks
represented. PNGs are written into one subfolder per task. Finishes by zipping the
whole output folder for easy transfer off-cluster.

Columns, in order:
  1. Observation        -- the query frame's own 3 camera views (ground truth).
  2. Vision              -- what the production "vision" metric retrieved (DINOv2
                             L2 distance on the zed/top camera ONLY -- see
                             yor_retrieval.RetrievalContextInputs/chunk_dictionary.py:
                             pool key_embeddings are built from primary_camera alone).
  3. Vision (+wrist)      -- OPTIONAL (--wrist-vision-pool-dir): what a hypothetical
                             vision metric using all 3 cameras (zed+fish0+fish1) would
                             retrieve instead -- z-scores each camera's DINOv2 L2
                             distance SEPARATELY across this task's pool then sums
                             them into one fused score, nearest chunk wins.
  4. Value               -- what the production "value" metric retrieved.
  5. Vision+Value        -- what the production "vision_value" metric retrieved
                             (fixed z-score fusion of zed-only vision + value).
  6. Vision+Wrist+Value  -- OPTIONAL (--wrist-vision-pool-dir): the 3-camera fused
                             vision score above (itself already a sum of 3 separately
                             z-scored per-camera distances) is z-scored AGAIN into one
                             combined visual term -- so it sits on the same ~unit-
                             variance footing as a single z-scored distance, matching
                             the production vision_value fusion's convention -- then
                             summed with value's own z-scored distance. Nearest chunk
                             under that sum wins.

Columns 3 and 6 are standalone offline computations for this comparison only -- they
do NOT touch or recompute the production pools/precomputed context that live training
jobs (yor_icl_victr_{vision,value,vision_value}_canonical_ctx3cam) read from, and do
not change what "vision"/"vision_value" mean anywhere else in the codebase.

The observation column is decoded live from the icl-dataset LeRobotDataset (the
precomputed *_images.npy files only ever hold the RETRIEVED neighbor's images, never
the query frame's own) -- grouped by episode so each episode's dataset/video is opened
at most once no matter how many of the top-K frames come from it. The Vision/Value/
Vision+Value columns are read directly out of scripts/precompute_retrieval_context.py's
--context-camera-keys output (k=1, 3 camera views per chunk, already resized+padded to
224x224) via a plain array index -- no retrieval recomputation needed. The optional
wrist columns instead load scripts/build_pool_canonical_3cam.py's own raw pool file per
task (has fish0/fish1 images per chunk already) and use fish0/fish1 DINOv2 embeddings
cached under --wrist-vision-cache-dir (the pool's own key_embeddings field is zed-only,
carried over unchanged by that script -- see its module docstring; see
scripts/precompute_wrist_vision_embeddings.py for how the cache is built).

Must run under openpi's own venv (not viktr's) -- reuses LeRobotDataset, _parse_image,
resize_with_pad_torch, and (for the wrist columns) embed_frames/load_pool/
robodopamine_value_at/_value_distances so every embedding/distance is computed exactly
like the live/offline retrieval paths do it.

Usage (after all scripts/find_divergent_retrievals.py shards have written their
shard_<i>.npz under --shards-dir):

    third_party/openpi/.venv/bin/python3 scripts/render_divergent_retrieval_grids.py \
        --shards-dir outputs/victr/divergent_retrievals/shards \
        --all-episodes-json outputs/victr/icl_pool_canonical/all_episodes.json \
        --context-dir outputs/victr/retrieval_context_canonical_3cam_k1 \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset-fixed-obs \
        --out-dir outputs/victr/divergent_retrieval_grids \
        --top-k-per-task 10 \
        --wrist-vision-pool-dir third_party/openpi/assets/victr_icl_pool_canonical_3cam_224 \
        --wrist-vision-cache-dir outputs/victr/wrist_vision_embeddings
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party/openpi/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from openpi.policies.yor_retrieval import (  # noqa: E402
    ChunkDictionary,
    _parse_image,
    _value_distances,
    embed_frames,
    load_pool,
    robodopamine_value_at,
    task_slug,
)
from openpi.shared.image_tools import resize_with_pad_torch  # noqa: E402

import torch  # noqa: E402

CELL = 224
LABEL_W = 100
HEADER_H = 34
TITLE_H = 40
CAMERA_KEYS = ("observation.images.zed", "observation.images.fish0", "observation.images.fish1")
ROW_LABELS = ("Top (zed)", "Left (fish0)", "Right (fish1)")

# Column kinds:
#   ("context", metric)   -- read straight out of the precomputed --context-dir arrays.
#   ("wrist_vision",)     -- z-score each of zed/fish0/fish1's DINOv2 distance
#                            separately, sum, nearest chunk wins.
#   ("wrist_vision_value",) -- same 3-camera fused score, z-scored AGAIN, summed with
#                            value's own z-scored distance, nearest chunk wins.
BASE_COLUMNS: list[tuple[str, tuple]] = [
    ("Vision", ("context", "vision")),
    ("Value", ("context", "value")),
    ("Vision+Value", ("context", "vision_value")),
]
WRIST_COLUMNS: list[tuple[str, tuple]] = [
    ("Vision (+wrist)", ("wrist_vision",)),
    ("Vision+Wrist+Value", ("wrist_vision_value",)),
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _load_shards(shards_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shard_files = sorted(shards_dir.glob("shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"no shard_*.npz files under {shards_dir}")
    episode_cols, frame_cols, score_cols = [], [], []
    for f in shard_files:
        d = np.load(f)
        episode_cols.append(d["episode_index"])
        frame_cols.append(d["frame_index"])
        score_cols.append(d["diff_score"])
    return np.concatenate(episode_cols), np.concatenate(frame_cols), np.concatenate(score_cols)


def _observation_views(dataset: LeRobotDataset, frame_index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (raw_parsed_frames, resized_224_frames) for the query's 3 camera views --
    raw is what should be fed to embed_frames (matches how the live retrieval path
    embeds an unresized query frame), resized is what goes into the grid cell."""
    item = dataset[frame_index]
    raw = [_parse_image(item[cam]) for cam in CAMERA_KEYS]
    resized = resize_with_pad_torch(torch.from_numpy(np.stack(raw, axis=0)), CELL, CELL).numpy()
    return raw, [resized[i] for i in range(3)]


def _context_views(context_dir: Path, metric: str, episode_index: int, frame_index: int) -> list[np.ndarray]:
    arr = np.load(context_dir / metric / f"{episode_index}_images.npy", mmap_mode="r")
    frame = np.asarray(arr[frame_index, 0])  # (3, 224, 224, 3) uint8, cam order = CAMERA_KEYS
    return [frame[i] for i in range(3)]


def _wrist_pool_embeddings(
    pool_dir: Path, cache_dir: Path, task: str
) -> tuple[ChunkDictionary, np.ndarray, np.ndarray]:
    """Loads task's raw canonical-3cam pool and returns (pool, fish0_embeddings,
    fish1_embeddings) -- fish0/fish1 DINOv2 embeddings for EVERY chunk in the pool
    (needed to rank the query against the whole pool under the fused 3-camera metric),
    computed once and cached to cache_dir/<task_slug>.npz since embedding a whole
    task's pool (some are 5GB+, thousands of chunks) is the expensive part of this
    script and would otherwise redo on every re-run."""
    slug = task_slug(task)
    cache_path = cache_dir / f"{slug}.npz"
    pool = load_pool(pool_dir / f"{slug}.pkl")
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["fish0"]) == len(pool):
            return pool, cached["fish0"], cached["fish1"]
        print(f"WARNING: cache {cache_path} has {len(cached['fish0'])} rows but pool has {len(pool)} chunks -- recomputing", flush=True)

    fish0_frames = np.stack([c.images["observation.images.fish0"][0] for c in pool.chunks], axis=0)
    fish1_frames = np.stack([c.images["observation.images.fish1"][0] for c in pool.chunks], axis=0)
    fish0_embeddings = embed_frames(fish0_frames)
    fish1_embeddings = embed_frames(fish1_frames)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, fish0=fish0_embeddings, fish1=fish1_embeddings)
    return pool, fish0_embeddings, fish1_embeddings


def _zscore(d: np.ndarray) -> np.ndarray:
    return (d - d.mean()) / (d.std() + 1e-8)


def _wrist_camera_distances(
    pool: ChunkDictionary,
    fish0_embeddings: np.ndarray,
    fish1_embeddings: np.ndarray,
    raw_query_views: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-camera DINOv2 L2 distance (query vs. every chunk in this task's pool) for
    zed/fish0/fish1 -- the shared building block for both wrist-vision columns."""
    query_zed, query_fish0, query_fish1 = (embed_frames(raw[None])[0] for raw in raw_query_views)
    d_zed = np.linalg.norm(pool.key_embeddings - query_zed[None, :], axis=1)
    d_fish0 = np.linalg.norm(fish0_embeddings - query_fish0[None, :], axis=1)
    d_fish1 = np.linalg.norm(fish1_embeddings - query_fish1[None, :], axis=1)
    return d_zed, d_fish0, d_fish1


def _chunk_views(pool: ChunkDictionary, chunk_idx: int) -> list[np.ndarray]:
    chunk = pool.chunks[chunk_idx]
    return [chunk.images[cam][0] for cam in CAMERA_KEYS]


def _wrist_vision_views(d_zed: np.ndarray, d_fish0: np.ndarray, d_fish1: np.ndarray, pool: ChunkDictionary) -> list[np.ndarray]:
    """3-camera vision retrieval: z-scores the zed/fish0/fish1 distance arrays
    SEPARATELY across this task's pool, then sums them (same z-score-then-sum
    convention as yor_retrieval.fused_retrieve's vision+value fusion, extended from 2
    terms to 3) -- nearest chunk under the summed score wins. k=1, so no need for
    _topk's per-episode diversity constraint (irrelevant once only 1 chunk is kept)."""
    fused = _zscore(d_zed) + _zscore(d_fish0) + _zscore(d_fish1)
    return _chunk_views(pool, int(np.argmin(fused)))


def _wrist_vision_value_views(
    d_zed: np.ndarray, d_fish0: np.ndarray, d_fish1: np.ndarray, pool: ChunkDictionary, query_value: float
) -> list[np.ndarray]:
    """Vision(+wrist)+Value fusion: the 3-camera fused visual score (already a sum of
    3 separately-z-scored per-camera distances, so it has ~3x the variance of any one
    of them) is z-scored AGAIN to put it back on the same ~unit-variance footing as a
    single distance term -- matching how the production vision_value metric compares
    ONE z-scored vision term against ONE z-scored value term -- then summed with
    value's own z-scored distance. Nearest chunk under that sum wins."""
    visual_combined = _zscore(d_zed) + _zscore(d_fish0) + _zscore(d_fish1)
    visual_z = _zscore(visual_combined)
    val_z = _zscore(_value_distances(query_value, pool))
    return _chunk_views(pool, int(np.argmin(visual_z + val_z)))


def _render_grid(
    columns: list[list[np.ndarray]],
    column_names: tuple[str, ...],
    *,
    episode_index: int,
    frame_index: int,
    task: str | None,
    diff_score: float,
) -> Image.Image:
    """columns[col][row] -> (224, 224, 3) uint8 array, col order matches
    column_names, row order matches ROW_LABELS."""
    width = LABEL_W + CELL * len(column_names)
    height = TITLE_H + HEADER_H + CELL * len(ROW_LABELS)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"demo {episode_index}  |  frame {frame_index}  |  diff {diff_score:.1f}"
    if task:
        title += f"  |  {task}"
    draw.text((8, 8), title, fill="black", font=_font(18))

    for col_idx, name in enumerate(column_names):
        x = LABEL_W + col_idx * CELL
        draw.text((x + 8, TITLE_H + 6), name, fill="black", font=_font(16))

    for row_idx, name in enumerate(ROW_LABELS):
        y = TITLE_H + HEADER_H + row_idx * CELL
        draw.text((6, y + CELL // 2 - 8), name, fill="black", font=_font(14))

    for col_idx in range(len(column_names)):
        for row_idx in range(len(ROW_LABELS)):
            tile = Image.fromarray(columns[col_idx][row_idx])
            x = LABEL_W + col_idx * CELL
            y = TITLE_H + HEADER_H + row_idx * CELL
            canvas.paste(tile, (x, y))

    draw.rectangle([0, 0, width - 1, height - 1], outline="black", width=1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--all-episodes-json", required=True, type=Path)
    parser.add_argument("--context-dir", required=True, type=Path)
    parser.add_argument("--icl-dataset-root", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k-per-task", type=int, default=10)
    parser.add_argument(
        "--wrist-vision-pool-dir",
        type=Path,
        default=None,
        help="scripts/build_pool_canonical_3cam.py output dir (has fish0/fish1 images per chunk) -- when set, "
        "adds 'Vision (+wrist)' and 'Vision+Wrist+Value' columns (see module docstring). Does not modify this "
        "pool or any production output.",
    )
    parser.add_argument(
        "--wrist-vision-cache-dir",
        type=Path,
        default=None,
        help="where to cache each task's fish0/fish1 pool embeddings (required if --wrist-vision-pool-dir is set)",
    )
    args = parser.parse_args()
    if args.wrist_vision_pool_dir is not None and args.wrist_vision_cache_dir is None:
        parser.error("--wrist-vision-pool-dir requires --wrist-vision-cache-dir")

    columns_spec = [("Observation", ("observation",)), *BASE_COLUMNS]
    if args.wrist_vision_pool_dir is not None:
        # Interleave so wrist variants sit next to their non-wrist counterpart:
        # Observation, Vision, Vision (+wrist), Value, Vision+Value, Vision+Wrist+Value.
        columns_spec = [
            ("Observation", ("observation",)),
            ("Vision", ("context", "vision")),
            WRIST_COLUMNS[0],
            ("Value", ("context", "value")),
            ("Vision+Value", ("context", "vision_value")),
            WRIST_COLUMNS[1],
        ]
    column_names = tuple(name for name, _ in columns_spec)

    episode_col, frame_col, score_col = _load_shards(args.shards_dir)

    all_episodes: dict[str, list[int]] = json.loads(args.all_episodes_json.read_text())
    episode_to_task: dict[int, str] = {ep: task for task, eps in all_episodes.items() for ep in eps}

    # Group every scored frame by task, then take --top-k-per-task WITHIN each task --
    # a single global top-K (the first version of this script) just returns the one
    # task with the largest-magnitude vision/vision_value disagreements, starving
    # every other task of representation.
    by_task: dict[str, list[int]] = {}
    for i, ep in enumerate(episode_col):
        by_task.setdefault(episode_to_task[int(ep)], []).append(i)

    selected: list[tuple[str, int, int, int, float]] = []  # (task, rank_in_task, episode, frame, score)
    for task in sorted(all_episodes):
        idxs = by_task.get(task, [])
        if not idxs:
            print(f"WARNING: no scored frames found for task {task!r}", flush=True)
            continue
        idxs_arr = np.array(idxs)
        task_scores = score_col[idxs_arr]
        order = np.argsort(-task_scores)[: args.top_k_per_task]
        for rank, j in enumerate(order):
            i = idxs_arr[j]
            selected.append((task, rank, int(episode_col[i]), int(frame_col[i]), float(score_col[i])))
        print(
            f"task {task!r}: selected top {len(order)} of {len(idxs)} scored frames "
            f"(max={task_scores.max():.1f}, min={task_scores[order[-1]]:.1f})",
            flush=True,
        )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    # Grouped by (task, episode_index): a task's wrist-vision pool/embeddings (if
    # requested) are loaded once per task, and within a task each episode's
    # LeRobotDataset/video is opened at most once -- episodes never span tasks, so
    # this single sort key gives both groupings for free.
    by_task_episode = sorted(selected, key=lambda s: (s[0], s[2]))
    current_task: str | None = None
    current_episode: int | None = None
    current_dataset: LeRobotDataset | None = None
    wrist_state: tuple[ChunkDictionary, np.ndarray, np.ndarray] | None = None
    for n, (task, rank, episode_index, frame_index, diff_score) in enumerate(by_task_episode):
        if task != current_task:
            current_task = task
            if args.wrist_vision_pool_dir is not None:
                print(f"loading wrist-vision pool/embeddings for task {task!r}...", flush=True)
                wrist_state = _wrist_pool_embeddings(args.wrist_vision_pool_dir, args.wrist_vision_cache_dir, task)
        if episode_index != current_episode:
            current_dataset = LeRobotDataset(
                "icl-dataset", root=args.icl_dataset_root, episodes=[episode_index], tolerance_s=2e-4
            )
            current_episode = episode_index
        dataset = current_dataset

        raw_obs_views, resized_obs_views = _observation_views(dataset, frame_index)

        # zed/fish0/fish1 distances against this frame's task pool, computed once and
        # shared by both wrist columns (if requested) -- avoids embedding the query
        # frame's 3 camera views twice.
        wrist_distances = None
        if args.wrist_vision_pool_dir is not None:
            pool, fish0_embeddings, fish1_embeddings = wrist_state
            wrist_distances = _wrist_camera_distances(pool, fish0_embeddings, fish1_embeddings, raw_obs_views)

        columns = []
        for _, spec in columns_spec:
            kind = spec[0]
            if kind == "observation":
                columns.append(resized_obs_views)
            elif kind == "context":
                columns.append(_context_views(args.context_dir, spec[1], episode_index, frame_index))
            elif kind == "wrist_vision":
                columns.append(_wrist_vision_views(*wrist_distances, pool))
            elif kind == "wrist_vision_value":
                query_value = robodopamine_value_at(episode_index, frame_index, args.icl_dataset_root)
                columns.append(_wrist_vision_value_views(*wrist_distances, pool, query_value))
            else:
                raise ValueError(f"unknown column kind {kind!r}")

        grid = _render_grid(
            columns,
            column_names,
            episode_index=episode_index,
            frame_index=frame_index,
            task=task,
            diff_score=diff_score,
        )
        task_dir = args.out_dir / task_slug(task)
        task_dir.mkdir(exist_ok=True)
        out_path = task_dir / f"{rank:02d}_demo{episode_index}_frame{frame_index}.png"
        grid.save(out_path)
        print(f"[{n + 1}/{len(by_task_episode)}] wrote {out_path.relative_to(args.out_dir)} (diff={diff_score:.1f})", flush=True)

    zip_path = args.out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=args.out_dir)
    print(f"zipped {args.out_dir} -> {zip_path}")


if __name__ == "__main__":
    main()

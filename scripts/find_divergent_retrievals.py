"""Shard 1 of the vision-vs-vision+value retrieval deep-dive: scans every precomputed
context frame under a scripts/precompute_retrieval_context.py --context-camera-keys
output dir (outputs/victr/retrieval_context_canonical_3cam_k1/{vision,vision_value}/
*_images.npy) and scores how much the "vision" and "vision_value" metrics' retrieved
context differs at that frame -- pure pixel L1 distance between their (already
resized-and-padded) context images, mean over the 3 camera views x 224x224x3 pixels.
Same chunk retrieved by both metrics -> identical arrays -> score 0; a different
chunk -> a nonzero score, larger the more visually different the two retrieved
neighbors are. Pairs this with scripts/render_divergent_retrieval_grids.py, which
merges every shard's output, takes the global top-K by score, and renders the
side-by-side comparison grids.

Pure numpy (mmap reads of already-precomputed .npy arrays, no DINOv2/pool/dataset
access needed) -- CPU-only, shardable like precompute_retrieval_context.py itself.
Run under third_party/openpi/.venv only for convenience of sharing one venv with the
render step, not because this script needs anything from it.

Usage (one shard of a job array; --num-shards/--shard-index split EPISODES into
independent chunks, bin-packed by each episode's vision _images.npy file size so no
shard is stuck with a disproportionate share of the total bytes to read):

    third_party/openpi/.venv/bin/python3 scripts/find_divergent_retrievals.py \
        --all-episodes-json outputs/victr/icl_pool_canonical/all_episodes.json \
        --context-dir outputs/victr/retrieval_context_canonical_3cam_k1 \
        --out-dir outputs/victr/divergent_retrievals/shards \
        --num-shards 20 --shard-index 0

Each shard writes exactly one outputs/victr/divergent_retrievals/shards/shard_<i>.npz
with three parallel 1-D arrays (episode_index, frame_index, diff_score) covering
every frame of every episode assigned to that shard -- not just a truncated top-K --
so the merge step's global ranking is exact, not an approximation of shard-local
top-Ks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _assign_episodes_to_shards(episodes: list[int], context_dir: Path, num_shards: int) -> list[list[int]]:
    """Greedy longest-processing-time-first bin-packing by each episode's vision
    _images.npy file size (a direct proxy for that episode's frame count, i.e. how
    many bytes this script has to read for it) -- mirrors
    precompute_retrieval_context._assign_tasks_to_shards' reasoning, just keyed by
    episode instead of task."""
    sizes = [(context_dir / "vision" / f"{ep}_images.npy").stat().st_size for ep in episodes]
    order = sorted(range(len(episodes)), key=lambda i: sizes[i], reverse=True)
    bins: list[list[int]] = [[] for _ in range(num_shards)]
    bin_totals = [0] * num_shards
    for i in order:
        j = min(range(num_shards), key=lambda b: bin_totals[b])
        bins[j].append(episodes[i])
        bin_totals[j] += sizes[i]
    return bins


_WINDOW = 256


def score_episode(context_dir: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_indices, diff_scores) for one episode: mean absolute pixel
    difference between the "vision" and "vision_value" metrics' retrieved context
    images at every frame, over all camera views and pixels of the k=1 retrieved
    chunk. Reads both arrays via mmap and processes _WINDOW frames at a time (rather
    than casting the whole episode to int16 at once) -- bounds peak memory to one
    window regardless of episode length, matching precompute_retrieval_context.py's
    own frame_window_size convention; some episodes' images.npy are >800MB, ~1.6GB
    once upcast to int16, and this interactive/CPU environment's memory headroom
    shouldn't gate correctness."""
    vision = np.load(context_dir / "vision" / f"{episode_index}_images.npy", mmap_mode="r")
    vision_value = np.load(context_dir / "vision_value" / f"{episode_index}_images.npy", mmap_mode="r")
    if vision.shape != vision_value.shape:
        raise ValueError(
            f"episode {episode_index}: vision shape {vision.shape} != vision_value shape {vision_value.shape}"
        )
    num_frames = vision.shape[0]
    scores = np.empty(num_frames, dtype=np.float32)
    for start in range(0, num_frames, _WINDOW):
        end = min(start + _WINDOW, num_frames)
        diff = np.abs(vision[start:end].astype(np.int16) - vision_value[start:end].astype(np.int16))
        scores[start:end] = diff.reshape(end - start, -1).mean(axis=1)
    return np.arange(num_frames, dtype=np.int32), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-episodes-json", required=True, type=Path)
    parser.add_argument("--context-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    all_episodes: dict[str, list[int]] = json.loads(args.all_episodes_json.read_text())
    episodes = sorted({ep for eps in all_episodes.values() for ep in eps})

    shard_episodes = _assign_episodes_to_shards(episodes, args.context_dir, args.num_shards)[args.shard_index]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard_episodes)} episodes to score", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_episode_col: list[np.ndarray] = []
    all_frame_col: list[np.ndarray] = []
    all_score_col: list[np.ndarray] = []
    for i, episode_index in enumerate(shard_episodes):
        start = time.time()
        frame_indices, scores = score_episode(args.context_dir, episode_index)
        all_episode_col.append(np.full(len(frame_indices), episode_index, dtype=np.int32))
        all_frame_col.append(frame_indices)
        all_score_col.append(scores)
        elapsed = time.time() - start
        print(
            f"[{i + 1}/{len(shard_episodes)}] episode {episode_index}: {len(frame_indices)} frames, "
            f"max_score={scores.max():.2f}, {elapsed:.1f}s",
            flush=True,
        )

    out_path = args.out_dir / f"shard_{args.shard_index}.npz"
    np.savez(
        out_path,
        episode_index=np.concatenate(all_episode_col) if all_episode_col else np.zeros(0, dtype=np.int32),
        frame_index=np.concatenate(all_frame_col) if all_frame_col else np.zeros(0, dtype=np.int32),
        diff_score=np.concatenate(all_score_col) if all_score_col else np.zeros(0, dtype=np.float32),
    )
    print(f"shard {args.shard_index}: wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

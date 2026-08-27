"""Demonstration chunk dictionary construction (paper Sec III-B, Eq. 6).

Splits each demonstration into overlapping fixed-size chunks (50% stride) and
keys each chunk by the DINOv2 embedding of its first frame. This is the
key-value store both the vision-only metric and the fused vision+value
metric (retrieval/metrics.py, retrieval/fusion.py) query against.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from viktr.retrieval.embeddings import embed_frames, load_dinov2


@dataclass
class Chunk:
    task: str
    episode_index: int
    start_frame: int  # inclusive, index into the episode
    end_frame: int  # exclusive
    images: dict[str, np.ndarray]  # camera_key -> (L, H, W, 3) uint8, one entry per camera view
    proprio: np.ndarray  # (L, state_dim) float32
    actions: np.ndarray  # (L, action_dim) float32
    value: np.ndarray | None = None  # (L,) float32 — per-frame progress, filled in by a ValueEstimator


@dataclass
class ChunkDictionary:
    chunks: list[Chunk]
    key_embeddings: np.ndarray  # (num_chunks, EMBED_DIM) float32 — first-frame DINOv2 embedding per chunk
    key_values: np.ndarray | None = None  # (num_chunks,) float32 — first-frame value, once annotated

    def __len__(self) -> int:
        return len(self.chunks)


def chunk_starts(num_frames: int, chunk_size: int) -> list[int]:
    """50%-overlap sliding-window start indices, matching Eq. 6: (T/L)*2 - 1 chunks."""
    if num_frames < chunk_size:
        return []
    stride = max(chunk_size // 2, 1)
    return list(range(0, num_frames - chunk_size + 1, stride))


def split_episode_into_chunks(
    task: str,
    episode_index: int,
    images: dict[str, np.ndarray],
    proprio: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
    values: np.ndarray | None = None,
) -> list[Chunk]:
    num_frames = len(proprio)
    chunks = []
    for start in chunk_starts(num_frames, chunk_size):
        end = start + chunk_size
        chunks.append(
            Chunk(
                task=task,
                episode_index=episode_index,
                start_frame=start,
                end_frame=end,
                images={cam: arr[start:end] for cam, arr in images.items()},
                proprio=proprio[start:end],
                actions=actions[start:end],
                value=values[start:end] if values is not None else None,
            )
        )
    return chunks


def build_chunk_dictionary(
    episodes: list[dict],
    task: str,
    chunk_size: int,
    primary_camera: str,
    embed_model=None,
) -> ChunkDictionary:
    """episodes: one dict per demo, each with "episode_index", "images" (camera_key -> (T,H,W,3) uint8),
    "proprio", "actions", and optionally "values" (T,) float32 (see viktr.value.annotate). `primary_camera`
    selects which view's first frame keys each chunk (both for the DINOv2 embedding and, if annotated, value)."""
    model = embed_model if embed_model is not None else load_dinov2()
    all_chunks: list[Chunk] = []
    for ep in episodes:
        all_chunks.extend(
            split_episode_into_chunks(
                task=task,
                episode_index=ep["episode_index"],
                images=ep["images"],
                proprio=ep["proprio"],
                actions=ep["actions"],
                chunk_size=chunk_size,
                values=ep.get("values"),
            )
        )
    if not all_chunks:
        raise ValueError(f"no chunks produced for task={task!r}; check episode lengths vs chunk_size={chunk_size}")
    first_frames = np.stack([c.images[primary_camera][0] for c in all_chunks], axis=0)
    key_embeddings = embed_frames(first_frames, model=model)
    key_values = None
    if all(c.value is not None for c in all_chunks):
        key_values = np.array([c.value[0] for c in all_chunks], dtype=np.float32)
    return ChunkDictionary(chunks=all_chunks, key_embeddings=key_embeddings, key_values=key_values)


def save_pool(pool: ChunkDictionary, path: Path) -> None:
    # Write-to-temp-then-rename: an OOM kill (or any crash) mid pickle.dump leaves
    # a truncated file at `path` otherwise, which build_retrieval_pool.py's
    # out_path.exists() skip-check would then silently treat as a valid, already-
    # built pool on the next resume. os.replace is atomic on the same filesystem,
    # so `path` only ever exists as a complete file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(pool, f)
    os.replace(tmp_path, path)


def load_pool(path: Path) -> ChunkDictionary:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pools(pool_dir: Path, tasks: list[str], task_slug) -> dict[str, ChunkDictionary]:
    """task_slug: viktr.data.icl_dataset.task_slug, passed in rather than imported
    here to avoid a retrieval -> data import (data already imports retrieval)."""
    return {task: load_pool(pool_dir / f"{task_slug(task)}.pkl") for task in tasks}

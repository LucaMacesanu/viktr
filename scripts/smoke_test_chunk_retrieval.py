"""Smoke test for the demonstration chunk dictionary + vision-only retrieval
(paper Sec III-B/III-G), against a couple of real lerobot/libero episodes.

Usage: uv run python scripts/smoke_test_chunk_retrieval.py
"""

from __future__ import annotations

import numpy as np

from viktr.data.libero import DEFAULT_REPO_ID, episodes_for_task, load_episode_arrays
from viktr.retrieval.chunk_dictionary import build_chunk_dictionary, chunk_starts
from viktr.retrieval.embeddings import load_dinov2
from viktr.retrieval.metrics import vision_retrieve


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(DEFAULT_REPO_ID)
    task = meta.tasks.index[0]
    print(f"task: {task!r}")

    episode_indices = episodes_for_task(DEFAULT_REPO_ID, task)[:3]
    print(f"episodes for task: {episode_indices}")
    assert len(episode_indices) >= 2, "need at least 2 demos to build a meaningful chunk dictionary"

    episodes = load_episode_arrays(DEFAULT_REPO_ID, episode_indices)
    for ep in episodes:
        n = len(ep["proprio"])
        print(f"  episode {ep['episode_index']}: {n} frames, chunks(L=30)={len(chunk_starts(n, 30))}")

    model = load_dinov2()
    pool = build_chunk_dictionary(episodes, task=task, chunk_size=30, primary_camera=meta.camera_keys[0], embed_model=model)
    print(f"chunk dictionary size: {len(pool)}")

    # Query with the last chunk's first frame from a held-out episode's start: verify the nearest
    # neighbor found is a chunk from the SAME episode near frame 0 (sanity check on the metric).
    query_chunk = pool.chunks[0]
    query_embedding = pool.key_embeddings[0]
    result = vision_retrieve(query_embedding, pool, k=5)

    print("top-5 nearest chunks to chunks[0]:")
    for chunk, score in zip(result.chunks, result.scores):
        print(f"  ep={chunk.episode_index} frames=[{chunk.start_frame},{chunk.end_frame}) score={score:.4f}")

    assert result.chunks[0].episode_index == query_chunk.episode_index
    assert result.chunks[0].start_frame == query_chunk.start_frame
    assert np.isclose(result.scores[0], 0.0, atol=1e-3), "self-retrieval distance should be ~0"
    print("OK: self-retrieval sanity check passed.")


if __name__ == "__main__":
    main()

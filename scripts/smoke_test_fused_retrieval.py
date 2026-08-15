"""Smoke test for value + fused vision-value retrieval (paper Sec III-G),
using a trivial synthetic ValueEstimator so it doesn't depend on Robometer
being loaded — validates the chunk_dictionary/metrics/fusion wiring only.

Usage: uv run python scripts/smoke_test_fused_retrieval.py
"""

from __future__ import annotations

import numpy as np

from viktr.data.libero import DEFAULT_REPO_ID, episodes_for_task, load_episode_arrays
from viktr.retrieval.chunk_dictionary import build_chunk_dictionary
from viktr.retrieval.embeddings import load_dinov2
from viktr.retrieval.fusion import ValueFusionMLP
from viktr.retrieval.metrics import f_retrieve
from viktr.value.annotate import annotate_episode_values
from viktr.value.base import ValueEstimator


class LinearProgressEstimator(ValueEstimator):
    """Ground-truth-shaped stand-in: linear ramp 0 -> 1 over the episode, matching
    Eq. 7's relative-progress target. Only used to validate wiring."""

    def estimate(self, images: np.ndarray, instruction: str) -> float:
        return 1.0

    def estimate_episode(self, images: np.ndarray, instruction: str) -> np.ndarray:
        n = len(images)
        return np.linspace(0.0, 1.0, n, dtype=np.float32)


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(DEFAULT_REPO_ID)
    task = meta.tasks.index[0]
    camera = meta.camera_keys[0]
    episode_indices = episodes_for_task(DEFAULT_REPO_ID, task)[:3]
    episodes = load_episode_arrays(DEFAULT_REPO_ID, episode_indices, camera_keys=[camera])

    estimator = LinearProgressEstimator()
    episodes = annotate_episode_values(episodes, estimator, task, camera)

    model = load_dinov2()
    pool = build_chunk_dictionary(episodes, task=task, chunk_size=30, primary_camera=camera, embed_model=model)
    print(f"chunk dictionary size: {len(pool)}, key_values is not None: {pool.key_values is not None}")

    query_chunk = pool.chunks[10]
    query_embedding = pool.key_embeddings[10]
    query_value = float(pool.key_values[10])
    print(f"query: ep={query_chunk.episode_index} frames=[{query_chunk.start_frame},{query_chunk.end_frame}) value={query_value:.3f}")

    for metric in ("vision", "value"):
        result = f_retrieve(metric, query_embedding, query_value, pool, k=5)
        print(f"\n[{metric}] top-5:")
        for c, s in zip(result.chunks, result.scores):
            print(f"  ep={c.episode_index} frames=[{c.start_frame},{c.end_frame}) value={c.value[0]:.3f} score={s:.4f}")

    fusion = ValueFusionMLP()  # untrained — validates the forward pass shape/dispatch only
    result = f_retrieve("vision+value", query_embedding, query_value, pool, k=5, fusion=fusion)
    print("\n[vision+value, untrained fusion] top-5:")
    for c, s in zip(result.chunks, result.scores):
        print(f"  ep={c.episode_index} frames=[{c.start_frame},{c.end_frame}) value={c.value[0]:.3f} score={s:.4f}")

    # Sanity: self-retrieval is always the nearest neighbor for "vision" and "value" (both are
    # exact-zero-distance metrics at the query chunk itself).
    vis_result = f_retrieve("vision", query_embedding, query_value, pool, k=1)
    val_result = f_retrieve("value", query_embedding, query_value, pool, k=1)
    assert vis_result.chunks[0].start_frame == query_chunk.start_frame
    assert val_result.chunks[0].start_frame == query_chunk.start_frame
    print("\nOK: vision/value self-retrieval sanity checks passed.")


if __name__ == "__main__":
    main()

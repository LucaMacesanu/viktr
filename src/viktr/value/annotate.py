"""Offline value annotation: attaches a "values" (T,) array to each episode
dict, from a ValueEstimator, so chunk_dictionary.build_chunk_dictionary can
slice it alongside images/proprio/actions per chunk.
"""

from __future__ import annotations

from viktr.value.base import ValueEstimator


def annotate_episode_values(
    episodes: list[dict],
    estimator: ValueEstimator,
    task: str,
    camera_key: str,
) -> list[dict]:
    """Returns new episode dicts (shallow-copied) with a "values" (T,) float32 array added."""
    annotated = []
    for ep in episodes:
        values = estimator.estimate_episode(ep["images"][camera_key], task)
        annotated.append({**ep, "values": values})
    return annotated

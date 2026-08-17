"""Retrieval metrics: f_retrieve variants (paper Sec III-G, Eq. 4).

Three modes, all returning nearest-to-farthest RetrievalResults over the same
ChunkDictionary:
  - "vision": pure DINOv2 embedding-space distance (RICL's metric).
  - "value":  pure |v_hat_t - v'| value-alignment distance.
  - "vision+value": learned fusion g_psi(d_vis, d_val) (retrieval/fusion.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from viktr.retrieval.chunk_dictionary import Chunk, ChunkDictionary
from viktr.retrieval.fusion import ValueFusionMLP

RetrievalMetric = Literal["vision", "value", "vision+value"]


@dataclass
class RetrievalResult:
    chunks: list[Chunk]  # ordered nearest-to-farthest
    scores: np.ndarray  # (k,) float32 — the ranking score used (lower = nearer/better)


def _topk(scores: np.ndarray, pool: ChunkDictionary, k: int) -> RetrievalResult:
    if k > len(pool):
        raise ValueError(f"k={k} exceeds pool size {len(pool)}")
    order = np.argsort(scores)[:k]
    return RetrievalResult(chunks=[pool.chunks[i] for i in order], scores=scores[order])


def vision_retrieve(query_embedding: np.ndarray, pool: ChunkDictionary, k: int) -> RetrievalResult:
    """Pure visual-similarity retrieval: L2 distance in DINOv2 embedding space."""
    distances = np.linalg.norm(pool.key_embeddings - query_embedding[None, :], axis=1)
    return _topk(distances, pool, k)


def value_retrieve(query_value: float, pool: ChunkDictionary, k: int) -> RetrievalResult:
    """Pure value-alignment retrieval: |v_hat_t - v'| in progress space."""
    if pool.key_values is None:
        raise ValueError("pool has no key_values; annotate it first (viktr.value.annotate)")
    distances = np.abs(pool.key_values - query_value)
    return _topk(distances, pool, k)


def fused_retrieve(
    query_embedding: np.ndarray,
    query_value: float,
    pool: ChunkDictionary,
    k: int,
    fusion: ValueFusionMLP,
) -> RetrievalResult:
    """Learned fusion of vision + value distances (Sec III-G)."""
    if pool.key_values is None:
        raise ValueError("pool has no key_values; annotate it first (viktr.value.annotate)")
    d_vis = np.linalg.norm(pool.key_embeddings - query_embedding[None, :], axis=1)
    d_val = np.abs(pool.key_values - query_value)
    device = next(fusion.parameters()).device
    with torch.no_grad():
        d_vis_t = torch.from_numpy(d_vis).float().to(device)
        d_val_t = torch.from_numpy(d_val).float().to(device)
        scores = fusion(d_vis_t, d_val_t).cpu().numpy()
    return _topk(scores, pool, k)


def f_retrieve(
    metric: RetrievalMetric,
    query_embedding: np.ndarray,
    query_value: float | None,
    pool: ChunkDictionary,
    k: int,
    fusion: ValueFusionMLP | None = None,
) -> RetrievalResult:
    """Single dispatch point for the metric choice wired into training config (task 6)."""
    if metric == "vision":
        return vision_retrieve(query_embedding, pool, k)
    if metric == "value":
        if query_value is None:
            raise ValueError("metric='value' requires query_value")
        return value_retrieve(query_value, pool, k)
    if metric == "vision+value":
        if query_value is None or fusion is None:
            raise ValueError("metric='vision+value' requires both query_value and a fusion module")
        return fused_retrieve(query_embedding, query_value, pool, k, fusion)
    raise ValueError(f"unknown retrieval metric: {metric!r}")

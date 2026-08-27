"""Differentiable (soft) top-k retrieval for the vision+value arm, training gψ
(ValueFusionMLP, retrieval/fusion.py) jointly with the IC-VLA per the paper's Sec
III-G: "gψ's parameters receive gradient signal indirectly through the IC-VLA's
training objective... by scoring the full retrieval-pool candidate set with a soft
(temperature-annealed softmax) relaxation of top-k selection during meta-training,
and switching to hard top-k selection at evaluation and deployment time."

The paper doesn't give an exact equation for the soft relaxation itself, so this is
my concrete implementation of that description, not a verbatim reproduction:

  - Score every pool candidate with gψ (cheap: a 2-layer MLP over precomputed
    (d_vis, d_val) pairs, no VLA forward involved -- unlike hard top-k, this never
    needs to embed the whole pool through the expensive SigLIP/PaliGemma prefix).
  - Select the actual k chunks to embed via sequential Gumbel-max sampling without
    replacement (standard Plackett-Luce reparameterization) -- so, same as hard
    top-k, only the k SELECTED chunks are ever run through the policy's embed_prefix.
  - Weight each selected chunk's contribution to the loss by a straight-through
    scalar `y_soft[i] + (1 - y_soft[i]).detach()`: exactly 1.0 in the forward pass
    (no numeric effect vs. hard top-k), but its gradient w.r.t. gψ's parameters is
    `d(y_soft[i])/d(gψ)`, so training pushes gψ to raise the softmax weight of
    chunks whose retrieval led to LOWER downstream flow-matching loss, and lower it
    for chunks that led to higher loss -- a REINFORCE-style signal from the actual
    task objective, applied at the loss level rather than by patching gradients
    into the frozen SigLIP/PaliGemma embedding path (pi05_context.py is left
    untouched; this module is a pure addition).
  - At eval/deployment (`training=False`), this is exactly `fused_retrieve`
    (retrieval/metrics.py): hard top-k, weight fixed at 1.0.

Vision-only and value-only retrieval have no learned parameters in their score, so
they don't need any of this -- they keep using `vision_retrieve`/`value_retrieve`
(retrieval/metrics.py) directly with plain hard top-k, at both train and eval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from viktr.retrieval.chunk_dictionary import ChunkDictionary
from viktr.retrieval.fusion import ValueFusionMLP
from viktr.retrieval.metrics import RetrievalResult, fused_retrieve


@dataclass
class SoftRetrievalResult(RetrievalResult):
    weights: list[torch.Tensor] = field(default_factory=list)  # length k; ST scalar per selected chunk, ~1.0 in forward


def soft_fused_retrieve(
    query_embedding: np.ndarray,
    query_value: float,
    pool: ChunkDictionary,
    k: int,
    fusion: ValueFusionMLP,
    tau: float,
    training: bool,
) -> SoftRetrievalResult:
    if not training:
        hard = fused_retrieve(query_embedding, query_value, pool, k, fusion)
        ones = [torch.tensor(1.0) for _ in range(k)]
        return SoftRetrievalResult(chunks=hard.chunks, scores=hard.scores, weights=ones)

    if pool.key_values is None:
        raise ValueError("pool has no key_values; annotate it first (viktr.data.icl_dataset.robodopamine_values)")
    if k > len(pool):
        raise ValueError(f"k={k} exceeds pool size {len(pool)}")

    device = next(fusion.parameters()).device
    d_vis = torch.from_numpy(
        np.linalg.norm(pool.key_embeddings - query_embedding[None, :], axis=1)
    ).float().to(device)
    d_val = torch.from_numpy(np.abs(pool.key_values - query_value)).float().to(device)
    scores = fusion(d_vis, d_val)  # (N_pool,) -- lower = more relevant, matches fused_retrieve's convention

    logits = -scores / tau
    remaining = torch.ones(len(pool), dtype=torch.bool, device=device)

    chosen_indices: list[int] = []
    chosen_weights: list[torch.Tensor] = []
    for _ in range(k):
        masked_logits = logits.masked_fill(~remaining, float("-inf"))
        y_soft = torch.softmax(masked_logits, dim=0)

        # Gumbel-max sampling over the remaining candidates -- equivalent to sampling
        # from y_soft, but with an explicit reparameterizable noise draw so this is a
        # genuine stochastic selection (not a deterministic argmax) during training.
        u = torch.rand(len(pool), device=device).clamp_min(1e-20)
        gumbel_noise = -torch.log(-torch.log(u))
        gumbel_noise = gumbel_noise.masked_fill(~remaining, float("-inf"))
        idx = int(torch.argmax(masked_logits + gumbel_noise).item())

        weight = y_soft[idx] + (1.0 - y_soft[idx]).detach()  # straight-through: forward == 1.0 exactly
        chosen_indices.append(idx)
        chosen_weights.append(weight)
        remaining[idx] = False

    return SoftRetrievalResult(
        chunks=[pool.chunks[i] for i in chosen_indices],
        scores=scores[chosen_indices].detach().cpu().numpy(),
        weights=chosen_weights,
    )

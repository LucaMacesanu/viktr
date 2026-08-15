"""Learned vision+value fusion g_psi (paper Sec III-G): a small 2-layer MLP
over [d_vis, |v_hat - v'|] producing one fused ranking score. Trained jointly
with the IC-VLA (task 5/6) via soft top-k during meta-training; used with hard
top-k here at eval/deployment time (retrieve_fused below).
"""

from __future__ import annotations

import torch
from torch import nn


class ValueFusionMLP(nn.Module):
    """g_psi(d_vis, d_val) -> fused score. Lower score = more relevant, matching
    the plain-distance convention used by vision_retrieve/value_retrieve."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, d_vis: torch.Tensor, d_val: torch.Tensor) -> torch.Tensor:
        x = torch.stack([d_vis, d_val], dim=-1)
        return self.net(x).squeeze(-1)

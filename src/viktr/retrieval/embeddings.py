"""DINOv2 frame embeddings for vision-similarity retrieval (paper Sec III-B/III-G).

Mirrors ricl_openpi's embedding convention (torch.hub dinov2_vitb14, CLS token)
so the "vision" retrieval metric matches RICL's actual mechanism.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_DINOV2_HUB_REPO = "facebookresearch/dinov2"
_DINOV2_MODEL_NAME = "dinov2_vitb14"
EMBED_DIM = 768

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_model_cache: torch.nn.Module | None = None


def load_dinov2(device: str | None = None) -> torch.nn.Module:
    """Loads (and caches) the frozen DINOv2 ViT-B/14 backbone used as the retrieval key encoder."""
    global _model_cache
    if _model_cache is None:
        model = torch.hub.load(_DINOV2_HUB_REPO, _DINOV2_MODEL_NAME)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _model_cache = model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _model_cache.to(device)


def _to_dinov2_input(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, 224, 224) ImageNet-normalized float tensor."""
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) images, got shape {images.shape}")
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


@torch.no_grad()
def embed_frames(images: np.ndarray, model: torch.nn.Module | None = None, batch_size: int = 256) -> np.ndarray:
    """CLS-token DINOv2 embedding per frame. images: (N, H, W, 3) uint8 -> (N, EMBED_DIM) float32."""
    model = model if model is not None else load_dinov2()
    device = next(model.parameters()).device
    embeddings = []
    for start in range(0, len(images), batch_size):
        batch = _to_dinov2_input(images[start : start + batch_size]).to(device)
        features = model.forward_features(batch)
        embeddings.append(features["x_norm_clstoken"].float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)

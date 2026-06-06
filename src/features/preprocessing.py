"""Uniform feature preprocessing shared by every encoder.

Proposal 2.5 fixes the pipeline so cross-encoder comparisons are fair:

  1. Random projection to a common dim (384) with entries ~ N(0, 1/384),
     seed=42. Applied uniformly so SVM margins are not confounded by the
     512-d (ResNet) vs 1024-d (ViT-L) output mismatch.
  2. L2 normalization of every feature vector, so margin and distance metrics
     are comparable across encoders with different output scales.
  3. Mean-centering (column means) -- applied inside the SVD step so that
     Z = U S V^T equals PCA. See analysis/svd.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..utils.seed import get_generator


class RandomProjection:
    """Fixed Gaussian random projection R in R^{d_in x d_out}, entries ~ N(0, 1/d_out).

    The matrix is generated deterministically from a seed, so the same encoder
    always sees the same projection across runs (and across fit/eval splits).
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 42):
        self.d_in = d_in
        self.d_out = d_out
        self.seed = seed
        g = get_generator(seed=seed, device="cpu")
        std = 1.0 / math.sqrt(d_out)
        self.R = torch.randn(d_in, d_out, generator=g) * std  # [d_in, d_out]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.d_in:
            raise ValueError(f"RandomProjection expected d_in={self.d_in}, got {x.shape[1]}")
        return x @ self.R.to(x.dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def mean_center(x: torch.Tensor, mean: torch.Tensor | None = None):
    """Subtract column means. Returns (centered, mean) so the same mean can be
    reused on a held-out split."""
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    return x - mean, mean


def build_preprocessor(cfg, d_in: int):
    """Compose the configured (random projection -> L2 normalize) transform.

    Mean-centering is intentionally excluded here: SVD needs it but the SVM
    probe operates on the L2-normalized (uncentered) features.
    """
    steps = []
    if cfg.features.random_projection.enabled:
        rp = RandomProjection(
            d_in=d_in,
            d_out=cfg.features.random_projection.target_dim,
            seed=cfg.features.random_projection.seed,
        )
        steps.append(rp)
    do_l2 = cfg.features.l2_normalize

    def _apply(x: torch.Tensor) -> torch.Tensor:
        for s in steps:
            x = s(x)
        if do_l2:
            x = l2_normalize(x)
        return x

    return _apply

"""Part A -- SVD spectral analysis.

Given a feature matrix Z in R^{N x d} (already L2-normalized), we mean-center the
columns and compute Z = U S V^T via torch.linalg.svd. Because of the centering,
this decomposition is equivalent to PCA, so the singular values describe the
variance structure of the learned manifold.

Diagnostics:
  * RankMe effective rank (Garrido et al., 2023) -- entropy of the normalized
    singular-value distribution, exp(H(p)).
  * PCA effective rank -- participation ratio (sum s^2)^2 / sum s^4.
  * Dimensional collapse -- count of singular values carrying meaningful variance.

Per the proposal, effective rank is a *descriptor of geometry*, not a quality
score: higher rank does not by itself mean a better representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..features.preprocessing import mean_center


@dataclass
class SVDResult:
    singular_values: torch.Tensor          # [k], descending
    explained_variance_ratio: torch.Tensor  # [k], sums to 1
    rankme: float                          # entropy-based effective rank
    pca_effective_rank: float              # participation ratio
    n_dims_for_90pct: int                  # dims to reach 90% variance
    n_dims_for_99pct: int
    collapsed_dims: int                    # dims below collapse threshold
    ambient_dim: int
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "rankme": self.rankme,
            "pca_effective_rank": self.pca_effective_rank,
            "n_dims_for_90pct": self.n_dims_for_90pct,
            "n_dims_for_99pct": self.n_dims_for_99pct,
            "collapsed_dims": self.collapsed_dims,
            "ambient_dim": self.ambient_dim,
        }


def svd_spectrum(z: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Return singular values (descending) of (optionally mean-centered) Z."""
    z = z.float()
    if center:
        z, _ = mean_center(z)
    # economy SVD; we only need singular values.
    s = torch.linalg.svdvals(z)
    return s


def rankme(singular_values: torch.Tensor, epsilon: float = 1e-7) -> float:
    """RankMe effective rank = exp(-sum p_k log p_k), p_k = s_k / sum(s) + eps.

    Garrido et al. (2023), "RankMe: Assessing the downstream performance of
    pretrained self-supervised representations by their rank."
    """
    s = singular_values.float()
    p = s / (s.sum() + epsilon) + epsilon
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


def effective_rank_pca(singular_values: torch.Tensor) -> float:
    """Participation-ratio effective rank: (sum s^2)^2 / sum s^4.

    Equals the number of equally-weighted dimensions that would produce the same
    variance dispersion; a complementary, non-entropy view of spectral spread.
    """
    s2 = singular_values.float() ** 2
    return float((s2.sum() ** 2) / (s2.pow(2).sum() + 1e-12))


def dimensional_collapse(explained_variance_ratio: torch.Tensor, threshold: float = 0.01) -> int:
    """Count dimensions whose explained-variance ratio falls below ``threshold``.

    A large count relative to ambient dim indicates dimensional collapse
    (Jing et al., 2022).
    """
    return int((explained_variance_ratio < threshold).sum().item())


def analyze_svd(z: torch.Tensor, rankme_epsilon: float = 1e-7, collapse_threshold: float = 0.01,
                meta: dict | None = None) -> SVDResult:
    """Full Part-A SVD analysis on a feature matrix Z."""
    s = svd_spectrum(z, center=True)
    var = s ** 2
    evr = var / var.sum()
    cum = torch.cumsum(evr, dim=0)

    return SVDResult(
        singular_values=s,
        explained_variance_ratio=evr,
        rankme=rankme(s, epsilon=rankme_epsilon),
        pca_effective_rank=effective_rank_pca(s),
        n_dims_for_90pct=int((cum < 0.90).sum().item()) + 1,
        n_dims_for_99pct=int((cum < 0.99).sum().item()) + 1,
        collapsed_dims=dimensional_collapse(evr, threshold=collapse_threshold),
        ambient_dim=z.shape[1],
        meta=meta or {},
    )

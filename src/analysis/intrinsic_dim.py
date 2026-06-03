"""Two-NN intrinsic dimension estimator (Facco et al., 2017).

A non-linear complement to SVD effective rank. For each point we take the ratio
mu = r2 / r1 of its two nearest-neighbor distances. Under a locally uniform
density of intrinsic dimension d, mu follows a Pareto(d) distribution, so

    F(mu) = 1 - mu^{-d}   =>   -log(1 - F(mu)) = d * log(mu).

Fitting a line through the origin to (log mu, -log(1 - F_emp)) gives slope d.

If this estimate diverges substantially from the SVD effective rank, the latent
manifold has significant curvature -- a linear subspace is a poor fit (proposal,
Part A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class TwoNNResult:
    dimension: float
    n_points: int
    discard_fraction: float

    def summary(self) -> dict:
        return {"two_nn_dim": self.dimension, "n_points": self.n_points}


def two_nn_dimension(
    z: torch.Tensor | np.ndarray,
    discard_fraction: float = 0.1,
    max_points: int | None = None,
    seed: int = 42,
) -> TwoNNResult:
    """Estimate intrinsic dimension via the Two-NN method.

    Args:
        z: [N, d] feature matrix.
        discard_fraction: drop the top fraction of mu ratios before the linear
            fit (Facco et al. recommend discarding the heavy tail for robustness).
        max_points: optionally subsample for speed on large N.
    """
    x = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

    if max_points is not None and x.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[idx]

    # 3 neighbors: self + first + second.
    nbrs = NearestNeighbors(n_neighbors=3).fit(x)
    dists, _ = nbrs.kneighbors(x)
    r1, r2 = dists[:, 1], dists[:, 2]

    # Guard against duplicate points (r1 == 0).
    valid = r1 > 1e-12
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]  # mu must exceed 1 by construction
    mu.sort()

    n = mu.shape[0]
    # Empirical CDF F_emp(mu_i) = i / n  (i = 1..n).
    f_emp = np.arange(1, n + 1) / n

    # Keep the lower (1 - discard_fraction) portion for the linear fit.
    keep = int(np.floor(n * (1 - discard_fraction)))
    x_fit = np.log(mu[:keep])
    y_fit = -np.log(1.0 - f_emp[:keep])

    # Line through the origin: d = (x . y) / (x . x).
    d = float(np.dot(x_fit, y_fit) / np.dot(x_fit, x_fit))

    return TwoNNResult(dimension=d, n_points=int(z.shape[0]), discard_fraction=discard_fraction)

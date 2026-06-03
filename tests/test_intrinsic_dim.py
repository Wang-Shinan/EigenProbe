import numpy as np
import torch

from src.analysis.intrinsic_dim import two_nn_dimension


def test_two_nn_recovers_linear_subspace_dim():
    # Points uniformly in a 5-D subspace embedded in 50-D: Two-NN ~ 5.
    rng = np.random.default_rng(0)
    coeffs = rng.uniform(0, 1, size=(4000, 5)).astype("float32")
    embed = rng.standard_normal((5, 50)).astype("float32")
    z = torch.from_numpy(coeffs @ embed)
    res = two_nn_dimension(z, discard_fraction=0.1, seed=0)
    assert 4.0 < res.dimension < 6.5


def test_two_nn_low_for_1d_curve():
    # A 1-D curve (helix) in 3-D should give intrinsic dim near 1.
    # NB: Two-NN assumes points are sampled randomly (locally Poisson), so we
    # draw t at random -- a regular grid would make r1 == r2 and break the ratio.
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, size=4000)).astype("float32")
    z = torch.from_numpy(np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1))
    res = two_nn_dimension(z, discard_fraction=0.1, seed=0)
    assert res.dimension < 1.6


def test_two_nn_subsampling_runs():
    z = torch.randn(20000, 20)
    res = two_nn_dimension(z, max_points=2000, seed=0)
    assert res.n_points == 20000  # reports original N
    assert res.dimension > 0

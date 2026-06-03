import torch

from src.features.preprocessing import RandomProjection, l2_normalize, mean_center


def test_random_projection_is_deterministic():
    a = RandomProjection(512, 384, seed=42)
    b = RandomProjection(512, 384, seed=42)
    assert torch.allclose(a.R, b.R)


def test_random_projection_seed_changes_matrix():
    a = RandomProjection(512, 384, seed=42)
    c = RandomProjection(512, 384, seed=1)
    assert not torch.allclose(a.R, c.R)


def test_random_projection_shape_and_scale():
    rp = RandomProjection(512, 384, seed=42)
    x = torch.randn(100, 512)
    y = rp(x)
    assert y.shape == (100, 384)
    # entries ~ N(0, 1/384) -> std ~ 1/sqrt(384)
    assert abs(rp.R.std().item() - (1 / 384) ** 0.5) < 0.01


def test_l2_normalize_unit_norm():
    x = torch.randn(50, 384) * 7.0
    y = l2_normalize(x)
    norms = y.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_mean_center_zeroes_column_means():
    x = torch.randn(200, 16) + 3.0
    c, mean = mean_center(x)
    assert torch.allclose(c.mean(dim=0), torch.zeros(16), atol=1e-5)
    assert torch.allclose(mean.squeeze(0), x.mean(dim=0), atol=1e-5)

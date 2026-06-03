import torch

from src.analysis.svd import analyze_svd, effective_rank_pca, rankme, svd_spectrum


def test_rankme_full_rank_isotropic_is_high():
    # Isotropic Gaussian -> singular values roughly equal -> RankMe near dim.
    torch.manual_seed(0)
    z = torch.randn(5000, 50)
    s = svd_spectrum(z)
    r = rankme(s)
    assert r > 35  # close to ambient 50, allowing for finite-sample decay


def test_rankme_low_rank_is_low():
    # Data living in a 3-D subspace embedded in 50-D -> low effective rank.
    torch.manual_seed(0)
    basis = torch.randn(3, 50)
    coeffs = torch.randn(5000, 3)
    z = coeffs @ basis
    s = svd_spectrum(z)
    assert rankme(s) < 6
    assert effective_rank_pca(s) < 6


def test_collapse_detected_in_low_rank():
    torch.manual_seed(0)
    basis = torch.randn(2, 30)
    z = torch.randn(3000, 2) @ basis
    res = analyze_svd(z, collapse_threshold=0.01)
    assert res.collapsed_dims >= 25  # most of 30 dims carry ~no variance
    assert res.n_dims_for_90pct <= 2


def test_rankme_monotone_under_rank_increase():
    torch.manual_seed(0)
    def er(k):
        z = torch.randn(4000, k) @ torch.randn(k, 40)
        return rankme(svd_spectrum(z))
    assert er(2) < er(10) < er(30)

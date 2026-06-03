import torch

from src.analysis.svm_probe import linear_svm_probe


def _two_blobs(n=400, dim=384, sep=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    a = torch.randn(half, dim, generator=g)
    b = torch.randn(half, dim, generator=g) + sep
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
    return x, y


def test_separable_blobs_high_accuracy():
    xf, yf = _two_blobs(seed=0)
    xe, ye = _two_blobs(seed=1)
    res = linear_svm_probe(xf, yf, xe, ye, C=1.0)
    assert res.accuracy > 0.98
    assert res.margins_per_problem.shape[0] == 1  # one binary problem (2 classes)


def test_wider_separation_gives_larger_margin():
    xf_easy, yf = _two_blobs(sep=8.0, seed=0)
    xe_easy, ye = _two_blobs(sep=8.0, seed=1)
    easy = linear_svm_probe(xf_easy, yf, xe_easy, ye, C=1.0)

    xf_hard, yf2 = _two_blobs(sep=2.0, seed=0)
    xe_hard, ye2 = _two_blobs(sep=2.0, seed=1)
    hard = linear_svm_probe(xf_hard, yf2, xe_hard, ye2, C=1.0)

    assert easy.mean_margin > hard.mean_margin
    # Cleaner boundary -> fewer support vectors.
    assert easy.n_support_vectors < hard.n_support_vectors


def test_confusion_and_hardest_pairs_shapes():
    xf, yf = _two_blobs(seed=0)
    xe, ye = _two_blobs(seed=1)
    res = linear_svm_probe(xf, yf, xe, ye)
    assert res.confusion.shape == (2, 2)
    assert len(res.hardest_pairs) >= 1

"""Part B -- linear SVM probe.

Trains a linear SVM on frozen features and reports interpretable diagnostics
beyond accuracy (proposal 2.2 Part B):

  * Margin size: for a linear SVM the geometric margin is 1 / ||w||. We report
    the mean margin over one-vs-one (or one-vs-rest) binary problems. A larger
    margin => the latent classes are more linearly organized.
  * Support vector count: fewer SVs suggest a cleaner decision boundary.
  * Class confusion: which classes are hardest to separate, and whether that
    tracks semantic similarity.

Features are expected to be already preprocessed (random projection + L2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


@dataclass
class SVMResult:
    accuracy: float
    mean_margin: float                 # mean over binary sub-problems of 1/||w||
    margins_per_problem: np.ndarray    # [n_binary] geometric margins
    n_support_vectors: int
    support_vector_fraction: float     # SVs / n_fit
    confusion: np.ndarray              # [C, C]
    hardest_pairs: list                # [(class_i, class_j, off_diag_count), ...]
    C: float
    n_fit: int
    n_eval: int
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "mean_margin": self.mean_margin,
            "n_support_vectors": self.n_support_vectors,
            "support_vector_fraction": self.support_vector_fraction,
            "C": self.C,
        }


def _geometric_margins(svc: SVC) -> np.ndarray:
    """Geometric margin 1/||w|| for each binary sub-problem of a fitted SVC.

    sklearn's one-vs-one stores one coef_ row per class pair (linear kernel).
    """
    w = svc.coef_  # [n_binary, n_features]
    norms = np.linalg.norm(w, axis=1)
    return 1.0 / np.clip(norms, 1e-12, None)


def _hardest_pairs(cm: np.ndarray, top_k: int = 5) -> list:
    """Return the class pairs with the most mutual confusion."""
    c = cm.shape[0]
    pairs = []
    for i in range(c):
        for j in range(i + 1, c):
            pairs.append((i, j, int(cm[i, j] + cm[j, i])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def linear_svm_probe(
    feats_fit: torch.Tensor,
    labels_fit: torch.Tensor,
    feats_eval: torch.Tensor,
    labels_eval: torch.Tensor,
    C: float = 1.0,
    max_samples_fit: int | None = None,
    class_weight=None,
    seed: int = 42,
    meta: dict | None = None,
) -> SVMResult:
    """Fit a linear SVM and compute margin / support-vector / confusion diagnostics."""
    Xf = feats_fit.detach().cpu().numpy()
    yf = labels_fit.detach().cpu().numpy()
    Xe = feats_eval.detach().cpu().numpy()
    ye = labels_eval.detach().cpu().numpy()

    if max_samples_fit is not None and Xf.shape[0] > max_samples_fit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Xf.shape[0], size=max_samples_fit, replace=False)
        Xf, yf = Xf[idx], yf[idx]

    svc = SVC(kernel="linear", C=C, class_weight=class_weight, random_state=seed)
    svc.fit(Xf, yf)

    pred = svc.predict(Xe)
    acc = float(accuracy_score(ye, pred))
    cm = confusion_matrix(ye, pred)
    margins = _geometric_margins(svc)

    n_sv = int(svc.support_vectors_.shape[0])
    return SVMResult(
        accuracy=acc,
        mean_margin=float(margins.mean()),
        margins_per_problem=margins,
        n_support_vectors=n_sv,
        support_vector_fraction=n_sv / Xf.shape[0],
        confusion=cm,
        hardest_pairs=_hardest_pairs(cm),
        C=C,
        n_fit=int(Xf.shape[0]),
        n_eval=int(Xe.shape[0]),
        meta=meta or {},
    )

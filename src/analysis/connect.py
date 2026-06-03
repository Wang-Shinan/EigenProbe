"""Connecting SVD to SVM -- the key analytical contribution (proposal 2.2).

Tests whether spectral richness (effective rank from Part A) is *associated* with
geometric separability (SVM margin from Part B) across encoders. This is an
empirical structural correlation, NOT a causal claim. The MAE/I-JEPA pair --
identical ViT-S/16 architecture, different objective -- is the cleanest test
because architecture is held fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class ConnectResult:
    encoders: list
    rankme: np.ndarray
    pca_effective_rank: np.ndarray
    mean_margin: np.ndarray
    svm_accuracy: np.ndarray
    pearson_rank_margin: tuple   # (r, p)
    spearman_rank_margin: tuple  # (rho, p)
    mae_ijepa_delta: dict        # the architecture-controlled comparison

    def to_rows(self) -> list[dict]:
        rows = []
        for i, name in enumerate(self.encoders):
            rows.append(
                {
                    "encoder": name,
                    "rankme": float(self.rankme[i]),
                    "pca_effective_rank": float(self.pca_effective_rank[i]),
                    "mean_margin": float(self.mean_margin[i]),
                    "svm_accuracy": float(self.svm_accuracy[i]),
                }
            )
        return rows


def rank_margin_table(per_encoder: dict[str, dict]) -> ConnectResult:
    """Assemble the cross-encoder association.

    Args:
        per_encoder: {encoder_name: {"rankme":..., "pca_effective_rank":...,
                      "mean_margin":..., "svm_accuracy":...}, ...}
    """
    names = list(per_encoder.keys())
    rankme = np.array([per_encoder[n]["rankme"] for n in names])
    pca_er = np.array([per_encoder[n]["pca_effective_rank"] for n in names])
    margin = np.array([per_encoder[n]["mean_margin"] for n in names])
    acc = np.array([per_encoder[n]["svm_accuracy"] for n in names])

    # Correlations are descriptive given the small number of encoders.
    if len(names) >= 3:
        pear = pearsonr(rankme, margin)
        spear = spearmanr(rankme, margin)
        pearson = (float(pear[0]), float(pear[1]))
        spearman = (float(spear[0]), float(spear[1]))
    else:
        pearson = spearman = (float("nan"), float("nan"))

    # Architecture-controlled delta: MAE vs I-JEPA (same ViT-S/16).
    delta = {}
    if "ijepa_vits16" in per_encoder and "mae_vits16" in per_encoder:
        j, m = per_encoder["ijepa_vits16"], per_encoder["mae_vits16"]
        delta = {
            "rankme_diff_ijepa_minus_mae": j["rankme"] - m["rankme"],
            "margin_diff_ijepa_minus_mae": j["mean_margin"] - m["mean_margin"],
            "accuracy_diff_ijepa_minus_mae": j["svm_accuracy"] - m["svm_accuracy"],
        }

    return ConnectResult(
        encoders=names,
        rankme=rankme,
        pca_effective_rank=pca_er,
        mean_margin=margin,
        svm_accuracy=acc,
        pearson_rank_margin=pearson,
        spearman_rank_margin=spearman,
        mae_ijepa_delta=delta,
    )

"""Part D (optional) -- latent stability under input perturbation.

Measures how much an encoder's representation shifts when the input is perturbed:

    sensitivity(x, eps) = ||f(x) - f(x + eps)|| / ||f(x)||

averaged over images, as a function of perturbation magnitude. I-JEPA's
predictive objective is hypothesized to yield more stable (abstract) features
than a supervised CNN.

This module is encoder-agnostic: callers pass in already-extracted clean and
perturbed feature tensors. Driving the perturbations lives in
scripts/06_perturbation.py using data/perturbations.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PerturbationResult:
    perturbation: str
    magnitudes: list
    mean_sensitivity: list   # mean relative shift per magnitude
    std_sensitivity: list
    n_images: int

    def summary(self) -> dict:
        return {
            "perturbation": self.perturbation,
            "magnitudes": self.magnitudes,
            "mean_sensitivity": self.mean_sensitivity,
        }


def relative_shift(clean: torch.Tensor, perturbed: torch.Tensor, eps: float = 1e-12):
    """Per-sample ||f(x) - f(x')|| / ||f(x)|| for batched features [N, d]."""
    num = (clean - perturbed).norm(dim=1)
    den = clean.norm(dim=1).clamp_min(eps)
    return num / den


def perturbation_sensitivity(
    clean_feats: torch.Tensor,                 # [N, d]
    perturbed_feats_by_mag: dict[float, torch.Tensor],  # magnitude -> [N, d]
    perturbation: str,
) -> PerturbationResult:
    """Aggregate relative shift across magnitudes into a sensitivity curve."""
    mags = sorted(perturbed_feats_by_mag.keys())
    means, stds = [], []
    for m in mags:
        shift = relative_shift(clean_feats, perturbed_feats_by_mag[m])
        means.append(float(shift.mean()))
        stds.append(float(shift.std()))

    return PerturbationResult(
        perturbation=perturbation,
        magnitudes=[float(m) for m in mags],
        mean_sensitivity=means,
        std_sensitivity=stds,
        n_images=int(clean_feats.shape[0]),
    )

"""Part C (optional) -- multi-view viewpoint consistency on ModelNet40.

Tests whether an encoder treats the 12 rendered views of one 3D object as more
similar to each other than to other objects of the same class -- evidence that
the latent space is organized by object identity rather than appearance.

Metrics (proposal 2.3 Part C), computed on L2-normalized embeddings:
  * Intra-instance compactness: mean pairwise distance among an object's views.
  * Inter-instance separation: mean pairwise distance between instances of the
    same class (using per-instance mean embeddings).
  * Viewpoint consistency score: intra / inter. Lower is better.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ViewpointResult:
    intra_instance: float
    inter_instance: float
    consistency_score: float    # intra / inter, lower = more viewpoint-invariant
    per_class: dict             # class_idx -> consistency score
    n_instances: int

    def summary(self) -> dict:
        return {
            "intra_instance": self.intra_instance,
            "inter_instance": self.inter_instance,
            "consistency_score": self.consistency_score,
            "n_instances": self.n_instances,
        }


def _mean_pairwise_distance(x: torch.Tensor) -> float:
    """Mean pairwise Euclidean distance among rows of x ([k, d])."""
    if x.shape[0] < 2:
        return 0.0
    d = torch.cdist(x, x)
    n = x.shape[0]
    return float(d.sum() / (n * (n - 1)))


def viewpoint_consistency(
    embeddings: torch.Tensor,   # [num_instances, num_views, d], L2-normalized
    classes: torch.Tensor,      # [num_instances]
) -> ViewpointResult:
    """Compute intra/inter-instance distances and the consistency score."""
    n_inst, n_views, _ = embeddings.shape

    # Intra-instance compactness: average over instances.
    intra = float(np.mean([_mean_pairwise_distance(embeddings[i]) for i in range(n_inst)]))

    # Inter-instance separation: per-class, between instance-mean embeddings.
    inst_mean = embeddings.mean(dim=1)  # [n_inst, d]
    classes_np = classes.cpu().numpy()
    inter_per_class = {}
    for c in np.unique(classes_np):
        members = inst_mean[classes == int(c)]
        if members.shape[0] >= 2:
            inter_per_class[int(c)] = _mean_pairwise_distance(members)

    inter = float(np.mean(list(inter_per_class.values()))) if inter_per_class else float("nan")

    per_class = {c: (intra / v if v > 0 else float("nan")) for c, v in inter_per_class.items()}

    return ViewpointResult(
        intra_instance=intra,
        inter_instance=inter,
        consistency_score=(intra / inter) if inter and inter > 0 else float("nan"),
        per_class=per_class,
        n_instances=n_inst,
    )

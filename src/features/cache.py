"""Persist extracted features so analysis scripts don't re-run encoders.

A bundle stores RAW features (before random projection / L2). Preprocessing is
cheap and deterministic, so it's re-applied at analysis time; caching raw
features lets you change the preprocessing config without re-extracting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class FeatureBundle:
    encoder: str
    dataset: str
    split: str
    features: torch.Tensor  # [N, D] raw
    labels: torch.Tensor    # [N]
    meta: dict

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]

    @property
    def dim(self) -> int:
        return self.features.shape[1]


def _path(cache_root: str | Path, encoder: str, dataset: str, split: str) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{dataset}__{encoder}__{split}.pt"


def save_bundle(bundle: FeatureBundle, cache_root: str | Path) -> Path:
    path = _path(cache_root, bundle.encoder, bundle.dataset, bundle.split)
    torch.save(
        {
            "encoder": bundle.encoder,
            "dataset": bundle.dataset,
            "split": bundle.split,
            "features": bundle.features,
            "labels": bundle.labels,
            "meta": bundle.meta,
        },
        path,
    )
    return path


def load_bundle(cache_root: str | Path, encoder: str, dataset: str, split: str) -> FeatureBundle:
    path = _path(cache_root, encoder, dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features at {path}. Run scripts/01_extract_features.py first."
        )
    d = torch.load(path, map_location="cpu")
    return FeatureBundle(
        encoder=d["encoder"],
        dataset=d["dataset"],
        split=d["split"],
        features=d["features"],
        labels=d["labels"],
        meta=d["meta"],
    )

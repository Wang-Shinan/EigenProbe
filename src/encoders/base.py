"""Common encoder interface.

Every encoder is a frozen feature extractor: it maps a batch of images to a
single feature vector per image. Weights are frozen throughout all analyses
(proposal 2.1), so encoders run in eval mode with no grad.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderSpec:
    """Static metadata describing how to build/load an encoder."""

    name: str
    family: str          # "resnet" | "vit"
    arch: str
    paradigm: str        # supervised | contrastive | predictive | reconstruction
    feature_dim: int
    source: str          # "torchvision" | "timm" | "local"
    checkpoint: str | None
    pool: str            # "avgpool" | "mean" | "cls"


class Encoder(nn.Module):
    """Wraps a backbone and exposes a uniform ``forward -> [B, feature_dim]``."""

    def __init__(self, backbone: nn.Module, spec: EncoderSpec):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.freeze()

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W] -> [B, D]
        feats = self.backbone(x)
        if feats.ndim != 2:
            raise RuntimeError(
                f"{self.spec.name}: expected [B, D] features, got {tuple(feats.shape)}"
            )
        return feats

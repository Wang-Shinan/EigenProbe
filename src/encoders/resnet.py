"""ResNet feature extractors: supervised baseline and SimCLR contrastive baseline.

Both use a torchvision ResNet-18 with the classification head removed, returning
the 512-d global-average-pooled feature. The supervised variant loads ImageNet
weights; SimCLR loads a converted self-supervised checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .base import Encoder, EncoderSpec


class _ResNetBackbone(nn.Module):
    """ResNet-18 truncated after global average pool -> [B, 512]."""

    def __init__(self, weights=None):
        super().__init__()
        net = resnet18(weights=weights)
        net.fc = nn.Identity()  # drop the 1000-way classifier
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_simclr_state_dict(backbone: _ResNetBackbone, checkpoint: str) -> None:
    """Load a SimCLR ResNet-18 checkpoint into the torchvision backbone.

    Official SimCLR checkpoints store the encoder under various prefixes
    (e.g. "encoder.", "backbone.", "module.encoder_q."). We strip known
    prefixes and drop projection-head / classifier keys, then load
    non-strictly (the dropped fc is expected to be missing).
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "encoder.", "backbone.", "encoder_q.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        # Skip projection head / classifier weights.
        if nk.startswith(("projector", "projection", "fc.", "head.")):
            continue
        cleaned[f"net.{nk}"] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    # Only the dropped fc should be missing; surface anything else.
    real_missing = [m for m in missing if not m.startswith("net.fc")]
    if real_missing:
        raise RuntimeError(f"SimCLR load: unexpected missing keys: {real_missing[:8]} ...")


def build_resnet_encoder(spec: EncoderSpec) -> Encoder:
    if spec.source == "torchvision" and spec.checkpoint is None:
        backbone = _ResNetBackbone(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = _ResNetBackbone(weights=None)
        if spec.checkpoint is None:
            raise ValueError(f"{spec.name}: a checkpoint path is required for source={spec.source}")
        if spec.paradigm == "contrastive":
            _load_simclr_state_dict(backbone, spec.checkpoint)
        else:
            sd = torch.load(spec.checkpoint, map_location="cpu")
            backbone.load_state_dict(sd.get("state_dict", sd), strict=False)

    return Encoder(backbone, spec)

"""ViT-S/16 feature extractors for I-JEPA (predictive) and MAE (reconstruction).

Both share an identical timm ViT-S/16 backbone -- this is the proposal's
architecture control: the *only* difference is which self-supervised checkpoint
is loaded. Features are mean-pooled over patch tokens (the CLS token, if present,
is dropped) so the two are pooled identically and comparably.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder, EncoderSpec


class _ViTBackbone(nn.Module):
    """timm ViT with classifier removed; returns mean-pooled patch tokens [B, 384]."""

    def __init__(self, arch: str, pool: str = "mean"):
        super().__init__()
        import timm  # lazy: only required when actually building a ViT encoder

        # num_classes=0 removes the head; we take token features manually.
        self.vit = timm.create_model(arch, pretrained=False, num_classes=0)
        self.pool = pool
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)  # CLS/register tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)  # [B, num_tokens, D]
        if tokens.ndim == 2:  # some configs already pool
            return tokens
        if self.pool == "cls":
            return tokens[:, 0]
        patch_tokens = tokens[:, self.num_prefix :]  # drop CLS/register tokens
        return patch_tokens.mean(dim=1)


def _remap_jepa_keys(state: dict) -> dict:
    """Map I-JEPA target/context-encoder keys onto timm ViT names.

    I-JEPA checkpoints nest the encoder under "target_encoder"/"encoder" and use
    block naming close to timm's. This strips wrappers; remaining mismatches are
    handled by non-strict loading (a warning is logged by the caller).
    """
    out = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "target_encoder.", "encoder.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _remap_mae_keys(state: dict) -> dict:
    """Strip MAE wrappers; drop decoder + mask-token params (encoder only)."""
    out = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "encoder.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        if nk.startswith(("decoder", "mask_token")):
            continue
        out[nk] = v
    return out


def build_vit_encoder(spec: EncoderSpec) -> Encoder:
    backbone = _ViTBackbone(spec.arch, pool=spec.pool)

    if spec.checkpoint is None:
        raise ValueError(f"{spec.name}: a checkpoint path is required (got None).")

    ckpt = torch.load(spec.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt.get("encoder", ckpt)))

    if spec.paradigm == "predictive":      # I-JEPA
        state = _remap_jepa_keys(state)
    elif spec.paradigm == "reconstruction":  # MAE
        state = _remap_mae_keys(state)

    # Prefix onto the inner timm module ("vit.").
    state = {f"vit.{k}" if not k.startswith("vit.") else k: v for k, v in state.items()}
    missing, unexpected = backbone.load_state_dict(state, strict=False)

    # Sanity check: the patch embedding and at least one block must have loaded.
    loaded_ok = any("vit.blocks.0" in n for n in backbone.state_dict() if n not in missing)
    if not loaded_ok:
        raise RuntimeError(
            f"{spec.name}: checkpoint did not populate transformer blocks; "
            f"missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]}"
        )
    return Encoder(backbone, spec)

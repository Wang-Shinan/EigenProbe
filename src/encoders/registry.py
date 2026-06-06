"""Encoder factory: resolves a name -> built, frozen Encoder.

Specs live in configs/encoders.yaml. The four encoders compared throughout:
    resnet18_supervised  (supervised CNN baseline)
    simclr_resnet18      (contrastive baseline)
    ijepa_vitl16         (predictive -- focus of the study)
    mae_vitl16           (reconstruction -- architecture control vs I-JEPA)
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import Encoder, EncoderSpec
from .resnet import build_resnet_encoder
from .vit import build_vit_encoder

_DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "encoders.yaml"


def load_encoder_specs(path: str | Path = _DEFAULT_SPEC_PATH) -> dict[str, EncoderSpec]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {name: EncoderSpec(name=name, **fields) for name, fields in raw.items()}


def list_encoders(path: str | Path = _DEFAULT_SPEC_PATH) -> list[str]:
    return list(load_encoder_specs(path).keys())


def build_encoder(
    name: str,
    device: str = "cuda",
    specs: dict[str, EncoderSpec] | None = None,
) -> Encoder:
    specs = specs or load_encoder_specs()
    if name not in specs:
        raise KeyError(f"Unknown encoder '{name}'. Available: {list(specs)}")
    spec = specs[name]

    if spec.family == "resnet":
        encoder = build_resnet_encoder(spec)
    elif spec.family == "vit":
        encoder = build_vit_encoder(spec)
    else:
        raise ValueError(f"Unknown family '{spec.family}' for encoder '{name}'")

    return encoder.to(device).eval()

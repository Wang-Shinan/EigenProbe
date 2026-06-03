"""Extract raw (pre-preprocessing) features for a dataset with a frozen encoder."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..encoders.base import Encoder


@torch.no_grad()
def extract_features(
    encoder: Encoder,
    dataset: Dataset,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 4,
    amp: bool = True,
    desc: str = "extract",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder over the dataset.

    Returns:
        features: [N, D] float32 CPU tensor (raw encoder output, no preprocessing).
        labels:   [N]    int64  CPU tensor.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )
    encoder.eval()

    feats, labels = [], []
    use_amp = amp and device == "cuda"
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = encoder(images)
        feats.append(out.float().cpu())
        labels.append(targets.clone())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

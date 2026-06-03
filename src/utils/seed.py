"""Reproducibility helpers.

The proposal fixes seed=42 (notably for the random projection matrix). Every
stochastic step in the pipeline routes through here so results are reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int = 42, device: str = "cpu") -> torch.Generator:
    """A local Generator for ops that should not perturb the global RNG state
    (e.g. building the fixed random projection matrix)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

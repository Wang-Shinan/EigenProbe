"""Plotting helpers for the report figures. All save to disk and return the path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_singular_spectra(spectra: dict[str, np.ndarray], out_path, log_y: bool = True) -> Path:
    """Overlay normalized singular-value spectra for all encoders."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in spectra.items():
        s = np.asarray(s, dtype=float)
        ax.plot(np.arange(1, len(s) + 1), s / s.max(), label=name, linewidth=1.8)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value")
    if log_y:
        ax.set_yscale("log")
    ax.set_title("Singular value spectra (mean-centered, L2-normalized features)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_rank_vs_margin(rows: list[dict], out_path) -> Path:
    """Scatter effective rank (RankMe) vs SVM mean margin, labeled per encoder."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in rows:
        ax.scatter(r["rankme"], r["mean_margin"], s=80)
        ax.annotate(r["encoder"], (r["rankme"], r["mean_margin"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("RankMe effective rank")
    ax.set_ylabel("SVM mean margin (1/||w||)")
    ax.set_title("Spectral richness vs geometric separability")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_confusion(cm: np.ndarray, class_names, out_path, title: str = "Confusion") -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_tsne(embedding_2d: np.ndarray, colors, out_path, title: str = "t-SNE") -> Path:
    """Scatter a precomputed 2-D t-SNE embedding colored by instance/class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=8, cmap="tab20")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


def plot_perturbation_curves(results: dict[str, dict], out_path, perturbation: str) -> Path:
    """One sensitivity curve per encoder for a given perturbation type.

    results: {encoder: {"magnitudes": [...], "mean_sensitivity": [...]}}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        ax.plot(r["magnitudes"], r["mean_sensitivity"], marker="o", label=name)
    ax.set_xlabel(f"{perturbation} magnitude")
    ax.set_ylabel("Relative representation shift  ||f(x)-f(x')|| / ||f(x)||")
    ax.set_title(f"Latent stability under {perturbation}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)

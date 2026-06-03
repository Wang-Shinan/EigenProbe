"""Step 2 -- Part A: SVD spectral analysis + Two-NN intrinsic dimension.

For each encoder, loads cached eval features, applies the uniform preprocessing
(random projection -> L2), then computes RankMe, PCA effective rank, dimensional
collapse, and the Two-NN intrinsic dimension. Writes a JSON summary, a CSV table,
and the overlaid singular-spectrum figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _bootstrap import init

from src.analysis.intrinsic_dim import two_nn_dimension
from src.analysis.svd import analyze_svd
from src.features import build_preprocessor, load_bundle
from src.viz import plot_singular_spectra


def main():
    cfg, log = init("svd-analysis")
    out_dir = Path(cfg.paths.results_root) / cfg.dataset.name / "svd"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_split = cfg.dataset.splits.eval
    summary, spectra = {}, {}

    for name in cfg.encoders:
        bundle = load_bundle(cfg.paths.feature_cache, name, cfg.dataset.name, eval_split)
        pre = build_preprocessor(cfg, d_in=bundle.dim)
        z = pre(bundle.features)  # [N, target_dim]

        svd_res = analyze_svd(
            z,
            rankme_epsilon=cfg.svd.rankme_epsilon,
            collapse_threshold=cfg.svd.collapse_threshold,
            meta={"encoder": name},
        )
        twonn = two_nn_dimension(
            z,
            discard_fraction=cfg.intrinsic_dim.discard_fraction,
            max_points=10000,
            seed=cfg.seed,
        )

        spectra[name] = svd_res.singular_values.numpy()
        # Curvature flag: large gap between non-linear ID and linear effective rank.
        curvature_gap = svd_res.rankme - twonn.dimension
        summary[name] = {
            **svd_res.summary(),
            **twonn.summary(),
            "curvature_gap_rankme_minus_twonn": curvature_gap,
            "paradigm": bundle.meta.get("paradigm"),
        }
        log.info(
            "%-22s RankMe=%.1f  PCA-eff=%.1f  Two-NN=%.1f  collapsed=%d/%d",
            name, svd_res.rankme, svd_res.pca_effective_rank,
            twonn.dimension, svd_res.collapsed_dims, svd_res.ambient_dim,
        )

    # Persist outputs.
    (out_dir / "svd_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "spectra.npz", **{k: v for k, v in spectra.items()})
    fig = plot_singular_spectra(spectra, out_dir / "singular_spectra.png")
    log.info("Wrote %s and %s", out_dir / "svd_summary.json", fig)


if __name__ == "__main__":
    main()

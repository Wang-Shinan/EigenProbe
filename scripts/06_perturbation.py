"""Step 6 (optional) -- Part D: latent stability under input perturbation.

For a sample of test images, embeds the clean image and perturbed versions at a
range of magnitudes, then measures the relative representation shift
||f(x)-f(x')|| / ||f(x)||. Produces a sensitivity curve per encoder/perturbation.

Enable with: ... perturbation.enabled=true
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from _bootstrap import init
from torch.utils.data import DataLoader, Subset

from src.analysis.perturbation import perturbation_sensitivity
from src.data import build_dataset
from src.data.perturbations import PERTURBATIONS
from src.encoders import build_encoder, load_encoder_specs
from src.features import build_preprocessor
from src.utils.seed import get_generator
from src.viz import plot_perturbation_curves


@torch.no_grad()
def _embed(encoder, images, cfg, pre):
    feats = encoder(images.to(cfg.device)).float().cpu()
    return pre(feats)


def main():
    cfg, log = init("perturbation")
    if not cfg.perturbation.enabled:
        log.warning("perturbation.enabled=false -- skipping Part D. Set it true to run.")
        return

    specs = load_encoder_specs()
    out_dir = Path(cfg.paths.results_root) / cfg.dataset.name / "perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    full = build_dataset(cfg.dataset.name, cfg.dataset.splits.eval, cfg.paths.data_root,
                         image_size=cfg.dataset.image_size, interpolation=cfg.dataset.interpolation)
    n = min(cfg.perturbation.num_images, len(full))
    subset = Subset(full, list(range(n)))
    loader = DataLoader(subset, batch_size=cfg.features.batch_size, shuffle=False,
                        num_workers=cfg.dataset.num_workers)

    gen = get_generator(seed=cfg.seed, device="cpu")
    # {perturbation: {encoder: result_summary}}
    all_results = {p: {} for p in cfg.perturbation.types}

    for name in cfg.encoders:
        encoder = build_encoder(name, device=cfg.device, specs=specs)
        pre = build_preprocessor(cfg, d_in=encoder.feature_dim)

        for ptype in cfg.perturbation.types:
            fn = PERTURBATIONS[ptype]
            clean_chunks = []
            pert_chunks = {m: [] for m in cfg.perturbation.magnitudes}

            for images, _ in loader:
                clean_chunks.append(_embed(encoder, images, cfg, pre))
                for m in cfg.perturbation.magnitudes:
                    pert = torch.stack([fn(img, m, generator=gen) for img in images])
                    pert_chunks[m].append(_embed(encoder, pert, cfg, pre))

            clean = torch.cat(clean_chunks)
            pert_by_mag = {m: torch.cat(v) for m, v in pert_chunks.items()}
            res = perturbation_sensitivity(clean, pert_by_mag, ptype)
            all_results[ptype][name] = {
                "magnitudes": res.magnitudes,
                "mean_sensitivity": res.mean_sensitivity,
                "std_sensitivity": res.std_sensitivity,
            }
            log.info("%-22s %-16s sens@max=%.4f", name, ptype, res.mean_sensitivity[-1])

    for ptype, results in all_results.items():
        plot_perturbation_curves(results, out_dir / f"sensitivity_{ptype}.png", ptype)
    (out_dir / "perturbation_summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote %s", out_dir / "perturbation_summary.json")


if __name__ == "__main__":
    main()

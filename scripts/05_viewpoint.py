"""Step 5 (optional) -- Part C: multi-view viewpoint consistency on ModelNet40.

Embeds all `num_views` renderings of each object instance with every encoder,
applies the uniform preprocessing, and computes intra-/inter-instance distances,
the viewpoint consistency score, and a t-SNE figure colored by instance.

Enable with: ... viewpoint.enabled=true viewpoint.modelnet_root=/path/to/data
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from _bootstrap import init
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.analysis.viewpoint import viewpoint_consistency
from src.data.modelnet import ModelNet40MultiView
from src.encoders import build_encoder, load_encoder_specs
from src.features import build_preprocessor
from src.viz import plot_tsne


@torch.no_grad()
def _embed_instances(encoder, dataset, cfg):
    """Return embeddings [n_inst, n_views, raw_dim] and classes [n_inst]."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=cfg.dataset.num_workers)
    all_emb, all_cls = [], []
    for views, cls_idx, _ in tqdm(loader, desc="modelnet"):
        b, v, c, h, w = views.shape
        flat = views.view(b * v, c, h, w).to(cfg.device)
        feats = encoder(flat).float().cpu().view(b, v, -1)
        all_emb.append(feats)
        all_cls.append(cls_idx)
    return torch.cat(all_emb), torch.cat(all_cls)


def main():
    cfg, log = init("viewpoint")
    if not cfg.viewpoint.enabled:
        log.warning("viewpoint.enabled=false -- skipping Part C. Set it true to run.")
        return

    specs = load_encoder_specs()
    out_dir = Path(cfg.paths.results_root) / "modelnet40" / "viewpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ModelNet40MultiView(
        cfg.viewpoint.modelnet_root, split="test",
        num_views=cfg.viewpoint.num_views, image_size=cfg.dataset.image_size,
    )
    log.info("ModelNet40: %d instances, %d views each", len(dataset), cfg.viewpoint.num_views)

    summary = {}
    for name in cfg.encoders:
        encoder = build_encoder(name, device=cfg.device, specs=specs)
        emb, cls = _embed_instances(encoder, dataset, cfg)  # [n,v,raw_dim]

        pre = build_preprocessor(cfg, d_in=emb.shape[-1])
        n, v, _ = emb.shape
        emb_pp = pre(emb.view(n * v, -1)).view(n, v, -1)

        res = viewpoint_consistency(emb_pp, cls)
        summary[name] = res.summary()
        log.info("%-22s consistency=%.4f (intra=%.4f inter=%.4f)",
                 name, res.consistency_score, res.intra_instance, res.inter_instance)

        # t-SNE over all views, colored by instance id.
        flat = emb_pp.view(n * v, -1).numpy()
        inst_ids = torch.arange(n).repeat_interleave(v).numpy()
        tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=cfg.seed)
        emb2d = tsne.fit_transform(flat)
        plot_tsne(emb2d, inst_ids, out_dir / f"tsne_{name}.png", title=f"{name}: views by instance")

    (out_dir / "viewpoint_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "viewpoint_summary.json")


if __name__ == "__main__":
    main()

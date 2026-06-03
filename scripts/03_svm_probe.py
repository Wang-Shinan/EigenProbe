"""Step 3 -- Part B: linear SVM probe.

Fits a linear SVM on preprocessed fit-split features and evaluates on the eval
split for every encoder. Records accuracy, mean margin, support-vector count, and
the confusion matrix, plus the hardest-to-separate class pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _bootstrap import init

from src.analysis.svm_probe import linear_svm_probe
from src.data import build_dataset
from src.features import build_preprocessor, load_bundle
from src.viz import plot_confusion


def _class_names(cfg):
    ds = build_dataset(cfg.dataset.name, cfg.dataset.splits.eval, cfg.paths.data_root,
                       image_size=cfg.dataset.image_size, download=False)
    return list(getattr(ds, "classes", [str(i) for i in range(10)]))


def main():
    cfg, log = init("svm-probe")
    out_dir = Path(cfg.paths.results_root) / cfg.dataset.name / "svm"
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_split, eval_split = cfg.dataset.splits.fit, cfg.dataset.splits.eval
    class_names = _class_names(cfg)
    summary = {}

    for name in cfg.encoders:
        b_fit = load_bundle(cfg.paths.feature_cache, name, cfg.dataset.name, fit_split)
        b_eval = load_bundle(cfg.paths.feature_cache, name, cfg.dataset.name, eval_split)
        pre = build_preprocessor(cfg, d_in=b_fit.dim)

        res = linear_svm_probe(
            feats_fit=pre(b_fit.features),
            labels_fit=b_fit.labels,
            feats_eval=pre(b_eval.features),
            labels_eval=b_eval.labels,
            C=cfg.svm.C,
            max_samples_fit=cfg.svm.max_samples_fit,
            class_weight=cfg.svm.class_weight,
            seed=cfg.seed,
            meta={"encoder": name},
        )

        summary[name] = {
            **res.summary(),
            "support_vector_fraction": res.support_vector_fraction,
            "hardest_pairs": [
                (class_names[i], class_names[j], n) for (i, j, n) in res.hardest_pairs
            ],
        }
        np.save(out_dir / f"confusion_{name}.npy", res.confusion)
        plot_confusion(res.confusion, class_names, out_dir / f"confusion_{name}.png",
                       title=f"{name} (acc={res.accuracy:.3f})")
        log.info(
            "%-22s acc=%.3f  margin=%.4f  SV=%d (%.1f%%)",
            name, res.accuracy, res.mean_margin, res.n_support_vectors,
            100 * res.support_vector_fraction,
        )

    (out_dir / "svm_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "svm_summary.json")


if __name__ == "__main__":
    main()

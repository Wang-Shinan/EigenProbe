"""Step 4 -- connect SVD to SVM (the key analytical contribution).

Merges the Part A (svd_summary.json) and Part B (svm_summary.json) results, tests
the association between effective rank and SVM margin across encoders, and
highlights the architecture-controlled MAE vs I-JEPA comparison.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from _bootstrap import init

from src.analysis.connect import rank_margin_table
from src.viz import plot_rank_vs_margin


def main():
    cfg, log = init("connect-analysis")
    base = Path(cfg.paths.results_root) / cfg.dataset.name
    svd = json.loads((base / "svd" / "svd_summary.json").read_text())
    svm = json.loads((base / "svm" / "svm_summary.json").read_text())

    per_encoder = {}
    for name in cfg.encoders:
        per_encoder[name] = {
            "rankme": svd[name]["rankme"],
            "pca_effective_rank": svd[name]["pca_effective_rank"],
            "mean_margin": svm[name]["mean_margin"],
            "svm_accuracy": svm[name]["accuracy"],
        }

    res = rank_margin_table(per_encoder)
    rows = res.to_rows()

    out_dir = base / "connect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV table.
    with open(out_dir / "rank_vs_margin.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "table": rows,
        "pearson_rankme_vs_margin": res.pearson_rank_margin,
        "spearman_rankme_vs_margin": res.spearman_rank_margin,
        "mae_ijepa_delta": res.mae_ijepa_delta,
        "note": "Structural correlation across frozen encoders, NOT a causal claim.",
    }
    (out_dir / "connect_report.json").write_text(json.dumps(report, indent=2))
    plot_rank_vs_margin(rows, out_dir / "rank_vs_margin.png")

    log.info("Pearson(RankMe, margin)  = r=%.3f p=%.3f", *res.pearson_rank_margin)
    log.info("Spearman(RankMe, margin) = rho=%.3f p=%.3f", *res.spearman_rank_margin)
    if res.mae_ijepa_delta:
        d = res.mae_ijepa_delta
        log.info(
            "I-JEPA - MAE (same ViT-S/16):  dRankMe=%+.2f  dMargin=%+.4f  dAcc=%+.3f",
            d["rankme_diff_ijepa_minus_mae"],
            d["margin_diff_ijepa_minus_mae"],
            d["accuracy_diff_ijepa_minus_mae"],
        )
    log.info("Wrote %s", out_dir / "connect_report.json")


if __name__ == "__main__":
    main()

"""Step 1 -- extract and cache RAW features for every encoder x split.

Runs each frozen encoder over the fit and eval splits and saves the raw feature
matrices to features/cache. Preprocessing (random projection + L2) is applied
later at analysis time so the config can change without re-extraction.
"""
from __future__ import annotations

from _bootstrap import init

from src.data import build_dataset
from src.encoders import build_encoder, load_encoder_specs
from src.features import FeatureBundle, extract_features, save_bundle


def main():
    cfg, log = init("extract-features")
    specs = load_encoder_specs()
    splits = sorted({cfg.dataset.splits.fit, cfg.dataset.splits.eval})

    for name in cfg.encoders:
        log.info("Building encoder: %s", name)
        encoder = build_encoder(name, device=cfg.device, specs=specs)

        for split in splits:
            dataset = build_dataset(
                cfg.dataset.name,
                split=split,
                data_root=cfg.paths.data_root,
                image_size=cfg.dataset.image_size,
                interpolation=cfg.dataset.interpolation,
            )
            feats, labels = extract_features(
                encoder,
                dataset,
                device=cfg.device,
                batch_size=cfg.features.batch_size,
                num_workers=cfg.dataset.num_workers,
                amp=cfg.features.amp,
                desc=f"{name}/{split}",
            )
            bundle = FeatureBundle(
                encoder=name,
                dataset=cfg.dataset.name,
                split=split,
                features=feats,
                labels=labels,
                meta={"raw_dim": feats.shape[1], "paradigm": specs[name].paradigm},
            )
            path = save_bundle(bundle, cfg.paths.feature_cache)
            log.info("  saved %s  [%d x %d] -> %s", split, *feats.shape, path)


if __name__ == "__main__":
    main()

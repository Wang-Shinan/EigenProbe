"""Print instructions / sanity-check for the four encoder checkpoints.

The supervised ResNet-18 is fetched automatically by torchvision. The other
three require manual download from their official repos (links below); drop the
files at the paths in configs/encoders.yaml. This script verifies presence and
loads each encoder once to confirm the checkpoints parse.
"""
from __future__ import annotations

from pathlib import Path

from _bootstrap import init

from src.encoders import build_encoder, load_encoder_specs

INSTRUCTIONS = """
Download the pretrained checkpoints and place them where configs/encoders.yaml
points (default ./weights/):

  I-JEPA ViT-S/16   -> weights/ijepa_vits16.pth
      https://github.com/facebookresearch/ijepa  (Meta Research)
  MAE ViT-S/16      -> weights/mae_vits16.pth
      https://github.com/facebookresearch/mae    (Facebook Research)
  SimCLR ResNet-18  -> weights/simclr_resnet18.pth
      e.g. https://github.com/google-research/simclr (convert TF->torch) or a
      community PyTorch SimCLR checkpoint.

  ResNet-18 (supervised) downloads automatically via torchvision -- no action.
"""


def main():
    cfg, log = init("download-weights")
    Path(cfg.paths.weights_root).mkdir(parents=True, exist_ok=True)
    specs = load_encoder_specs()

    print(INSTRUCTIONS)
    missing = []
    for name in cfg.encoders:
        spec = specs[name]
        if spec.checkpoint and not Path(spec.checkpoint).exists():
            missing.append((name, spec.checkpoint))

    if missing:
        log.warning("Missing checkpoints:")
        for name, path in missing:
            log.warning("  %-22s -> %s", name, path)
        log.warning("Place the files and re-run to verify they load.")
        return

    log.info("All checkpoints present. Verifying each encoder loads on CPU...")
    for name in cfg.encoders:
        enc = build_encoder(name, device="cpu", specs=specs)
        log.info("  OK  %-22s feature_dim=%d", name, enc.feature_dim)
    log.info("All encoders loaded successfully.")


if __name__ == "__main__":
    main()

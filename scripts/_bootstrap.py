"""Shared CLI bootstrap: make `src` importable and parse `--config` + overrides.

Every script does:
    from _bootstrap import init
    cfg, log = init("script-name")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so `import src...` works without installation.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import get_logger, load_config, seed_everything  # noqa: E402


def init(name: str):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_everything(cfg.get("seed", 42))
    log = get_logger(name)
    log.info("Config: %s", cfg.to_dict())
    return cfg, log

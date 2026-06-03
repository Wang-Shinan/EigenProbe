"""Lightweight YAML config with attribute access and CLI dot-overrides.

Usage:
    cfg = load_config("configs/default.yaml",
                      overrides=["dataset.name=cifar10", "svm.C=10"])
    cfg.dataset.name        # -> "cifar10"
    cfg["svm"]["C"]         # -> 10  (int, auto-cast)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict whose keys are also accessible as attributes, recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out


def _autocast(value: str) -> Any:
    """Cast a CLI string to bool/int/float/None where unambiguous."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load YAML and apply ``key.sub=value`` overrides from the CLI."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected key.sub=value")
        key, val = ov.split("=", 1)
        _apply_override(raw, key.strip(), _autocast(val.strip()))

    return Config(raw)

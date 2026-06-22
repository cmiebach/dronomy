"""Load and access the YAML configuration as nested attribute objects."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Repo root = three levels up from this file: src/dronomy_loc/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load config.yaml. Returns a SimpleNamespace (cfg.frames.out_dir, ...)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_namespace(raw)
    cfg.repo_root = REPO_ROOT  # convenient for resolving relative paths
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p

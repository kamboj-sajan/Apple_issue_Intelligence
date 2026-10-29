"""Configuration loading: YAML file + environment overrides (+ optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass
class Paths:
    data_dir: Path

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def state(self) -> Path:
        return self.data_dir / "state"

    @property
    def artifacts(self) -> Path:
        return self.data_dir / "artifacts"


@dataclass
class Config:
    raw: dict[str, Any]
    paths: Paths
    api_key: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    ingestion: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)

    @property
    def entities_config_path(self) -> Path:
        return _resolve(self.preprocessing.get("entities_config", "configs/entities.yaml"))


def load_config(path: str | Path | None = None, data_dir: str | Path | None = None) -> Config:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = _resolve(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(cfg_path.read_text())
    resolved_data_dir = _resolve(data_dir or os.environ.get("AII_DATA_DIR") or raw["data_dir"])
    return Config(
        raw=raw,
        paths=Paths(data_dir=resolved_data_dir),
        api_key=os.environ.get("STACKEXCHANGE_KEY") or None,
        source=raw.get("source", {}),
        ingestion=raw.get("ingestion", {}),
        preprocessing=raw.get("preprocessing", {}),
        splits=raw.get("splits", {}),
    )

"""Small helpers for YAML configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dataset_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "datasets" / f"{name}.yaml")


def workload_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "workload.yaml")


def topology_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "topology.yaml")


def processed_dir(name: str) -> Path:
    return ROOT / "data" / "processed" / name


def raw_dir(name: str) -> Path:
    return ROOT / "data" / "raw" / name

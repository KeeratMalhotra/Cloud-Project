"""Simulator configuration: configs/sim.yaml plus a named system preset plus ad-hoc overrides."""
from __future__ import annotations

import copy
from typing import Any

from edgesim.config import CONFIG_DIR, load_yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg['a']['b'] = value for dotted = 'a.b'."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def sim_config(system: str | None = None, overrides: dict | None = None) -> dict:
    cfg = load_yaml(CONFIG_DIR / "sim.yaml")
    if system:
        presets = load_yaml(CONFIG_DIR / "systems.yaml")
        if system not in presets:
            raise KeyError(f"unknown system preset {system!r}; known: {sorted(presets)}")
        cfg = deep_merge(cfg, presets[system])
        cfg["system"] = system
    for dotted, value in (overrides or {}).items():
        set_path(cfg, dotted, value)
    return cfg

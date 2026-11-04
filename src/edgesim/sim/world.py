"""Static description of a city: edge sites, neighbour graph, and server inventory."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgesim.config import processed_dir, topology_config
from edgesim.topology import build_servers

STAT, MOB, CLOUD = 0, 1, 2
KIND_NAMES = ("stationary", "mobile", "cloud")
_KIND_CODE = {name: code for code, name in enumerate(KIND_NAMES)}


@dataclass
class World:
    centers: np.ndarray        # (C, 2) site centres, km
    neighbors: list[list[int]]
    kind: np.ndarray           # (S,) 0 stationary, 1 mobile, 2 cloud
    site0: np.ndarray          # (S,) initial site (-1 for cloud)
    pos0: np.ndarray           # (S, 2) initial position, km (nan for cloud)
    cores: np.ndarray          # (S,)
    ghz_per_core: np.ndarray   # (S,) before capacity.scale
    mem_mb: np.ndarray         # (S,)
    speed_mps: np.ndarray      # (S,) driving speed (0 if not mobile)
    battery_j: np.ndarray      # (S,) inf if not battery powered
    name: str = "world"

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    @property
    def n_servers(self) -> int:
        return len(self.kind)

    @property
    def adjacency(self) -> np.ndarray:
        adj = np.zeros((self.n_sites, self.n_sites), dtype=bool)
        for i, nb in enumerate(self.neighbors):
            adj[i, nb] = True
        return adj

    @classmethod
    def from_servers(cls, centers: np.ndarray, neighbors: list[list[int]], servers: pd.DataFrame,
                     name: str = "world") -> World:
        bat = servers["battery_kwh"].to_numpy(np.float64) * 3.6e6
        return cls(
            centers=np.asarray(centers, np.float64),
            neighbors=[list(map(int, n)) for n in neighbors],
            kind=servers["kind"].map(_KIND_CODE).to_numpy(np.int8),
            site0=servers["site"].to_numpy(np.int64),
            pos0=servers[["x_km", "y_km"]].to_numpy(np.float64),
            cores=servers["cores"].to_numpy(np.int64),
            ghz_per_core=servers["ghz_per_core"].to_numpy(np.float64),
            mem_mb=servers["mem_gb"].to_numpy(np.float64) * 1024.0,
            speed_mps=servers["speed_mps"].to_numpy(np.float64),
            battery_j=np.where(np.isnan(bat), np.inf, bat),
            name=name,
        )

    def summary(self) -> dict:
        out = {"sites": self.n_sites, "servers": self.n_servers}
        for code, nm in enumerate(KIND_NAMES):
            m = self.kind == code
            out[nm] = {"count": int(m.sum()), "cores": int(self.cores[m].sum()),
                       "gcycles_per_s": float((self.cores[m] * self.ghz_per_core[m]).sum())}
        return out


def load_world(dataset: str, topo_cfg: dict | None = None) -> World:
    """World for a prepared dataset (scripts/prepare_data.py), using configs/topology.yaml by default."""
    topo = json.loads((processed_dir(dataset) / "topology.json").read_text())
    centers = np.asarray(topo["centers_km"], np.float64)
    servers = build_servers(centers, topo_cfg or topology_config(), np.asarray(topo["train_site_shares"]))
    return World.from_servers(centers, topo["neighbors"], servers, name=dataset)

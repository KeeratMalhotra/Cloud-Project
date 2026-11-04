"""City geometry, edge sites (weighted K-means + Delaunay neighbours), and the server inventory.

Site indices are 0..C-1. Cloud servers have site = -1 (EdgePlus calls the cloud "cluster 0").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


@dataclass
class Frame:
    """Maps latitude/longitude to a local planar frame in km."""

    mode: str  # "square" (rescale both axes to side_km, as EdgePlus) or "true_scale"
    lat0: float
    lon0: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    side_km: float | None = None

    def _raw(self, lat, lon):
        x = (np.asarray(lon, np.float64) - self.lon0) * KM_PER_DEG_LON_EQ * np.cos(np.radians(self.lat0))
        y = (np.asarray(lat, np.float64) - self.lat0) * KM_PER_DEG_LAT
        return x, y

    def transform(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._raw(lat, lon)
        if self.mode == "square":
            x = (x - self.x_min) / (self.x_max - self.x_min) * self.side_km
            y = (y - self.y_min) / (self.y_max - self.y_min) * self.side_km
        else:
            x, y = x - self.x_min, y - self.y_min
        return x.astype(np.float32), y.astype(np.float32)

    @property
    def extent_km(self) -> tuple[float, float]:
        if self.mode == "square":
            return float(self.side_km), float(self.side_km)
        return float(self.x_max - self.x_min), float(self.y_max - self.y_min)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_frame(lat: np.ndarray, lon: np.ndarray, geometry_cfg: dict) -> Frame:
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    f = Frame(mode=geometry_cfg["mode"], lat0=lat0, lon0=lon0, x_min=0, x_max=0, y_min=0, y_max=0,
              side_km=geometry_cfg.get("side_km"))
    x, y = f._raw(lat, lon)
    f.x_min, f.x_max, f.y_min, f.y_max = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return f


def weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, seed: int, n_init: int = 10,
                    max_iter: int = 300, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with weighted k-means++ seeding. Returns (centers, labels, inertia)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(points, np.float64)
    w = np.asarray(weights, np.float64)
    best = None
    for _ in range(n_init):
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.choice(len(X), p=w / w.sum())]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for j in range(1, k):
            p = w * d2
            centers[j] = X[rng.choice(len(X), p=p / p.sum())]
            d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(1))
        for _it in range(max_iter):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            labels = dist.argmin(1)
            new = centers.copy()
            for j in range(k):
                m = labels == j
                if m.any():
                    new[j] = np.average(X[m], axis=0, weights=w[m])
            shift = float(np.abs(new - centers).max())
            centers = new
            if shift < tol:
                break
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = dist.argmin(1)
        inertia = float((w * dist[np.arange(len(X)), labels]).sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)
    centers, labels, inertia = best
    # Stable site numbering: order by x, then y.
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return centers[order], remap[labels], inertia


def delaunay_neighbors(centers: np.ndarray) -> list[list[int]]:
    """Neighbour lists from the Delaunay triangulation of the site centres."""
    tri = Delaunay(centers)
    nbrs: list[set[int]] = [set() for _ in range(len(centers))]
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = int(simplex[a]), int(simplex[b])
                nbrs[i].add(j)
                nbrs[j].add(i)
    return [sorted(n) for n in nbrs]


def nearest_site(x: np.ndarray, y: np.ndarray, centers: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int16)
    for s in range(0, len(x), chunk):
        dx = x[s:s + chunk, None] - centers[None, :, 0]
        dy = y[s:s + chunk, None] - centers[None, :, 1]
        out[s:s + chunk] = (dx * dx + dy * dy).argmin(1)
    return out


def build_servers(centers: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Server inventory: per site `stationary.per_cluster` stationary and `mobile.per_cluster` mobile, plus cloud."""
    st, mo, cl = cfg["servers"]["stationary"], cfg["servers"]["mobile"], cfg["servers"]["cloud"]
    rows = []
    for c, (cx, cy) in enumerate(centers):
        for _ in range(st["per_cluster"]):
            rows.append(dict(kind="stationary", site=c, home_site=c, x_km=cx, y_km=cy, cpu_ghz=st["cpu_ghz"],
                             mem_gb=st["mem_gb"], speed_mps=0.0, battery_kwh=np.nan))
        n_m = mo["per_cluster"]
        for j in range(n_m):
            ang = 2 * np.pi * j / max(n_m, 1)
            rows.append(dict(kind="mobile", site=c, home_site=c, x_km=cx + mo["offset_km"] * np.cos(ang),
                             y_km=cy + mo["offset_km"] * np.sin(ang), cpu_ghz=mo["cpu_ghz"], mem_gb=mo["mem_gb"],
                             speed_mps=mo["speed_kmh"] / 3.6, battery_kwh=mo["battery_kwh"]))
    for _ in range(cl["count"]):
        rows.append(dict(kind="cloud", site=-1, home_site=-1, x_km=np.nan, y_km=np.nan, cpu_ghz=cl["cpu_ghz"],
                         mem_gb=cl["mem_gb"], speed_mps=0.0, battery_kwh=np.nan))
    df = pd.DataFrame(rows)
    df.insert(0, "server_id", np.arange(len(df)))
    return df

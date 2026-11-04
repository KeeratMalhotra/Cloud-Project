"""Figures for datasets and topology (matplotlib, saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edgesim.data.labeling import SLICES  # noqa: E402

SLICE_COLORS = {"URLLC": "#d62728", "eMBB": "#1f77b4", "mMTC": "#2ca02c"}


def topology_figure(points: pd.DataFrame, centers: np.ndarray, neighbors: list[list[int]], servers: pd.DataFrame,
                    title: str, path: Path, extent_km: tuple[float, float]) -> None:
    """Points (x, y, site, size) coloured by site, Delaunay links, site centres, and edge servers."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * extent_km[1] / max(extent_km[0], 1e-9) if extent_km[0] else 7.2))
    cmap = plt.get_cmap("tab10")
    ax.scatter(points["x"], points["y"], c=[cmap(int(s) % 10) for s in points["site"]], s=points["size"],
               alpha=0.35, linewidths=0, zorder=1)
    for i, nb in enumerate(neighbors):
        for j in nb:
            if j > i:
                ax.plot(*centers[[i, j]].T, color="0.35", lw=0.8, zorder=2)
    st = servers[servers["kind"] == "stationary"]
    mo = servers[servers["kind"] == "mobile"]
    ax.scatter(mo["x_km"], mo["y_km"], marker="s", s=40, c="#2ca02c", edgecolors="k", lw=0.5, zorder=3,
               label="mobile server (start)")
    ax.scatter(st["x_km"], st["y_km"], marker="s", s=55, c="#1f4fd8", edgecolors="k", lw=0.5, zorder=4,
               label="stationary server")
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=80, c="#d62728", lw=2, zorder=5, label="site centre")
    for i, (cx, cy) in enumerate(centers):
        ax.annotate(str(i), (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=9, weight="bold")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def arrivals_figure(episodes: dict[str, pd.DataFrame], horizon_s: float, title: str, path: Path,
                    bin_s: float = 5.0) -> None:
    """Tasks per second over one episode for each load regime."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.arange(0.0, horizon_s + bin_s, bin_s)
    for name, ep in episodes.items():
        counts, _ = np.histogram(ep["t_arr"], bins=bins)
        ax.plot(bins[:-1] + bin_s / 2, counts / bin_s, label=f"{name} ({len(ep):,} tasks)", lw=1.2)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel(f"arrivals per s ({bin_s:g} s bins)")
    ax.set_title(title)
    ax.set_xlim(0, horizon_s)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def severity_figure(train_severity: np.ndarray, tier_thr: list[float], shares: dict[str, list[float]],
                    title: str, path: Path) -> None:
    """Training-day severity histogram with tier thresholds, and slice shares per split."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    lo, hi = np.quantile(train_severity, [0.001, 0.999])
    ax1.hist(np.clip(train_severity, lo, hi), bins=120, color="0.55")
    for thr, lab in zip(tier_thr, ["eMBB / mMTC", "URLLC / eMBB"]):
        ax1.axvline(thr, color="k", ls="--", lw=1)
        ax1.annotate(lab, (thr, ax1.get_ylim()[1] * 0.92), rotation=90, ha="right", va="top", fontsize=8)
    ax1.set_xlabel("severity (max robust z + rise flag)")
    ax1.set_ylabel("training readings")
    ax1.set_title(title)
    splits = list(shares)
    bottom = np.zeros(len(splits))
    for k, name in enumerate(SLICES):
        vals = np.array([shares[s][k] for s in splits]) * 100
        ax2.bar(splits, vals, bottom=bottom, color=SLICE_COLORS[name], label=name)
        for xi, (b, v) in enumerate(zip(bottom, vals)):
            ax2.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        bottom += vals
    ax2.set_ylabel("share of readings (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("slice shares by split\n(training-day thresholds)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

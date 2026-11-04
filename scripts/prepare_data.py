"""Prepare one dataset: labels, topology, and sample episodes (milestone M1).

Usage:
    python scripts/download_data.py --dataset citypulse     # fetch the Parquet mirror
    python scripts/prepare_data.py --dataset citypulse

Outputs (regenerated deterministically, so the first group is not committed):
    data/processed/<dataset>/readings.parquet   labeled readings: split, x_km, y_km, home site, slice, prio
    data/processed/<dataset>/label_params.json, topology.json, servers.csv
    results/summary/<dataset>_dataset.json      summary statistics (committed)
    docs/figures/<dataset>_*.png                topology, arrivals, severity figures (committed)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edgesim import plots, topology  # noqa: E402
from edgesim.config import dataset_config, processed_dir, topology_config, workload_config  # noqa: E402
from edgesim.data import labeling, mirror  # noqa: E402
from edgesim.data.episodes import arrival_stats, build_episode  # noqa: E402
from edgesim.data.labeling import SLICES  # noqa: E402

DISPLAY = {"citypulse": "CityPulse (Aarhus, generated)", "airdelhi": "AirDelhi (New Delhi, real)"}
EPISODE_COLS = ["ts", "sensor_id", "x_km", "y_km", "home", "slice", "prio", "severity"]


def _shares(values: np.ndarray, n: int) -> list[float]:
    counts = np.bincount(values.astype(np.int64), minlength=n)
    return (counts / max(counts.sum(), 1)).round(4).tolist()


def topology_points(df: pd.DataFrame, ds_cfg: dict) -> pd.DataFrame:
    """Points to cluster, from training rows: fixed sites, or grid cells for moving sensors."""
    tr = df[df["split"] == "train"]
    sev_pos = np.maximum(tr["severity"].to_numpy(np.float64), 0.0)
    if ds_cfg["topology_points"] == "sites":
        g = (pd.DataFrame({"sensor_id": tr["sensor_id"].to_numpy(), "x": tr["x_km"].to_numpy(),
                           "y": tr["y_km"].to_numpy(), "sev": sev_pos})
             .groupby("sensor_id").agg(x=("x", "mean"), y=("y", "mean"), n=("sev", "size"), sev=("sev", "mean")))
        g["w"] = 1.0 + g["sev"]
    else:
        cell = ds_cfg["cell_m"] / 1000.0
        g = (pd.DataFrame({"cx": np.floor(tr["x_km"].to_numpy() / cell), "cy": np.floor(tr["y_km"].to_numpy() / cell),
                           "sev": sev_pos})
             .groupby(["cx", "cy"]).agg(n=("sev", "size"), sev=("sev", "mean")).reset_index())
        g["x"] = (g["cx"] + 0.5) * cell
        g["y"] = (g["cy"] + 0.5) * cell
        g["w"] = g["n"] * (1.0 + g["sev"])
    return g[["x", "y", "n", "w"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["citypulse", "airdelhi"], required=True)
    ap.add_argument("--episode-seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = args.dataset
    ds_cfg, wl, topo_cfg = dataset_config(name), workload_config(), topology_config()
    pollutants = ds_cfg["pollutants"]
    t0 = time.monotonic()

    raw = mirror.load(mirror.fetch(name, ROOT / "data" / "mirror"))
    df, clean_report = labeling.clean(raw, pollutants, ds_cfg["valid_range"], ds_cfg.get("bbox"))
    del raw
    df["split"] = labeling.assign_splits(df["ts"], ds_cfg["splits"])
    outside = int(df["split"].isna().sum())
    df = df[df["split"].notna()].reset_index(drop=True)
    logging.info("clean: %s, outside splits: %d", clean_report, outside)

    train = df[df["split"] == "train"].reset_index(drop=True)
    params = labeling.fit_params(train, pollutants, wl["labels"], ds_cfg["rise_lag_s"])
    del train
    df = labeling.apply_labels(df, params)
    logging.info("labels: tier thresholds %s, rise threshold %.3f", params.tier_thr, params.rise_thr)

    frame = topology.fit_frame(df["lat"].to_numpy(), df["lon"].to_numpy(), ds_cfg["geometry"])
    df["x_km"], df["y_km"] = frame.transform(df["lat"].to_numpy(), df["lon"].to_numpy())
    pts = topology_points(df, ds_cfg)
    km = topo_cfg["kmeans"]
    centers, pt_site, inertia = topology.weighted_kmeans(pts[["x", "y"]].to_numpy(), pts["w"].to_numpy(),
                                                         topo_cfg["n_clusters"], km["seed"], km["n_init"],
                                                         km["max_iter"])
    neighbors = topology.delaunay_neighbors(centers)
    servers = topology.build_servers(centers, topo_cfg)
    df["home"] = topology.nearest_site(df["x_km"].to_numpy(), df["y_km"].to_numpy(), centers)
    logging.info("topology: %d sites, mean degree %.2f", len(centers), np.mean([len(n) for n in neighbors]))

    out = processed_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["sensor_id", "ts", "split", "lat", "lon", *pollutants, "x_km", "y_km", "home", "severity",
            "rise_flag", "slice", "prio"]
    df[cols].to_parquet(out / "readings.parquet", compression="zstd", index=False)
    (out / "label_params.json").write_text(json.dumps(params.to_dict(), indent=2))
    topo_json = {"frame": frame.to_dict(), "extent_km": frame.extent_km, "centers_km": centers.round(4).tolist(),
                 "neighbors": neighbors, "kmeans_inertia": inertia, "n_points": int(len(pts))}
    (out / "topology.json").write_text(json.dumps(topo_json, indent=2))
    servers.to_csv(out / "servers.csv", index=False)

    # Sample episodes on the test split, one per load regime.
    splits = {s: df.loc[df["split"] == s, EPISODE_COLS].reset_index(drop=True) for s in ds_cfg["splits"]}
    episodes, ep_summary = {}, {}
    for regime, n in wl["regimes"].items():
        ep = build_episode(splits["test"], n, args.episode_seed, ds_cfg, wl)
        episodes[regime] = ep
        site_load = np.bincount(ep["home"].to_numpy(np.int64), minlength=len(centers)) / n
        ep_summary[regime] = {
            "split": "test", "seed": args.episode_seed, "n_tasks": n, **ep.attrs,
            "arrivals_1s": arrival_stats(ep, wl["horizon_s"], 1.0),
            "arrivals_10s": arrival_stats(ep, wl["horizon_s"], 10.0),
            "slice_shares": _shares(ep["slice"].to_numpy(), 3),
            "rel_deadline_s_by_slice": [float(ep.loc[ep["slice"] == k, "rel_deadline"].mean()) for k in range(3)],
            "cycles_g_mean": float(ep["cycles"].mean() / 1e9),
            "payload_mb_mean": float(ep["payload_bits"].mean() / 8e6),
            "site_shares": site_load.round(4).tolist(),
            "busiest_site_over_mean": float(site_load.max() * len(centers)),
            "mean_rel_deadline_s": float(ep["rel_deadline"].mean()),
        }

    days = df["ts"].dt.normalize()
    split_summary = {}
    for s in ds_cfg["splits"]:
        m = (df["split"] == s).to_numpy()
        per_day = days[m].value_counts()
        split_summary[s] = {
            "rows": int(m.sum()), "days": int(per_day.size),
            "rows_per_day": {"min": int(per_day.min()), "median": float(per_day.median()), "max": int(per_day.max())},
            "sensors": int(df.loc[m, "sensor_id"].nunique()),
            "slice_shares": _shares(df.loc[m, "slice"].to_numpy(), 3),
            "prio_shares": _shares(df.loc[m, "prio"].to_numpy(), 3),
        }
    ex, ey = frame.extent_km
    summary = {
        "dataset": name, "display_name": DISPLAY[name], "clean": clean_report, "rows_outside_splits": outside,
        "sensors": int(df["sensor_id"].nunique()),
        "time_range": [str(df["ts"].min()), str(df["ts"].max())],
        "extent_km": [round(ex, 3), round(ey, 3)], "area_km2": round(ex * ey, 1),
        "pollutants": pollutants, "label_params": params.to_dict(), "splits": split_summary,
        "topology": {"n_sites": int(len(centers)), "neighbors": neighbors,
                     "mean_degree": float(np.mean([len(n) for n in neighbors])),
                     "centers_km": centers.round(3).tolist(), "points": int(len(pts)),
                     "servers": servers["kind"].value_counts().to_dict()},
        "episodes": ep_summary, "slices": list(SLICES), "runtime_s": round(time.monotonic() - t0, 1),
    }
    summ_dir = ROOT / "results" / "summary"
    summ_dir.mkdir(parents=True, exist_ok=True)
    (summ_dir / f"{name}_dataset.json").write_text(json.dumps(summary, indent=2, default=str))

    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pts_plot = pts.assign(site=pt_site, size=np.clip(np.sqrt(pts["n"]) * (4 if name == "citypulse" else 0.6), 2, 40))
    plots.topology_figure(pts_plot, centers, neighbors, servers,
                          f"{DISPLAY[name]}: {len(centers)} sites, {ex:.1f} km x {ey:.1f} km",
                          fig_dir / f"{name}_topology.png", (ex, ey))
    plots.arrivals_figure(episodes, wl["horizon_s"], f"{DISPLAY[name]}: arrivals in one test episode",
                          fig_dir / f"{name}_arrivals.png")
    train_sev = df.loc[df["split"] == "train", "severity"].to_numpy()
    plots.severity_figure(train_sev, params.tier_thr, {s: split_summary[s]["slice_shares"] for s in split_summary},
                          f"{DISPLAY[name]}: severity", fig_dir / f"{name}_severity.png")
    logging.info("done in %.1f s", time.monotonic() - t0)


if __name__ == "__main__":
    main()

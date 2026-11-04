"""Build task episodes from labeled readings (docs/CHORUS_Design.md, section 8.3, step 7).

An episode samples N readings from a window of whole consecutive days in one split and compresses the
window to `horizon_s` seconds of simulated arrivals. Footprints and deadlines are drawn from the episode
seed, so every scheduler sees exactly the same tasks for a given (dataset, split, N, seed).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from edgesim.data.labeling import SLICES

DAY_NS = 86_400 * 10**9


def _windows(day_ns: np.ndarray, counts: np.ndarray, min_days: int, need: int) -> list[tuple[int, int]]:
    """Candidate windows (i, j) over consecutive calendar days with at least `need` readings."""
    out = []
    n = len(day_ns)
    for i in range(n):
        j = min(i + min_days, n)
        total = int(counts[i:j].sum())
        while total < need and j < n:
            total += int(counts[j])
            j += 1
        contiguous = (day_ns[j - 1] - day_ns[i]) == (j - 1 - i) * DAY_NS
        if total >= need and contiguous and (j - i) >= min_days:
            out.append((i, j))
    return out


def build_episode(split_readings: pd.DataFrame, n_tasks: int, seed: int, ds_cfg: dict, wl_cfg: dict) -> pd.DataFrame:
    """Return one episode as a task table sorted by arrival time.

    `split_readings` must contain the readings of a single split with columns
    ts, sensor_id, x_km, y_km, home, slice, prio, severity.
    """
    rng = np.random.default_rng([seed, 1])
    ts_ns = split_readings["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    day = ts_ns - (ts_ns % DAY_NS)
    day_ns, counts = np.unique(day, return_counts=True)
    need = math.ceil(wl_cfg["window_oversample"] * n_tasks)
    wins = _windows(day_ns, counts, int(ds_cfg["episode_window_days"]), need)
    if not wins:
        raise ValueError(f"no window of consecutive days holds {need} readings")
    i, j = wins[rng.integers(len(wins))]
    w0, w1 = int(day_ns[i]), int(day_ns[j - 1]) + DAY_NS
    idx = np.flatnonzero((ts_ns >= w0) & (ts_ns < w1))
    pick = np.sort(rng.choice(idx, size=n_tasks, replace=False))
    ep = split_readings.iloc[pick]

    t = (ts_ns[pick] - w0) / 1e9
    jitter = float(ds_cfg.get("batch_jitter_s") or 0.0)
    if jitter > 0:
        t = t + rng.uniform(0.0, jitter, size=n_tasks)
    window_s = (w1 - w0) / 1e9
    t_arr = np.clip(t / window_s, 0.0, np.nextafter(1.0, 0.0)) * wl_cfg["horizon_s"]
    order = np.argsort(t_arr, kind="stable")
    ep = ep.iloc[order]
    t_arr = t_arr[order]

    sev = ep["severity"].to_numpy(np.float32)
    natural = ep["slice"].to_numpy(np.int8)
    mode = wl_cfg.get("slice_mode", "natural")
    if mode == "fixed":
        slice_idx = fixed_share_slices(sev, wl_cfg["target_shares"], np.random.default_rng([seed, 4]))
    elif mode == "natural":
        slice_idx = natural
    else:
        raise ValueError(f"unknown slice_mode {mode!r}")
    scale = 1.0 + wl_cfg["footprint"]["severity_scale"] * np.maximum(sev, 0.0)
    rng_fp = np.random.default_rng([seed, 2])
    rng_dl = np.random.default_rng([seed, 3])
    payload_kb = np.empty(n_tasks)
    cycles_g = np.empty(n_tasks)
    mem_mb = np.empty(n_tasks)
    rel_deadline = np.empty(n_tasks)
    for k, name in enumerate(SLICES):
        m = slice_idx == k
        cnt = int(m.sum())
        fp = wl_cfg["footprint"][name]
        payload_kb[m] = fp["payload_kb"][0] * np.exp(fp["payload_kb"][1] * rng_fp.standard_normal(cnt))
        cycles_g[m] = fp["cycles_g"][0] * np.exp(fp["cycles_g"][1] * rng_fp.standard_normal(cnt))
        mem_mb[m] = fp["mem_mb"][0] * np.exp(fp["mem_mb"][1] * rng_fp.standard_normal(cnt))
        lo, hi = wl_cfg["deadline_s"][name]
        rel_deadline[m] = rng_dl.uniform(lo, hi, size=cnt)

    tasks = pd.DataFrame({
        "task_id": np.arange(n_tasks, dtype=np.int64),
        "t_arr": t_arr,
        "deadline": t_arr + rel_deadline,
        "rel_deadline": rel_deadline,
        "slice": slice_idx,
        "prio": ep["prio"].to_numpy(np.int8),
        "severity": sev,
        "home": ep["home"].to_numpy(np.int16),
        "x_km": ep["x_km"].to_numpy(np.float32),
        "y_km": ep["y_km"].to_numpy(np.float32),
        "sensor_id": ep["sensor_id"].to_numpy(),
        "cycles": cycles_g * 1e9 * scale,
        "payload_bits": payload_kb * 8e3 * scale,
        "mem_mb": mem_mb.astype(np.float32),
    })
    nat_shares = np.bincount(natural.astype(np.int64), minlength=len(SLICES)) / n_tasks
    tasks.attrs.update({"seed": int(seed), "n_tasks": int(n_tasks), "window_start": str(pd.Timestamp(w0)),
                        "window_end": str(pd.Timestamp(w1)), "window_days": int(j - i), "slice_mode": mode,
                        "natural_slice_shares": nat_shares.round(4).tolist()})
    return tasks


def fixed_share_slices(severity: np.ndarray, shares: list[float], rng: np.random.Generator) -> np.ndarray:
    """Slice labels with exact shares: top `shares[0]` by severity -> URLLC, bottom `shares[2]` -> mMTC.

    Ties in severity (common for integer index data) are broken at random so the shares are exact.
    """
    n = len(severity)
    order = np.lexsort((rng.random(n), severity))  # ascending severity
    n_urllc, n_mmtc = round(shares[0] * n), round(shares[2] * n)
    out = np.ones(n, dtype=np.int8)
    out[order[:n_mmtc]] = 2
    out[order[n - n_urllc:]] = 0
    return out


def arrival_stats(tasks: pd.DataFrame, horizon_s: float, bin_s: float = 1.0) -> dict:
    """Arrival-rate summary: mean, coefficient of variation, and peak-to-mean ratio of tasks per bin."""
    bins = np.arange(0.0, horizon_s + bin_s, bin_s)
    counts, _ = np.histogram(tasks["t_arr"].to_numpy(), bins=bins)
    rate = counts / bin_s
    mean = float(rate.mean())
    return {"mean_per_s": mean, "cv": float(rate.std() / mean) if mean > 0 else 0.0,
            "peak_to_mean": float(rate.max() / mean) if mean > 0 else 0.0,
            "p99_per_s": float(np.quantile(rate, 0.99))}

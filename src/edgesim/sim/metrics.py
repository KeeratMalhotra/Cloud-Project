"""Episode KPIs from the simulator's per-task logs (EdgePlus Table V plus our additions)."""
from __future__ import annotations

import numpy as np

from edgesim.sim.env import REJECT_REASONS, T_CLOUD, T_HOME_MOB, T_HOME_STAT, EdgeEnv
from edgesim.sim.world import CLOUD, MOB, STAT

SLICE_NAMES = ("URLLC", "eMBB", "mMTC")


def jain(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.any(x):
        return float("nan")
    return float(x.sum() ** 2 / (x.size * (x * x).sum()))


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def summarize(env: EdgeEnv) -> dict:
    n = env.N
    ok = env.on_time
    rej = env.rejected
    ex = ~rej
    lat = env.latency[ex]
    srv = env.server_log[ex]
    kind = env.kind[srv] if srv.size else np.zeros(0, dtype=np.int64)
    tier = env.tier_log[ex]
    n_ex = max(int(ex.sum()), 1)
    out = {
        "n_tasks": int(n),
        "success": float(ok.mean()) if n else float("nan"),
        "violations": float(1.0 - ok.mean()) if n else float("nan"),
        "throughput": float(ok.sum() / env.horizon),
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "lat_mean": float(lat.mean()) if lat.size else float("nan"),
        "rejected": float(rej.mean()),
        "mask_empty": float(env.mask_empty.mean()),
        "best_effort": float(env.best_effort.mean()),
        "auto_decided": float(env.auto.mean()),
        "mean_options": float(env.n_options[env.n_options > 0].mean()) if (env.n_options > 0).any() else 0.0,
        "locality": float(np.isin(tier, (T_HOME_STAT, T_HOME_MOB)).sum() / n_ex),
        "cloud_share": float((tier == T_CLOUD).sum() / n_ex),
        "stationary_share": float((kind == STAT).sum() / n_ex),
        "mobile_share": float((kind == MOB).sum() / n_ex),
        "relocations": float(env.relocated.sum()),
        "energy_it_j": float(env.e_tx.sum() + env.e_comp.sum() + env.e_net.sum()),
        "energy_compute_j": float(env.e_comp.sum()),
        "energy_network_j": float(env.e_tx.sum() + env.e_net.sum()),
        "km_driven": float(env.km_driven), "drive_energy_j": float(env.move_j),
        "mobile_moves": int(env.n_moves), "distress_moves": int(env.n_distress_moves),
        "makespan_s": float(np.nanmax(env.finish_t)) if np.isfinite(env.finish_t).any() else float("nan"),
    }
    for k, name in enumerate(SLICE_NAMES):
        m = env.slice == k
        out[f"success_{name}"] = float(ok[m].mean()) if m.any() else float("nan")
        lk = env.latency[m & ex]
        out[f"lat_p95_{name}"] = _pct(lk, 95)
        out[f"rejected_{name}"] = float(rej[m].mean()) if m.any() else float("nan")
    for r, name in enumerate(REJECT_REASONS):
        out[f"reject_{name}"] = float((env.reject_reason == r).mean())
    for bit, name in ((1, "deadline"), (2, "budget"), (4, "quarantine"), (8, "physical")):
        out[f"empty_by_{name}"] = float(((env.empty_cause & bit) > 0).mean())
    site_ok = np.bincount(env.home, weights=ok.astype(np.float64), minlength=env.C)
    site_n = np.bincount(env.home, minlength=env.C)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["jain_site_success"] = jain(np.where(site_n > 0, site_ok / site_n, np.nan))
    out["jain_site_throughput"] = jain(site_ok)
    if env.calib is not None:
        bnd = env.bound[ex]
        out["bound_exceeded"] = float(np.mean(lat > bnd)) if lat.size else float("nan")
        for k, name in enumerate(SLICE_NAMES):
            m = env.slice[ex] == k
            out[f"bound_exceeded_{name}"] = float(np.mean(lat[m] > bnd[m])) if m.any() else float("nan")
    pe = env.pred_L[ex]
    good = np.isfinite(pe)
    out["pred_err_median_s"] = float(np.median(lat[good] - pe[good])) if good.any() else float("nan")
    out["pred_err_p95_s"] = _pct(lat[good] - pe[good], 95)
    out["share_kind_cloud_check"] = float((kind == CLOUD).sum() / n_ex)
    return out

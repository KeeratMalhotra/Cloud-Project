"""Cleaning and labeling of sensor readings (EdgePlus Algorithm 3, made concrete).

Steps (EdgePlus Alg. 3; docs/EdgePlus_Deep_Review.md, Section 2.11):
  1. clean: drop missing and non-physical values, duplicates; sort by sensor and time
  2. severity: robust z-score per pollutant (median/MAD from training days) plus a rise flag
  3. tier -> slice: thresholds at training-day severity quantiles (about 20% URLLC, 50% eMBB, 30% mMTC)
  4. priority: high / moderate / low from severity, with hysteresis per sensor

All statistics are fitted on the training split only, so validation and test labels do not leak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SLICES = ("URLLC", "eMBB", "mMTC")  # slice index 0, 1, 2
PRIORITIES = ("high", "moderate", "low")  # priority index 0, 1, 2


def clean(
    df: pd.DataFrame, pollutants: list[str], valid_range: list[float], bbox: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Drop missing, out-of-range, out-of-area, and duplicate readings; sort by sensor and time."""
    report = {"rows_in": int(len(df))}
    df = df.dropna(subset=["sensor_id", "ts", "lat", "lon", *pollutants])
    report["dropped_missing"] = report["rows_in"] - int(len(df))
    lo, hi = valid_range
    ok = np.ones(len(df), dtype=bool)
    for p in pollutants:
        v = df[p].to_numpy()
        ok &= (v >= lo) & (v <= hi)
    if bbox:
        ok &= df["lat"].between(*bbox["lat"]).to_numpy() & df["lon"].between(*bbox["lon"]).to_numpy()
    report["dropped_range_or_area"] = int((~ok).sum())
    df = df.loc[ok]
    n = len(df)
    df = df.drop_duplicates(subset=["sensor_id", "ts"], keep="first")
    report["dropped_duplicates"] = n - int(len(df))
    df = df.sort_values(["sensor_id", "ts"], kind="mergesort").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


def assign_splits(ts: pd.Series, splits: dict[str, list[str]]) -> pd.Series:
    """Map each timestamp to its split name by calendar day; days outside every split get <NA>."""
    day = ts.dt.normalize()
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name, (start, end) in splits.items():
        m = (day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))
        out[m] = name
    return out


@dataclass
class LabelParams:
    pollutants: list[str]
    winsor_hi: dict[str, float]
    median: dict[str, float]
    scale: dict[str, float]
    rise_lag_s: float
    rise_weight: float
    rise_thr: float = 0.0
    tier_thr: list[float] = field(default_factory=list)  # [low, high]: below low -> mMTC, above high -> URLLC
    prio_thr: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def robust_z(df: pd.DataFrame, params: LabelParams) -> np.ndarray:
    """Winsorized robust z-scores, shape (rows, pollutants)."""
    cols = []
    for p in params.pollutants:
        v = np.minimum(df[p].to_numpy(np.float64), params.winsor_hi[p])
        cols.append((v - params.median[p]) / params.scale[p])
    return np.stack(cols, axis=1).astype(np.float32)


def rise_score(df: pd.DataFrame, z: np.ndarray, lag_s: float) -> np.ndarray:
    """Largest per-pollutant increase of z versus the same sensor about `lag_s` seconds earlier.

    `df` must have a default RangeIndex. Readings with no earlier reading in [t - 2.5 lag, t - lag] get 0.
    """
    codes, _ = pd.factorize(df["sensor_id"])
    key = pd.DataFrame({"sensor_id": codes.astype(np.int32), "ts": df["ts"].to_numpy(),
                        "row": np.arange(len(df), dtype=np.int64)})
    left = key.assign(ts_lag=key["ts"] - pd.Timedelta(seconds=lag_s)).sort_values("ts_lag", kind="mergesort")
    right = key.rename(columns={"ts": "ts_r", "row": "row_r"}).sort_values("ts_r", kind="mergesort")
    m = pd.merge_asof(left, right, left_on="ts_lag", right_on="ts_r", by="sensor_id", direction="backward",
                      tolerance=pd.Timedelta(seconds=1.5 * lag_s))
    valid = m["row_r"].notna().to_numpy()
    lag_row = np.full(len(df), -1, dtype=np.int64)
    lag_row[m["row"].to_numpy()[valid]] = m["row_r"].to_numpy()[valid].astype(np.int64)
    rise = np.zeros(len(df), dtype=np.float32)
    has = lag_row >= 0
    rise[has] = (z[has] - z[lag_row[has]]).max(axis=1)
    return rise


def _hysteresis(groups: np.ndarray, values: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Per-group on/off state: switch on at >= enter, off below exit_, otherwise keep the previous state."""
    decisive = np.where(values >= enter, 1.0, np.where(values < exit_, 0.0, np.nan))
    state = pd.Series(decisive).groupby(groups).ffill().fillna(0.0)
    return state.to_numpy() > 0.5


def severity_of(df: pd.DataFrame, params: LabelParams) -> tuple[np.ndarray, np.ndarray]:
    z = robust_z(df, params)
    rise = rise_score(df, z, params.rise_lag_s)
    flag = rise >= params.rise_thr
    sev = z.max(axis=1) + params.rise_weight * flag.astype(np.float32)
    return sev.astype(np.float32), flag


def fit_params(train: pd.DataFrame, pollutants: list[str], label_cfg: dict, rise_lag_s: float) -> LabelParams:
    """Fit winsor limits, robust scales, rise threshold, tier and priority thresholds on training rows."""
    q_w = label_cfg["winsor_quantile"]
    winsor_hi, median, scale = {}, {}, {}
    for p in pollutants:
        raw = train[p].to_numpy(np.float64)
        winsor_hi[p] = float(np.quantile(raw, q_w))
        v = np.minimum(raw, winsor_hi[p])
        med = float(np.median(v))
        s = 1.4826 * float(np.median(np.abs(v - med)))
        if s <= 1e-9:
            s = float(np.std(v)) or 1.0
        median[p], scale[p] = med, s
    params = LabelParams(pollutants=list(pollutants), winsor_hi=winsor_hi, median=median, scale=scale,
                         rise_lag_s=float(rise_lag_s), rise_weight=float(label_cfg["rise_weight"]))
    z = robust_z(train, params)
    rise = rise_score(train, z, rise_lag_s)
    params.rise_thr = max(float(np.quantile(rise, label_cfg["rise_quantile"])), 1e-6)
    sev = z.max(axis=1) + params.rise_weight * (rise >= params.rise_thr)
    lo_q, hi_q = label_cfg["tier_quantiles"]
    params.tier_thr = [float(np.quantile(sev, lo_q)), float(np.quantile(sev, hi_q))]
    pc = label_cfg["priority"]
    params.prio_thr = {k: float(np.quantile(sev, q)) for k, q in pc.items()}
    return params


def apply_labels(df: pd.DataFrame, params: LabelParams) -> pd.DataFrame:
    """Add severity, rise_flag, slice (0 URLLC, 1 eMBB, 2 mMTC), and prio (0 high, 1 moderate, 2 low).

    `df` must be sorted by sensor and time with a default RangeIndex (as returned by `clean`).
    """
    sev, flag = severity_of(df, params)
    lo, hi = params.tier_thr
    slice_idx = np.where(sev >= hi, 0, np.where(sev >= lo, 1, 2)).astype(np.int8)
    groups, _ = pd.factorize(df["sensor_id"])
    pt = params.prio_thr
    high = _hysteresis(groups, sev, pt["high_enter"], pt["high_exit"])
    mod_or_high = _hysteresis(groups, sev, pt["mod_enter"], pt["mod_exit"])
    prio = np.where(high, 0, np.where(mod_or_high, 1, 2)).astype(np.int8)
    out = df.copy()
    out["severity"] = sev
    out["rise_flag"] = flag
    out["slice"] = slice_idx
    out["prio"] = prio
    return out

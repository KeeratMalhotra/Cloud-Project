"""Evaluate the rule-based dispatchers under one or more system presets.

Usage:
    python scripts/run_heuristics.py --dataset citypulse --split test --seeds 0 1 2 3 4
    python scripts/run_heuristics.py --dataset airdelhi --split test --systems edgeplus edgeplus_plus

Writes results/summary/heuristics_<dataset>_<split>.json (means and 95% CIs over seeds) and a CSV of
all episodes to results/raw/ (not committed).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from edgesim.agents.heuristics import HEURISTICS
from edgesim.config import workload_config
from edgesim.experiments import run_grid

ROOT = Path(__file__).resolve().parents[1]
KPIS = ["success", "violations", "success_URLLC", "success_eMBB", "success_mMTC", "lat_p50", "lat_p95",
        "lat_p99", "throughput", "cloud_share", "locality", "stationary_share", "energy_it_j", "km_driven",
        "rejected", "mask_empty", "best_effort", "relocations", "jain_site_success", "pred_err_p95_s",
        "decisions", "wall_s", "policy_us_per_decision"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="citypulse")
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="+", default=["edgeplus", "edgeplus_plus"])
    ap.add_argument("--policies", nargs="+", default=list(HEURISTICS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--regimes", nargs="+", default=None, help="subset of light / medium / stress")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="discard episodes saved by earlier runs")
    args = ap.parse_args()
    regimes = workload_config()["regimes"]
    names = args.regimes or list(regimes)
    jobs = [dict(dataset=args.dataset, split=args.split, n_tasks=regimes[r], seed=s, system=sy, policy=p)
            for sy in args.systems for p in args.policies for r in names for s in args.seeds]
    jobs.sort(key=lambda j: -j["n_tasks"])
    df = pd.DataFrame(run_grid(jobs, args.workers, log=lambda m: print(m, flush=True)))
    df["regime"] = df["n_tasks"].map({v: k for k, v in regimes.items()})
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"heuristics_{args.dataset}_{args.split}.csv"
    if raw_path.exists() and not args.fresh:  # merge with earlier chunks; new episodes replace old ones
        old = pd.read_csv(raw_path)
        key = ["system", "policy", "n_tasks", "seed"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(raw_path, index=False)
    names = [r for r in regimes if r in set(df["regime"])]

    rows = []
    for (sy, p, r), g in df.groupby(["system", "policy", "regime"], sort=False):
        row = {"system": sy, "policy": p, "regime": r, "episodes": int(len(g))}
        for k in KPIS:
            v = g[k].astype(float)
            row[k] = round(float(v.mean()), 5)
            row[f"{k}_ci95"] = round(float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), 5) if len(v) > 1 else None
        rows.append(row)
    out = {"dataset": args.dataset, "split": args.split, "seeds": args.seeds, "rows": rows}
    (ROOT / "results" / "summary" / f"heuristics_{args.dataset}_{args.split}.json").write_text(json.dumps(out, indent=2))
    show = pd.DataFrame(rows)[["system", "policy", "regime", "success", "success_ci95", "success_URLLC",
                               "lat_p50", "lat_p95", "cloud_share", "locality", "rejected", "km_driven"]]
    order = {r: n for n, r in enumerate(names)}
    show = show.sort_values(["regime", "system", "success"], key=lambda c: c.map(order) if c.name == "regime" else c,
                            ascending=[True, True, False])
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

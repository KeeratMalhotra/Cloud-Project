"""Sweep the global capacity scale (configs/sim.yaml: capacity.scale) on validation days.

Target regime (docs/Simulator.md): with the EdgePlus system preset and the local-first heuristic,
about 100% success at 50k tasks and clear overload (roughly 15-25% violations) at 150k.

Usage:
    python scripts/calibrate_capacity.py --dataset citypulse --scales 0.2 0.25 0.3 0.35 0.4 --seeds 0 1
Writes results/summary/calibration_<dataset>.json and docs/figures/calibration_<dataset>.png.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from edgesim.config import workload_config  # noqa: E402
from edgesim.experiments import run_grid  # noqa: E402

COLS = ["success", "success_URLLC", "success_eMBB", "success_mMTC", "lat_p50", "lat_p95", "cloud_share",
        "locality", "mask_empty", "rejected", "relocations", "wall_s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="citypulse")
    ap.add_argument("--split", default="val")
    ap.add_argument("--system", default="edgeplus")
    ap.add_argument("--policies", nargs="+", default=["local_first", "greedy_eft"])
    ap.add_argument("--scales", nargs="+", type=float, default=[0.2, 0.25, 0.3, 0.35, 0.4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    regimes = workload_config()["regimes"]
    jobs = [dict(dataset=args.dataset, split=args.split, n_tasks=n, seed=s, system=args.system, policy=p,
                 overrides={"capacity.scale": sc})
            for sc in args.scales for p in args.policies for n in regimes.values() for s in args.seeds]
    jobs.sort(key=lambda j: -j["n_tasks"])  # longest first for better load balance
    df = pd.DataFrame(run_grid(jobs, args.workers))
    df["regime"] = df["n_tasks"].map({v: k for k, v in regimes.items()})
    agg = (df.groupby(["policy", "cfg.capacity.scale", "regime"])[COLS].mean().reset_index()
           .sort_values(["policy", "cfg.capacity.scale", "regime"]))
    out = {"dataset": args.dataset, "split": args.split, "system": args.system, "seeds": args.seeds,
           "rows": agg.round(4).to_dict(orient="records")}
    (ROOT / "results" / "summary" / f"calibration_{args.dataset}.json").write_text(json.dumps(out, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(agg.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, len(args.policies), figsize=(5 * len(args.policies), 3.6), squeeze=False)
    for ax, p in zip(axes[0], args.policies):
        for reg in regimes:
            g = agg[(agg["policy"] == p) & (agg["regime"] == reg)]
            ax.plot(g["cfg.capacity.scale"], 100 * g["success"], marker="o", label=f"{reg} ({regimes[reg]:,})")
        ax.axhline(99.5, color="0.5", ls=":", lw=1)
        ax.axhspan(75, 85, color="0.9", zorder=0)
        ax.set_title(f"{args.dataset} {args.split}: {p}")
        ax.set_xlabel("capacity scale")
        ax.set_ylabel("on-time success (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / f"calibration_{args.dataset}.png", dpi=150)


if __name__ == "__main__":
    main()

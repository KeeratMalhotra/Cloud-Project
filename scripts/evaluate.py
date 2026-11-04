"""Evaluate trained runs (greedy policy, frozen) on validation or test episodes, in parallel.

Usage:
    python scripts/evaluate.py --runs runs/citypulse/tuneA_default runs/citypulse/tuneB_g090 \
        --split val --regimes stress medium --seeds 0 1

Each run is evaluated with the system preset and simulator overrides it was trained with. Results are
appended to <run>/eval_<dataset>_<split>.csv (one row per episode) and a summary is printed.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from edgesim.agents.ccppo import load_agent
from edgesim.config import workload_config
from edgesim.runner import episode_tasks, run_episode
from edgesim.sim.config import sim_config
from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import load_world

COLS = ["success", "success_URLLC", "success_eMBB", "success_mMTC", "lat_p50", "lat_p95", "cloud_share",
        "locality", "rejected", "mask_empty", "relocations", "energy_it_j"]


def _job(args):
    run_dir, dataset, split, n_tasks, seed = args
    torch.set_num_threads(1)
    rec = json.loads((Path(run_dir) / "config.json").read_text())
    cfg = rec["config"]
    world = load_world(dataset)
    env = EdgeEnv(world, sim_config(cfg["system"], cfg.get("sim")))
    out = run_episode(env, load_agent(Path(run_dir), world), episode_tasks(dataset, split, n_tasks, seed), seed)
    return {"run": Path(run_dir).name, "dataset": dataset, "split": split, "n_tasks": n_tasks, "seed": seed, **out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default=None, help="default: the dataset each run was trained on")
    ap.add_argument("--split", default="val")
    ap.add_argument("--regimes", nargs="+", default=["light", "medium", "stress"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    regimes = workload_config()["regimes"]
    jobs = []
    for run in args.runs:
        ds = args.dataset or json.loads((Path(run) / "config.json").read_text())["dataset"]
        jobs += [(run, ds, args.split, regimes[r], s) for r in args.regimes for s in args.seeds]
    jobs.sort(key=lambda j: -j[3])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"{n}/{len(jobs)} {r['run']} {r['n_tasks']} seed{r['seed']}: success {r['success']:.4f} "
                  f"({r['wall_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    for (run, ds, split), g in df.groupby(["run", "dataset", "split"]):
        path = next(Path(r) for r in args.runs if Path(r).name == run) / f"eval_{ds}_{split}.csv"
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old.set_index(["n_tasks", "seed"]).index.isin(g.set_index(["n_tasks", "seed"]).index)]
            g = pd.concat([old, g], ignore_index=True)
        g.to_csv(path, index=False)
    inv = {v: k for k, v in regimes.items()}
    df["regime"] = df["n_tasks"].map(inv)
    summary = df.groupby(["run", "regime"])[COLS].mean().round(4)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.to_string())


if __name__ == "__main__":
    main()

"""Evaluate trained runs (greedy policy, frozen) on validation or test episodes, in parallel.

Usage:
    python scripts/evaluate.py --runs runs/citypulse/tuneA_default runs/citypulse/tuneB_g090 \
        --split val --regimes stress medium --seeds 0 1

Each run is evaluated with the system preset and simulator overrides it was trained with. Every finished
episode is written immediately to <run>/eval_<dataset>_<split>.csv (one row per episode). Episodes already in
that file are skipped, and no new episode starts after --minutes, so large evaluations are done by repeating
the same command, just like training.
"""
from __future__ import annotations

import argparse
import json
import time
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
    ap.add_argument("--minutes", type=float, default=7.0,
                    help="stop starting episodes after this many minutes (running ones finish, about 1 min)")
    ap.add_argument("--redo", action="store_true", help="re-run episodes that are already saved")
    args = ap.parse_args()
    t_end = time.time() + 60 * args.minutes
    regimes = workload_config()["regimes"]
    jobs = []
    for run in args.runs:
        ds = args.dataset or json.loads((Path(run) / "config.json").read_text())["dataset"]
        done = set()
        path = Path(run) / f"eval_{ds}_{args.split}.csv"
        if path.exists() and not args.redo:
            done = set(map(tuple, pd.read_csv(path)[["n_tasks", "seed"]].to_numpy().tolist()))
        jobs += [(run, ds, args.split, regimes[r], s) for r in args.regimes for s in args.seeds
                 if (regimes[r], s) not in done]
    jobs.sort(key=lambda j: -j[3])
    print(f"{len(jobs)} episodes to run (already saved ones are skipped)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = list(jobs)
        running = {}
        while pending or running:
            while pending and len(running) < args.workers and time.time() < t_end:
                j = pending.pop(0)
                running[pool.submit(_job, j)] = j
            if not running:
                break
            f = next(as_completed(running))
            running.pop(f)
            r = f.result()
            rows.append(r)
            _save(r, args.runs)  # each episode is written as soon as it finishes
            print(f"{len(rows)}/{len(jobs)} {r['run']} {r['n_tasks']} seed{r['seed']}: success {r['success']:.4f} "
                  f"({r['wall_s']:.0f}s)", flush=True)
    if pending:
        print(f"time budget reached: {len(pending)} episodes left; run the same command again to continue")
    if not rows:
        return
    df = pd.DataFrame(rows)
    inv = {v: k for k, v in regimes.items()}
    df["regime"] = df["n_tasks"].map(inv)
    summary = df.groupby(["run", "regime"])[COLS].mean().round(4)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("results of this call (all saved results: scripts/status.py):")
        print(summary.to_string())


def _save(row: dict, runs: list[str]) -> None:
    path = next(Path(r) for r in runs if Path(r).name == row["run"]) / f"eval_{row['dataset']}_{row['split']}.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["n_tasks"] == row["n_tasks"]) & (old["seed"] == row["seed"]))]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


if __name__ == "__main__":
    main()

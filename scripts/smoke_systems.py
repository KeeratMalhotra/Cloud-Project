"""Quick check of the system presets with the greedy-EFT rule (one episode each, in parallel).

Usage: python scripts/smoke_systems.py --dataset citypulse --split val --n 150000 --seed 0 \
           --systems edgeplus edgeplus_pec --set ledger.consensus.mode=instant
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from edgesim.agents.heuristics import HEURISTICS
from edgesim.runner import episode_tasks, run_episode
from edgesim.sim.config import sim_config
from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import load_world

KEYS = ["success", "success_URLLC", "lat_p95", "cloud_share", "relocations", "km_driven", "energy_it_j",
        "drive_energy_j", "energy_total_j", "dvfs_slowed", "ledger_commit_p50_s", "ledger_commit_p95_s",
        "ledger_final_p50_s", "ledger_final_p95_s", "ledger_blocks", "ledger_cpu_s", "wall_s"]


def job(a):
    system, sets, dataset, split, n, seed, policy = a
    env = EdgeEnv(load_world(dataset), sim_config(system, sets))
    out = run_episode(env, HEURISTICS[policy](), episode_tasks(dataset, split, n, seed), seed)
    return system, {k: (round(out[k], 4) if isinstance(out.get(k), float) else out.get(k)) for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="citypulse")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="greedy_eft")
    ap.add_argument("--systems", nargs="+", default=["edgeplus", "edgeplus_pec"])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import yaml
    sets = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    jobs = [(s, sets, args.dataset, args.split, args.n, args.seed, args.policy) for s in args.systems]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        for system, out in pool.map(job, jobs):
            print(system, json.dumps(out))


if __name__ == "__main__":
    main()

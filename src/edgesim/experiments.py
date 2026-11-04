"""Run grids of simulator episodes in parallel worker processes.

A job is a dict with keys: dataset, split, n_tasks, seed, system, policy, and optional overrides
(dotted config paths, e.g. {"capacity.scale": 0.3}). Each worker caches worlds and readings.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from edgesim.agents import heuristics
from edgesim.runner import episode_tasks, run_episode
from edgesim.sim.config import sim_config
from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import load_world

KEYS = ("dataset", "split", "n_tasks", "seed", "system", "policy")


@lru_cache(maxsize=4)
def _world(dataset: str):
    return load_world(dataset)


@lru_cache(maxsize=16)
def _tasks(dataset: str, split: str, n_tasks: int, seed: int):
    return episode_tasks(dataset, split, n_tasks, seed)


def run_job(job: dict) -> dict:
    cfg = sim_config(job["system"], job.get("overrides"))
    env = EdgeEnv(_world(job["dataset"]), cfg)
    policy = heuristics.make(job["policy"], seed=job["seed"])
    tasks = _tasks(job["dataset"], job["split"], job["n_tasks"], job["seed"])
    out = run_episode(env, policy, tasks, seed=job["seed"])
    return {**{k: job[k] for k in KEYS}, **{f"cfg.{k}": v for k, v in (job.get("overrides") or {}).items()}, **out}


def run_grid(jobs: list[dict], workers: int | None = None, log=None) -> list[dict]:
    """Run jobs in parallel; results keep the job order. `log(msg)` gets one line per finished job."""
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    if workers == 1:
        out = []
        for n, j in enumerate(jobs, 1):
            out.append(run_job(j))
            if log:
                log(f"{n}/{len(jobs)} {_label(j)} {out[-1]['wall_s']:.1f}s")
        return out
    results: list[dict | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): k for k, j in enumerate(jobs)}
        for n, fut in enumerate(as_completed(futures), 1):
            k = futures[fut]
            results[k] = fut.result()
            if log:
                log(f"{n}/{len(jobs)} {_label(jobs[k])} {results[k]['wall_s']:.1f}s")
    return results


def _label(job: dict) -> str:
    return f"{job['system']}/{job['policy']}/{job['n_tasks']}/seed{job['seed']}"

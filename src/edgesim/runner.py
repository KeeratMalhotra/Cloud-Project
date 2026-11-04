"""Episode loading and running helpers shared by experiment scripts."""
from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd

from edgesim.config import dataset_config, processed_dir, workload_config
from edgesim.data.episodes import build_episode
from edgesim.sim.env import EdgeEnv
from edgesim.sim.metrics import summarize

EPISODE_COLS = ["ts", "sensor_id", "x_km", "y_km", "home", "slice", "prio", "severity"]


@lru_cache(maxsize=8)
def split_readings(dataset: str, split: str) -> pd.DataFrame:
    r = pd.read_parquet(processed_dir(dataset) / "readings.parquet", columns=[*EPISODE_COLS, "split"])
    return r[r["split"] == split].drop(columns=["split"]).reset_index(drop=True)


def episode_tasks(dataset: str, split: str, n_tasks: int, seed: int, workload: dict | None = None) -> pd.DataFrame:
    """Deterministic task table for (dataset, split, size, seed). A custom `workload` may shorten the horizon;
    the window is then sized for the equivalent 600 s regime so short episodes keep the same arrival rate."""
    wl = workload or workload_config()
    scale = 600.0 / float(wl["horizon_s"])
    if scale != 1.0:
        wl = {**wl, "window_oversample": wl["window_oversample"] * scale}
    return build_episode(split_readings(dataset, split), n_tasks, seed, dataset_config(dataset), wl)


def run_episode(env: EdgeEnv, policy, tasks, seed: int, slow_policy=None) -> dict:
    """Run one episode. `policy(env) -> server index`; `slow_policy(env) -> dict | None` for slow requests."""
    t0 = time.perf_counter()
    decisions = 0
    policy_s = 0.0
    env.reset(tasks, seed=seed)
    while not env.done:
        kind = env.request[0]
        if kind == "fast":
            p0 = time.perf_counter()
            a = policy(env)
            policy_s += time.perf_counter() - p0
            decisions += 1
            env.act(a)
        else:
            env.act_slow(slow_policy(env) if slow_policy is not None else None)
    out = summarize(env)
    out.update({"wall_s": time.perf_counter() - t0, "decisions": decisions,
                "policy_us_per_decision": 1e6 * policy_s / max(decisions, 1)})
    return out

"""Observation features for learned dispatchers, built from the simulator's state at a decision.

Local observation (actor input, EdgePlus Sec. IV-B4): the task, plus one row per server.
    task (TASK_DIM): slice one-hot, priority one-hot, slack, cycles, payload, memory, episode time
    server rows (S x SERVER_DIM): mask, tier one-hot, predicted latency vs slack, slack margin, queue wait,
        processing, drive time, utilization, memory used, reputation, kind one-hot, site debt,
        fastest-feasible flag, latency rank among feasible servers
Global observation (critic input only): per-site backlog, utilization, slice-budget fill, debt, parked
mobiles, plus cloud backhaul backlog and episode time.

All features are scaled to roughly [-2, 3]; no attack or ground-truth labels are included.
"""
from __future__ import annotations

import numpy as np

from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import MOB

TASK_DIM = 11
SERVER_DIM = 19
SITE_DIM = 7


def task_features(env: EdgeEnv) -> np.ndarray:
    i, t = env.i, env.now
    f = np.zeros(TASK_DIM, dtype=np.float32)
    f[int(env.slice[i])] = 1.0
    f[3 + min(int(env.prio[i]), 2)] = 1.0
    f[6] = (env.deadline[i] - t) / 600.0
    f[7] = env.cycles[i] / 1e9
    f[8] = env.payload_bits[i] / 1e7
    f[9] = env.mem_mb[i] / 256.0
    f[10] = t / env.horizon
    return f


def server_features(env: EdgeEnv, util: np.ndarray, debt: np.ndarray) -> np.ndarray:
    i, t = env.i, env.now
    slack = max(env.deadline[i] - t, 1e-3)
    S = env.S
    f = np.zeros((S, SERVER_DIM), dtype=np.float32)
    f[:, 0] = env.cur_mask
    tier = env.cur_tier
    for k in range(4):
        f[:, 1 + k] = tier == k
    L = env.cur_L
    f[:, 5] = np.clip(L / slack, 0.0, 3.0)
    f[:, 6] = np.clip((slack - L) / slack, -2.0, 1.0)
    f[:, 7] = np.clip((env.cur_start - env.cur_ready) / 30.0, 0.0, 3.0)
    f[:, 8] = np.minimum(env.cur_proc / 10.0, 3.0)
    f[:, 9] = np.minimum(env.cur_move / 60.0, 3.0)
    f[:, 10] = util
    f[:, 11] = env.mem_used / env.mem_cap
    f[:, 12] = env.rep
    kind = env.kind
    f[:, 13] = kind == 0
    f[:, 14] = kind == MOB
    f[:, 15] = kind == 2
    site = env.site
    f[:, 16] = np.where(site >= 0, debt[np.maximum(site, 0)], 0.0) / 10.0
    f[tier < 0, 5:12] = 0.0
    # Rank of the predicted latency among feasible servers: 0 = fastest, 1 = slowest or infeasible.
    feas = np.flatnonzero(env.cur_mask)
    f[:, 18] = 1.0
    if feas.size:
        order = feas[np.argsort(L[feas], kind="stable")]
        f[order[0], 17] = 1.0
        f[order, 18] = np.arange(order.size) / max(order.size - 1, 1)
    return f


def site_features(env: EdgeEnv, util: np.ndarray, debt: np.ndarray) -> np.ndarray:
    C = env.C
    g = np.zeros((C, SITE_DIM), dtype=np.float32)
    g[:, 0] = np.minimum(env.site_backlog_s() / 30.0, 5.0)
    parked = env.site >= 0
    sites = env.site[parked]
    g[:, 1] = np.bincount(sites, weights=util[parked], minlength=C) / np.maximum(np.bincount(sites, minlength=C), 1)
    rate = env.shares_c * (env.site_cap[:, None] * env.overbook) * env.window
    g[:, 2:5] = np.clip(env.comp_tok / np.maximum(rate, 1.0), -1.0, 1.0)
    g[:, 5] = debt / 10.0
    g[:, 6] = np.bincount(sites, weights=(env.kind[parked] == MOB).astype(np.float64), minlength=C) / 4.0
    return g


def observe(env: EdgeEnv, cache: dict | None = None, global_every_s: float = 0.25) -> dict:
    """Features for the pending dispatch decision (env.request == ('fast', i)).

    The global (critic-only) part changes slowly; with a `cache` dict it is recomputed at most every
    `global_every_s` seconds of simulated time.
    """
    util = env.utilization()
    debt = env.debt()
    task = task_features(env)
    servers = server_features(env, util, debt)
    if cache is not None and cache.get("env_id") == id(env) and env.now - cache.get("t", -1e9) < global_every_s \
            and env.now >= cache["t"]:
        glob = cache["global"]
    else:
        sites = site_features(env, util, debt)
        extra = np.array([max(env.bh_busy - env.now, 0.0) / 10.0, env.now / env.horizon], dtype=np.float32)
        glob = np.concatenate([sites.ravel(), extra])
        if cache is not None:
            cache["env_id"], cache["t"], cache["global"] = id(env), env.now, glob
    return {
        "task": task,
        "servers": servers,
        "global": glob,
        "mask": env.cur_mask.copy(),
        "site": int(env.home[env.i]),
        "util": util,
        "debt": debt,
    }


def global_dim(n_sites: int) -> int:
    return n_sites * SITE_DIM + 2

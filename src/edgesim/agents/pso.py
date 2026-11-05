"""Constricted particle swarm optimisation of the advisory gains (EdgePlus Sec. III-G, Eq. 13).

The search space is Theta = (lambda_N, lambda_P, lambda_L, rho_set). Each particle is scored by running the
current (frozen) policy with that advisory setting on short episodes from training days; validation and
test days are never used. A candidate replaces the current setting only if it passes the gate:

  gate "paper":    success improves and cloud share <= 0.07, failure rate <= 0.05, stationary share >= 0.80
  gate "relative": success improves by at least `min_gain` and cloud share does not rise by more than 1 pp
                   (our improvement I-5: the paper's absolute thresholds cannot hold under overload)
"""
from __future__ import annotations

import numpy as np

from edgesim.runner import episode_tasks, run_episode
from edgesim.sim.env import EdgeEnv

SPACE = {  # name: (low, high)
    "lambda_n": (0.0, 1.0),
    "lambda_p": (0.0, 1.0),
    "lambda_l": (0.0, 0.5),
    "rho_set": (0.5, 0.95),
}
# 200 s stress episodes (shorter ones never build the queues that make the advisory gains matter); the
# particles of one iteration are scored in parallel worker processes.
PSO_DEFAULTS = {"particles": 4, "iters": 2, "w_max": 0.9, "w_min": 0.4, "c1": 2.05, "c2": 2.05,
                "episode_s": 200.0, "regime": 150_000, "episodes": 1, "min_gain": 0.002, "workers": 4}


def _score_worker(job: tuple) -> dict:
    """Score one advisory setting with the frozen policy saved at `model_path` (runs in a worker process)."""
    import torch

    from edgesim.agents.ccppo import CCPPOAgent, load_agent
    from edgesim.sim.world import load_world

    model_path, dataset, cfg, params, seeds, regime, episode_s = job
    torch.set_num_threads(1)
    world = load_world(dataset)
    base = load_agent(model_path, world)
    make = lambda p: CCPPOAgent(base.model, p, world.n_sites, world.n_servers)  # noqa: E731
    return evaluate_setting(make, world, cfg, dataset, params, seeds, regime, episode_s)


def constriction(c1: float, c2: float) -> float:
    phi = c1 + c2
    return 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))  # Eq. 13d, phi > 4


def evaluate_setting(make_agent, world, cfg: dict, dataset: str, params: dict, seeds: list[int],
                     regime: int, episode_s: float) -> dict:
    """Mean KPIs of the frozen policy with advisory `params` on short training-day episodes."""
    from edgesim.config import workload_config

    wl = {**workload_config(), "horizon_s": episode_s}
    n = int(round(regime * episode_s / 600.0))
    out = []
    for s in seeds:
        env = EdgeEnv(world, {**cfg, "horizon_s": episode_s})
        out.append(run_episode(env, make_agent(params), episode_tasks(dataset, "train", n, s, wl), s))
    keys = ("success", "cloud_share", "stationary_share")
    return {k: float(np.mean([o[k] for o in out])) for k in keys}


def passes(gate: str, cand: dict, cur: dict, min_gain: float) -> bool:
    if cand["success"] <= cur["success"] + (min_gain if gate == "relative" else 0.0):
        return False
    if gate == "paper":
        return cand["cloud_share"] <= 0.07 and 1 - cand["success"] <= 0.05 and cand["stationary_share"] >= 0.80
    if gate == "relative":
        return cand["cloud_share"] <= cur["cloud_share"] + 0.01
    raise ValueError(f"unknown gate {gate!r}")


def pso_round(model_path, cfg: dict, dataset: str, current: dict, gate: str, seed: int,
              opts: dict | None = None, log=print) -> tuple[dict, dict]:
    """One PSO search around `current` for the policy checkpoint in `model_path`.

    Returns (accepted params, record of the search).
    """
    from concurrent.futures import ProcessPoolExecutor

    o = {**PSO_DEFAULTS, **(opts or {})}
    rng = np.random.default_rng([seed, 17])
    names = list(SPACE)
    lo = np.array([SPACE[k][0] for k in names])
    hi = np.array([SPACE[k][1] for k in names])
    k = constriction(o["c1"], o["c2"])
    seeds = [int(x) for x in rng.integers(1_000_000, size=o["episodes"])]
    pool = ProcessPoolExecutor(max_workers=o["workers"])

    def score_all(thetas) -> list[dict]:
        jobs = [(model_path, dataset, cfg, {**current, **dict(zip(names, th.tolist()))}, seeds, o["regime"],
                 o["episode_s"]) for th in thetas]
        return list(pool.map(_score_worker, jobs))

    x0 = np.array([current.get(nm, (SPACE[nm][0] + SPACE[nm][1]) / 2) for nm in names], dtype=np.float64)
    pos = np.vstack([x0, lo + rng.random((o["particles"] - 1, len(names))) * (hi - lo)])
    vel = (rng.random(pos.shape) - 0.5) * (hi - lo) * 0.2
    fit = score_all(pos)
    base = fit[0]
    pbest, pbest_fit = pos.copy(), list(fit)
    g = int(np.argmax([f["success"] for f in fit]))
    log(f"PSO init: best success {pbest_fit[g]['success']:.4f} (baseline {base['success']:.4f})")
    for it in range(o["iters"]):
        w = o["w_max"] - (o["w_max"] - o["w_min"]) * it / max(o["iters"], 1)  # Eq. 13a
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = k * w * vel + k * o["c1"] * r1 * (pbest - pos) + k * o["c2"] * r2 * (pbest[g] - pos)  # Eq. 13b
        pos = np.clip(pos + vel, lo, hi)  # Eq. 13c
        for j, f in enumerate(score_all(pos)):
            if f["success"] > pbest_fit[j]["success"]:
                pbest[j], pbest_fit[j] = pos[j].copy(), f
        g = int(np.argmax([f["success"] for f in pbest_fit]))
        log(f"PSO iter {it + 1}/{o['iters']}: best success {pbest_fit[g]['success']:.4f} "
            f"(baseline {base['success']:.4f})")
    pool.shutdown()
    best = dict(zip(names, pbest[g].tolist()))
    ok = passes(gate, pbest_fit[g], base, o["min_gain"])
    record = {"baseline": base, "best": pbest_fit[g], "best_params": best, "accepted": bool(ok), "gate": gate,
              "evaluations": o["particles"] * (o["iters"] + 1)}
    return ({**current, **best} if ok else current), record

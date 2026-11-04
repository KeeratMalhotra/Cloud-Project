"""Rule-based dispatchers. Each is called when env.request == ("fast", i) and returns a server index.

They only read the arrays the environment prepares for the current decision:
    cur_mask (feasible servers), cur_L (predicted latency), cur_tier (0 home stationary, 1 home mobile,
    2 mobile elsewhere, 3 cloud), cur_start / cur_ready (predicted start and data-ready times).
"""
from __future__ import annotations

import numpy as np

from edgesim.sim.env import EdgeEnv


def _argmin_masked(values: np.ndarray, mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[np.argmin(values[idx])])


class RandomFeasible:
    """Uniform choice among feasible servers."""

    name = "random_feasible"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: EdgeEnv) -> int:
        idx = np.flatnonzero(env.cur_mask)
        return int(idx[self.rng.integers(idx.size)])


class LocalFirst:
    """EdgePlus routing order: the first tier with a feasible server; inside it, the lowest predicted latency."""

    name = "local_first"

    def __call__(self, env: EdgeEnv) -> int:
        tier, mask = env.cur_tier, env.cur_mask
        for t in (0, 1, 2, 3):
            m = mask & (tier == t)
            if m.any():
                return _argmin_masked(env.cur_L, m)
        return _argmin_masked(env.cur_L, mask)


class LeastLoaded:
    """Feasible server with the shortest predicted queueing wait (ties: lowest latency)."""

    name = "least_loaded"

    def __call__(self, env: EdgeEnv) -> int:
        wait = env.cur_start - env.cur_ready
        return _argmin_masked(wait + 1e-6 * env.cur_L, env.cur_mask)


class GreedyEFT:
    """Feasible server with the earliest predicted finish time."""

    name = "greedy_eft"

    def __call__(self, env: EdgeEnv) -> int:
        return _argmin_masked(env.cur_L, env.cur_mask)


class SlackAware:
    """Best fit: keep fast resources for urgent tasks.

    Among feasible servers that finish with at least `margin` of the task's slack to spare, pick the most
    local one (tier order), then the lowest latency. If none has that margin, fall back to earliest finish.
    """

    name = "slack_aware"

    def __init__(self, margin: float = 0.25):
        self.margin = margin

    def __call__(self, env: EdgeEnv) -> int:
        i = env.i
        slack = env.deadline[i] - env.now
        safe = env.cur_mask & (env.cur_L <= (1.0 - self.margin) * slack)
        if safe.any():
            return _argmin_masked(env.cur_tier * 1e6 + env.cur_L, safe)
        return _argmin_masked(env.cur_L, env.cur_mask)


HEURISTICS = {cls.name: cls for cls in (RandomFeasible, LocalFirst, LeastLoaded, GreedyEFT, SlackAware)}


def make(name: str, seed: int = 0):
    cls = HEURISTICS[name]
    return cls(seed) if cls is RandomFeasible else cls()

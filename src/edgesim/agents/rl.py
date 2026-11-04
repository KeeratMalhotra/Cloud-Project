"""Shared machinery for learned dispatchers: rewards, a delayed-reward step buffer, and parallel environments.

Rewards arrive late: a decision's reward is known only when its task finishes (or is dropped). Each
environment keeps a ring buffer of its decisions. A step becomes usable for training once its own reward is
known and the next step exists (for the bootstrap value). Advantages use GAE truncated at the first
successor whose reward is still unknown, so no step ever waits for unrelated tasks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from edgesim.agents.features import SERVER_DIM, TASK_DIM, observe
from edgesim.runner import episode_tasks
from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import CLOUD

REWARD_DEFAULTS = {
    "w": [3.0, 2.0, 1.0],     # slice weights w_p (URLLC, eMBB, mMTC)
    "eta_on": 1.0, "eta_late": 1.0,
    "eta_lat": 0.1, "lat_cap": 3.0,
    "eta_energy": 0.05, "e_ref_j": 0.1, "e_cap": 5.0,
    "eta_cloud": 0.2,
    "eta_rep": 0.5, "rep_th": 0.6, "eta_debt": 0.1,
    "ledger_shaping": True,
}


def task_rewards(env: EdgeEnv, ids: np.ndarray, rep: np.ndarray, debt: np.ndarray, cfg: dict,
                 lagrange: np.ndarray | None = None) -> np.ndarray:
    """EdgePlus Eq. 7 (+ Eq. 15 ledger shaping, + optional Lagrangian miss penalty per slice)."""
    k = env.slice[ids]
    w = np.asarray(cfg["w"], np.float64)[k]
    ok = env.on_time[ids]
    rej = env.rejected[ids]
    r = np.where(ok, cfg["eta_on"] * w, -cfg["eta_late"] * w)
    rel = np.maximum(env.deadline[ids] - env.t_arr[ids], 1e-3)
    lat = np.where(rej, rel, env.latency[ids])
    r -= cfg["eta_lat"] * np.minimum(lat / rel, cfg["lat_cap"])
    energy = env.e_tx[ids] + env.e_comp[ids] + env.e_net[ids]
    r -= cfg["eta_energy"] * np.minimum(energy / cfg["e_ref_j"], cfg["e_cap"])
    srv = env.server_log[ids]
    cloud = (srv >= 0) & (env.kind[np.maximum(srv, 0)] == CLOUD)
    r -= cfg["eta_cloud"] * cloud
    if cfg["ledger_shaping"]:
        r -= cfg["eta_rep"] * np.maximum(0.0, cfg["rep_th"] - rep) + cfg["eta_debt"] * debt / 10.0
    if lagrange is not None:
        r -= np.asarray(lagrange)[k] * ~ok
    return (r * cfg.get("scale", 1.0)).astype(np.float32)


@dataclass
class StepBuffer:
    """Ring buffer of one environment's decisions (absolute step indices, slot = index % cap)."""

    cap: int
    n_servers: int
    global_dim: int
    head: int = 0
    tail: int = 0
    task_to_step: dict = field(default_factory=dict)

    def __post_init__(self):
        c, S = self.cap, self.n_servers
        self.task = np.zeros((c, TASK_DIM), np.float32)
        self.servers = np.zeros((c, S, SERVER_DIM), np.float16)
        self.glob = np.zeros((c, self.global_dim), np.float32)
        self.mask = np.zeros((c, S), bool)
        self.bias = np.zeros((c, S), np.float16)
        self.site = np.zeros(c, np.int16)
        self.action = np.zeros(c, np.int64)
        self.logp = np.zeros(c, np.float32)
        self.value = np.zeros(c, np.float32)
        self.reward = np.zeros(c, np.float32)
        self.rep = np.zeros(c, np.float32)
        self.debt = np.zeros(c, np.float32)
        self.slice = np.zeros(c, np.int8)
        self.expert = np.zeros(c, np.int64)   # greedy-EFT action at this decision (imitation target)
        self.bc = np.zeros(c, bool)           # step collected by the expert during the warm-start phase
        self.resolved = np.zeros(c, bool)
        self.used = np.zeros(c, bool)
        self.done = np.zeros(c, bool)
        self.overflow = 0

    def add(self, obs: dict, bias: np.ndarray, action: int, logp: float, value: float, task_id: int,
            slice_idx: int, expert: int = 0, bc: bool = False) -> None:
        if self.head - self.tail >= self.cap:  # drop the oldest unresolved step (rare; counted)
            self._drop_tail()
        j = self.head % self.cap
        self.task[j] = obs["task"]
        self.servers[j] = obs["servers"]
        self.glob[j] = obs["global"]
        self.mask[j] = obs["mask"]
        self.bias[j] = bias
        self.site[j] = obs["site"]
        self.action[j] = action
        self.logp[j] = logp
        self.value[j] = value
        self.rep[j] = obs["servers"][action, 12] if action < self.n_servers else 1.0  # reputation column
        self.debt[j] = obs["servers"][action, 16] * 10.0 if action < self.n_servers else 0.0  # debt column
        self.slice[j] = slice_idx
        self.expert[j] = expert
        self.bc[j] = bc
        self.resolved[j] = False
        self.used[j] = False
        self.done[j] = False
        self.task_to_step[task_id] = self.head
        self.head += 1

    def _drop_tail(self) -> None:
        j = self.tail % self.cap
        self.used[j] = True
        self.resolved[j] = True
        self.overflow += 1
        self.tail += 1

    def mark_episode_end(self) -> None:
        if self.head > self.tail:
            self.done[(self.head - 1) % self.cap] = True

    def resolve(self, env: EdgeEnv, task_ids: list[int], reward_cfg: dict, lagrange=None) -> None:
        steps, ids = [], []
        for tid in task_ids:
            s = self.task_to_step.pop(tid, None)
            if s is not None and s >= self.tail:
                steps.append(s)
                ids.append(tid)
        if not steps:
            return
        slots = np.asarray(steps) % self.cap
        ids = np.asarray(ids)
        self.reward[slots] = task_rewards(env, ids, self.rep[slots], self.debt[slots], reward_cfg, lagrange)
        self.resolved[slots] = True

    def ready(self, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Slots usable for training now, and their GAE advantages (truncated at unresolved successors)."""
        n = self.head - self.tail
        if n == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        slots = (self.tail + np.arange(n)) % self.cap
        res, done = self.resolved[slots], self.done[slots]
        val, rew = self.value[slots], self.reward[slots]
        nxt_val = np.append(val[1:], 0.0)
        has_next = np.append(np.ones(n - 1, bool), False) | done
        nxt_res = np.append(res[1:], False)
        adv = np.zeros(n, np.float32)
        ok = np.zeros(n, bool)
        carry = 0.0
        carry_ok = False
        for k in range(n - 1, -1, -1):
            if not res[k] or not has_next[k]:
                carry, carry_ok = 0.0, False
                continue
            v_next = 0.0 if done[k] else nxt_val[k]
            delta = rew[k] + gamma * v_next - val[k]
            chain = (not done[k]) and nxt_res[k] and carry_ok
            a = delta + (gamma * lam * carry if chain else 0.0)
            adv[k] = a
            ok[k] = True
            carry, carry_ok = a, True
        take = ok & ~self.used[slots]
        return slots[take], adv[take]

    def mark_used(self, slots: np.ndarray) -> None:
        self.used[slots] = True
        while self.tail < self.head and self.used[self.tail % self.cap]:
            self.tail += 1


class EpisodeSource:
    """Endless stream of training episodes: training days, mixed loads, a new window each time.

    `loads` maps a regime size (tasks per 600 s) to its sampling weight. With `horizon_s` < 600 the episode
    keeps the regime's arrival rate but lasts `horizon_s` seconds (n_tasks scaled down accordingly), which
    shortens reward delays and gives more, shorter episodes from more distinct windows.
    """

    def __init__(self, dataset: str, split: str, loads: dict[int, float], seed: int, horizon_s: float = 600.0):
        from edgesim.config import workload_config

        self.dataset, self.split = dataset, split
        self.sizes = np.array(list(loads), dtype=np.int64)
        p = np.array(list(loads.values()), dtype=np.float64)
        self.p = p / p.sum()
        self.rng = np.random.default_rng([seed, 5])
        self.count = 0
        self.horizon = float(horizon_s)
        self.workload = {**workload_config(), "horizon_s": self.horizon}

    def next(self):
        regime = int(self.rng.choice(self.sizes, p=self.p))
        n = int(round(regime * self.horizon / 600.0))
        seed = int(self.rng.integers(1_000_000))
        self.count += 1
        return episode_tasks(self.dataset, self.split, n, seed, self.workload), seed, regime


class VecEnvs:
    """K environments advanced in lockstep: every env always has a pending dispatch decision."""

    def __init__(self, envs: list[EdgeEnv], source: EpisodeSource, buffer_cap: int, global_dim: int):
        self.envs = envs
        self.source = source
        self.buffers = [StepBuffer(buffer_cap, envs[0].S, global_dim) for _ in envs]
        self.caches = [{} for _ in envs]
        self.episode_log: list[dict] = []
        self.ep_meta = [None] * len(envs)
        for k in range(len(envs)):
            self._new_episode(k)

    def _new_episode(self, k: int) -> None:
        env = self.envs[k]
        while True:
            tasks, seed, n = self.source.next()
            env.reset(tasks, seed=seed)
            self.buffers[k].task_to_step.clear()
            self.ep_meta[k] = {"seed": seed, "n_tasks": n}
            if not env.done:
                return

    def observe_all(self) -> list[dict]:
        return [observe(env, self.caches[k]) for k, env in enumerate(self.envs)]

    def step(self, actions, obs, biases, logps, values, reward_cfg: dict, on_episode_end=None, lagrange=None,
             experts=None, bc: bool = False):
        """Apply one action per env, record steps, resolve finished tasks, and restart finished episodes."""
        for k, env in enumerate(self.envs):
            buf = self.buffers[k]
            i = env.i
            buf.add(obs[k], biases[k], int(actions[k]), float(logps[k]), float(values[k]), i, int(env.slice[i]),
                    int(experts[k]) if experts is not None else 0, bc)
            env.act(int(actions[k]))
            buf.resolve(env, env.pop_completions(), reward_cfg, lagrange)
            if env.done:
                buf.mark_episode_end()
                if on_episode_end is not None:
                    on_episode_end(k, env, self.ep_meta[k])
                self._new_episode(k)


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var == 0 else 1.0 - float(np.var(target - pred)) / var


def lr_schedule(base: float, progress: float, final_frac: float = 0.3) -> float:
    return base * (1.0 - (1.0 - final_frac) * min(max(progress, 0.0), 1.0))


def cosine(a: float) -> float:
    return 0.5 * (1 + math.cos(math.pi * a))

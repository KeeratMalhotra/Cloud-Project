"""EdgePlus advisory layer (Sec. III-F, Eq. 9): logit biases from Nash sharing, a PID controller, and a leader.

    z~_s = z_s + lambda_N * log s_c(s) + lambda_P * pid_c(s) + lambda_L * 1[s is the leader of c(s)]

The biases only reorder feasible servers; masking happens afterwards.

Two deliberate readings (docs/EdgePlus_Deep_Review.md, Section 5.4):
  * Nash: each site's payoff depends only on its own sharing level, so the best response has the closed form
    s_c = clip(1 - alpha_N * load_c / (2 * gamma_N), s_min, s_max). One evaluation equals the iterated
    best-response fixed point.
  * PID: the error is e_c = load_c - rho_set as in Eq. 11, and the bias is -lambda_P * u_c, so an overloaded site
    is pushed down. Adding +u_c as written would send more work to overloaded sites.
The cloud has no site, so it gets no Nash, PID, or leader bias.
"""
from __future__ import annotations

import math

import numpy as np

from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import MOB

DEFAULTS = {
    "lambda_n": 0.3, "lambda_p": 0.3, "lambda_l": 0.2,
    "alpha_n": 1.0, "gamma_n": 1.0, "s_min": 0.2, "s_max": 1.2,
    "kp": 1.90, "ki": 1.20, "kd": 0.06, "rho_set": 0.7, "pid_period_s": 1.0, "integral_clamp": 5.0,
    "omega1": 0.7, "omega2": 0.3,
}


class Advisory:
    def __init__(self, n_sites: int, n_servers: int, params: dict | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.C, self.S = n_sites, n_servers
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(self.C)
        self.prev_e = np.zeros(self.C)
        self.u = np.zeros(self.C)
        self.last_t = -math.inf

    def site_load(self, env: EdgeEnv, util: np.ndarray) -> np.ndarray:
        parked = env.site >= 0
        sites = env.site[parked]
        busy = np.bincount(sites, weights=(util * env.cores)[parked], minlength=self.C)
        cores = np.bincount(sites, weights=env.cores[parked].astype(np.float64), minlength=self.C)
        return busy / np.maximum(cores, 1.0)

    def _pid(self, load: np.ndarray, now: float) -> None:
        p = self.p
        if now - self.last_t < p["pid_period_s"]:
            return
        dt = p["pid_period_s"] if not math.isfinite(self.last_t) else now - self.last_t
        e = load - p["rho_set"]
        self.integral = np.clip(self.integral + e * dt, -p["integral_clamp"], p["integral_clamp"])
        de = (e - self.prev_e) / dt if math.isfinite(self.last_t) else np.zeros_like(e)
        self.u = np.clip(p["kp"] * e + p["ki"] * self.integral + p["kd"] * de, -1.0, 1.0)
        self.prev_e = e
        self.last_t = now

    def bias(self, env: EdgeEnv, util: np.ndarray) -> np.ndarray:
        p = self.p
        load = self.site_load(env, util)
        self._pid(load, env.now)
        s_c = np.clip(1.0 - p["alpha_n"] * load / (2.0 * p["gamma_n"]), p["s_min"], p["s_max"])
        b = np.zeros(self.S, dtype=np.float32)
        parked = env.site >= 0
        sites = env.site[parked]
        b[parked] = p["lambda_n"] * np.log(s_c[sites]) - p["lambda_p"] * self.u[sites]
        if p["lambda_l"]:
            cap = env.world.battery_j
            energy = np.where(env.kind == MOB, env.battery / np.where(np.isfinite(cap), cap, 1.0), 1.0)
            score = p["omega1"] * (1.0 - util[parked]) + p["omega2"] * energy[parked]
            best = np.full(self.C, -np.inf)
            np.maximum.at(best, sites, score)
            lead = np.flatnonzero(parked)[score >= best[sites]]
            b[lead] += p["lambda_l"]
        return b

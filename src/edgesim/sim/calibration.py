"""Online conformal calibration of latency predictions (adaptive conformal inference).

Groups are (slice, server kind). For a prediction L_hat the calibrated upper bound is
    U = L_hat + q_g * max(L_hat, floor)    (relative score)   or   U = L_hat + q_g    (absolute score)
where q_g is the (1 - alpha_g) empirical quantile of recent scores in group g. After each completion,
alpha_g <- alpha_g + gamma * (target_g - 1[L > U]) (Gibbs and Candes, 2021), which keeps the long-run
share of tasks exceeding their bound near target_g even when the load drifts.
"""
from __future__ import annotations

import numpy as np


class ConformalCalibrator:
    def __init__(self, n_slices: int, n_kinds: int, cfg: dict):
        self.n_kinds = n_kinds
        self.G = n_slices * n_kinds
        self.relative = cfg["score"] == "relative"
        self.floor = float(cfg["floor_s"])
        self.window = int(cfg["window"])
        self.gamma = float(cfg["gamma"])
        self.target = np.repeat(np.asarray(cfg["target_miss"], np.float64), n_kinds)
        self.refresh = int(cfg["refresh_every"])
        self.warmup = int(cfg["warmup"])
        self.buf = np.zeros((self.G, self.window))
        self.count = np.zeros(self.G, dtype=np.int64)
        self.since = np.zeros(self.G, dtype=np.int64)
        self.alpha = self.target.copy()
        self.q = np.full(self.G, float(cfg["init_q"]))
        self.errors = np.zeros(self.G, dtype=np.int64)
        self.updates = np.zeros(self.G, dtype=np.int64)

    def group(self, slice_idx, kind):
        return np.asarray(slice_idx) * self.n_kinds + np.asarray(kind)

    def bound(self, l_hat: np.ndarray, g: np.ndarray) -> np.ndarray:
        q = self.q[g]
        if self.relative:
            return l_hat + q * np.maximum(l_hat, self.floor)
        return l_hat + q

    def update(self, g: int, l_hat: float, l_real: float, u_used: float) -> None:
        score = (l_real - l_hat) / max(l_hat, self.floor) if self.relative else l_real - l_hat
        self.buf[g, self.count[g] % self.window] = score
        self.count[g] += 1
        err = 1.0 if l_real > u_used else 0.0
        self.errors[g] += int(err)
        self.updates[g] += 1
        self.alpha[g] += self.gamma * (self.target[g] - err)
        self.since[g] += 1
        if self.since[g] >= self.refresh and self.count[g] >= self.warmup:
            self.since[g] = 0
            n = min(self.count[g], self.window)
            level = 1.0 - self.alpha[g]
            vals = self.buf[g, :n]
            if level >= 1.0:
                self.q[g] = float(vals.max()) + 1e-3
            elif level <= 0.0:
                self.q[g] = float(vals.min())
            else:
                self.q[g] = float(np.quantile(vals, level, method="higher"))

    def report(self) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            miss = np.where(self.updates > 0, self.errors / np.maximum(self.updates, 1), np.nan)
        return {"q": self.q.round(4).tolist(), "alpha": self.alpha.round(4).tolist(),
                "observed_miss": np.round(miss, 4).tolist(), "updates": self.updates.tolist()}

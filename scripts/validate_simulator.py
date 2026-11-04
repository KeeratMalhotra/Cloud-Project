"""Check the simulator's queue engine against closed-form queueing results.

Single site, one server with c cores, Poisson arrivals, no network or masks. Measured mean wait in queue
(start - data ready) is compared with:
  1. M/M/c FIFO, Erlang C                     (c = 1, 4, 16; rho = 0.5, 0.7, 0.9)
  2. M/G/1 FIFO, Pollaczek-Khinchine          (lognormal service, sigma = 1; rho = 0.5, 0.7, 0.9)
  3. M/M/1 non-preemptive priority, Cobham    (two classes, 50/50, rho = 0.8)

Usage: python scripts/validate_simulator.py [--tasks 200000] [--seeds 3] [--workers 4]
Writes results/summary/sim_validation.json.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from edgesim.sim.config import sim_config
from edgesim.sim.env import EdgeEnv
from edgesim.sim.world import World

ROOT = Path(__file__).resolve().parents[1]
MEAN_S = 1.0          # mean service time (s)
F_CORE_GHZ = 1.0      # so cycles = MEAN_S * 1e9


def erlang_c_wait(lam: float, mean_s: float, c: int) -> float:
    a = lam * mean_s
    rho = a / c
    tail = a ** c / math.factorial(c) / (1 - rho)
    p_wait = tail / (sum(a ** k / math.factorial(k) for k in range(c)) + tail)
    return p_wait * mean_s / (c * (1 - rho))


def one_server_world(cores: int) -> World:
    servers = pd.DataFrame([dict(kind="stationary", site=0, x_km=0.0, y_km=0.0, cores=cores,
                                 ghz_per_core=F_CORE_GHZ, mem_gb=1e9, speed_mps=0.0, battery_kwh=np.nan)])
    return World.from_servers(np.zeros((1, 2)), [[]], servers, name="validation")


def tasks_poisson(n: int, lam: float, seed: int, slices: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng([seed, 99])
    t = np.cumsum(rng.exponential(1.0 / lam, n))
    sl = np.zeros(n, dtype=np.int64) if slices is None else slices
    return {"t_arr": t, "deadline": t + 1e12, "slice": sl, "prio": np.zeros(n, dtype=np.int64),
            "home": np.zeros(n, dtype=np.int64), "x_km": np.zeros(n), "y_km": np.zeros(n),
            "cycles": np.full(n, MEAN_S * F_CORE_GHZ * 1e9), "payload_bits": np.zeros(n),
            "mem_mb": np.zeros(n)}


def run_case(case: dict) -> dict:
    over = {"capacity.scale": 1.0, "slow_step_s": 0.0, "mask.level": "none", "mobility.mode": "none",
            "network.lan_delay_s": 0.0, "network.downlink_s": [0.0, 0.0, 0.0],
            "noise.service": case["noise"], "queue.discipline": case["discipline"], "ledger.feedback": False}
    env = EdgeEnv(one_server_world(case["cores"]), sim_config(None, over))
    n = case["tasks"]
    slices = None
    if case.get("two_class"):
        slices = np.where(np.random.default_rng([case["seed"], 7]).random(n) < 0.5, 0, 2)
    env.reset(tasks_poisson(n, case["lam"], case["seed"], slices), seed=case["seed"])
    if not env.done:
        raise RuntimeError("validation episode needs no decisions")
    warm = n // 10
    wq = (env.start_t - env.ready_t)[warm:]
    out = {**{k: v for k, v in case.items() if k != "noise"}, "wq_sim": float(wq.mean())}
    if case.get("two_class"):
        sl = slices[warm:]
        out["wq_sim_high"] = float(wq[sl == 0].mean())
        out["wq_sim_low"] = float(wq[sl == 2].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    expo = {"dist": "exponential", "sigma": 0.0, "pareto_alpha": 2.5}
    logn = {"dist": "lognormal", "sigma": 1.0, "pareto_alpha": 2.5}
    cases = []
    for seed in range(args.seeds):
        for c in (1, 4, 16):
            for rho in (0.5, 0.7, 0.9):
                cases.append(dict(test="M/M/c FIFO", cores=c, rho=rho, lam=rho * c / MEAN_S, noise=expo,
                                  discipline="fifo", tasks=args.tasks, seed=seed))
        for rho in (0.5, 0.7, 0.9):
            cases.append(dict(test="M/G/1 FIFO (lognormal sigma=1)", cores=1, rho=rho, lam=rho / MEAN_S, noise=logn,
                              discipline="fifo", tasks=args.tasks, seed=seed))
        cases.append(dict(test="M/M/1 priority (2 classes)", cores=1, rho=0.8, lam=0.8 / MEAN_S, noise=expo,
                          discipline="slice_priority", tasks=args.tasks, seed=seed, two_class=True))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_case, cases))

    rows = []
    df = pd.DataFrame(runs)
    for (test, cores, rho), g in df.groupby(["test", "cores", "rho"], sort=False):
        lam = rho * cores / MEAN_S
        if test.startswith("M/M/c"):
            pairs = [("wq", erlang_c_wait(lam, MEAN_S, int(cores)), g["wq_sim"])]
        elif test.startswith("M/G/1"):
            es2 = MEAN_S ** 2 * math.exp(1.0)  # lognormal sigma = 1, mean 1
            pairs = [("wq", lam * es2 / (2 * (1 - rho)), g["wq_sim"])]
        else:
            lam_i = lam / 2
            w0 = 2 * lam_i * (2 * MEAN_S ** 2) / 2
            rho1 = lam_i * MEAN_S
            pairs = [("wq_high", w0 / (1 - rho1), g["wq_sim_high"]),
                     ("wq_low", w0 / ((1 - rho1) * (1 - rho)), g["wq_sim_low"])]
        for metric, theory, sims in pairs:
            sims = sims.to_numpy()
            mean = float(sims.mean())
            half = float(1.96 * sims.std(ddof=1) / math.sqrt(len(sims))) if len(sims) > 1 else float("nan")
            rows.append({"test": test, "cores": int(cores), "rho": float(rho), "metric": metric,
                         "theory_s": round(theory, 4), "sim_s": round(mean, 4), "ci95_s": round(half, 4),
                         "rel_error": round((mean - theory) / theory, 4)})
    out = {"tasks_per_run": args.tasks, "seeds": args.seeds, "warmup_fraction": 0.1, "rows": rows}
    path = ROOT / "results" / "summary" / "sim_validation.json"
    path.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['test']:<32} c={r['cores']:<3} rho={r['rho']:.1f} {r['metric']:<8} theory {r['theory_s']:>8.4f}"
              f"  sim {r['sim_s']:>8.4f} +- {r['ci95_s']:.4f}  rel.err {100 * r['rel_error']:+.1f}%")


if __name__ == "__main__":
    main()

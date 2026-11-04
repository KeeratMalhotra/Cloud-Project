"""Event-driven simulator of the shared 6G edge-cloud system (docs/Simulator.md).

One instance runs one episode at a time. Tasks arrive in time order; for each task the environment builds
the candidate servers, predicted latencies, and the feasibility mask, then asks the policy for a decision
(`request == ("fast", task_index)`). Slow decisions (slice shares, mobile moves) happen every `slow_step_s`,
either by a built-in rule or by an external controller (`request == ("slow", time)`).

Servers are multi-core: each task runs on one core. Queued tasks wait in a per-server queue ordered FIFO
(by the time their data is ready) or by slice priority. Service, uplink, and driving times are random with
per-task factors drawn from the episode seed, so every scheduler faces the same randomness.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from edgesim.sim.calibration import ConformalCalibrator
from edgesim.sim.radio import uplink_time_s
from edgesim.sim.world import CLOUD, MOB, STAT, World

INF = math.inf
DRIVING = -2
N_SLICES = 3
T_HOME_STAT, T_HOME_MOB, T_OTHER_MOB, T_CLOUD = 0, 1, 2, 3
# rejection reasons
R_POLICY, R_INFEASIBLE, R_NO_RESOURCE, R_INVALID = 0, 1, 2, 3
REJECT_REASONS = ("policy", "infeasible", "no_resource", "invalid")
# bits of empty_cause: which single check removes every physically possible server
C_DEADLINE, C_BUDGET, C_QUARANTINE, C_PHYSICAL = 1, 2, 4, 8

TASK_FIELDS = ("t_arr", "deadline", "slice", "prio", "home", "x_km", "y_km", "cycles", "payload_bits", "mem_mb")


def _mean_one_factors(rng: np.random.Generator, n: int, cfg: dict) -> np.ndarray:
    dist = cfg["dist"]
    if dist == "none":
        return np.ones(n)
    if dist == "lognormal":
        s = float(cfg["sigma"])
        return np.exp(s * rng.standard_normal(n) - 0.5 * s * s)
    if dist == "exponential":
        return rng.exponential(1.0, n)
    if dist == "pareto":
        a = float(cfg["pareto_alpha"])
        return (1.0 + rng.pareto(a, n)) * (a - 1.0) / a
    raise ValueError(f"unknown noise distribution {dist!r}")


class EdgeEnv:
    def __init__(self, world: World, cfg: dict):
        self.world, self.cfg = world, cfg
        self.C, self.S = world.n_sites, world.n_servers
        self.REJECT = self.S
        self.kind = world.kind.astype(np.int64)
        self.is_stat, self.is_mob, self.is_cloud = self.kind == STAT, self.kind == MOB, self.kind == CLOUD
        self.cores = world.cores.astype(np.int64)
        self.f_core = world.ghz_per_core * 1e9 * float(cfg["capacity"]["scale"])
        self.cap = self.cores * self.f_core
        self.mem_cap = world.mem_mb.astype(np.float64)
        self.v = world.speed_mps.astype(np.float64)
        self.adj = world.adjacency
        self.centers = world.centers
        self.neighbors = world.neighbors

        net, en, q, m = cfg["network"], cfg["energy"], cfg["queue"], cfg["mask"]
        self.lan = float(net["lan_delay_s"])
        self.bh_rate = float(net["backhaul_gbps"]) * 1e9
        self.wan = float(net["wan_delay_s"])
        self.downlink_kind = np.asarray(net["downlink_s"], np.float64)
        self.downlink = self.downlink_kind[self.kind]
        self.road = float(net["road_factor"])
        self.p_tx = float(en["device_tx_w"])
        self.j_cycle = np.asarray(en["j_per_cycle"], np.float64)[self.kind]
        self.j_bit = float(en["backhaul_j_per_bit"])
        self.j_drive = float(en["drive_j_per_m"])
        self.prio_mode = q["discipline"] == "slice_priority"
        if q["discipline"] not in ("fifo", "slice_priority"):
            raise ValueError(f"unknown queue discipline {q['discipline']!r}")
        self.predictor = q["predictor"]
        self.kg = q["kingman"]
        self.level = m["level"]
        self.deadline_mode = m["deadline"]
        self.use_budgets = bool(m["slice_budgets"]) and self.level == "full"
        self.ledger_on = bool(cfg["ledger"]["feedback"])
        self.use_quarantine = bool(m["quarantine"]) and self.ledger_on and self.level == "full"
        self.infeasible = m["infeasible"]
        self.calibrated = bool(m["calibrated"])
        self.allow_reject = bool(cfg["dispatch"]["allow_reject"])
        self.auto_single = bool(cfg["dispatch"]["auto_single"])
        sl, mo, lg = cfg["slices"], cfg["mobility"], cfg["ledger"]
        self.window = float(sl["window_s"])
        self.overbook = float(sl["overbook"])
        self.airtime = float(sl["airtime_per_site"])
        self.default_shares_c = np.tile(np.asarray(sl["shares_compute"], np.float64), (self.C, 1))
        self.default_shares_r = np.tile(np.asarray(sl["shares_radio"], np.float64), (self.C, 1))
        self.mobility = mo["mode"]
        self.pull_on_assign = bool(mo["pull_on_assign"])
        self.bat_floor = float(mo["battery_reserve"]) * world.battery_j
        self.slow_dt = float(cfg["slow_step_s"])
        self.horizon = float(cfg["horizon_s"])
        self.rep_beta, self.rep_kappa = float(lg["ema"]), float(lg["kappa"])
        self.rep_urllc = float(lg["rep_urllc_threshold"])
        self.debt_tau = float(lg["debt_tau_s"])
        # Energy-saving options (EdgePlus-PEC). Off by default, so EdgePlus runs exactly as before.
        dv = en.get("dvfs", {}) or {}
        self.dvfs_on = bool(dv.get("enabled", False))
        self.dvfs_levels = sorted(float(x) for x in dv.get("levels", [1.0]))
        self.dvfs_margin = float(dv.get("margin", 0.5))
        # serve_remote: a neighbouring site's mobile serves the task where it is parked; the data goes over the
        # metro link between sites (FIFO per home site) instead of the mobile driving over (EdgePlus Eq. 1, L_move).
        self.serve_remote = bool(mo.get("serve_remote", False))
        self.metro_rate = float(net.get("metro_gbps", 1.0)) * 1e9
        self.metro_delay = float(net.get("metro_delay_s", 0.002))
        self.distress_persist = int(mo.get("distress_persist_steps", 1))
        self.donor_near = mo.get("donor_order", "calm") == "near"
        # Ledger timing (EdgePlus Sec. III-M). "instant": reputation and debt change at task completion (the
        # idealised reading used before Nov 2026). "blocks": they change when the record's partial block is
        # committed by edge PoA, as Sec. III-M.7 states; cloud BFT finality is tracked for the KPIs.
        lc = lg.get("consensus", {}) or {}
        self.ledger_mode = lc.get("mode", "instant")
        if self.ledger_mode not in ("instant", "blocks"):
            raise ValueError(f"unknown ledger.consensus.mode {self.ledger_mode!r}")
        if self.ledger_mode == "blocks":
            cu = lc["crypto_us"]
            self.c_sign, self.c_verify, self.c_tag = cu["sign"] * 1e-6, cu["verify"] * 1e-6, cu["tag"] * 1e-6
            self.c_hash0, self.c_hash1 = cu["hash_fixed"] * 1e-6, cu["hash_per_tuple"] * 1e-6
            self.blk_min, self.blk_max = float(lc["block_min"]), float(lc["block_max"])
            self.blk_rho_max, self.blk_write_max = float(lc["rho_max"]), float(lc["write_max"])
            self.blk_timeout = float(lc["timeout_s"]) if lc.get("timeout_s") else INF
            self.n_val = int(lc["validators"])
            self.quorum = self.n_val if lc.get("quorum", "all") == "all" else self.n_val * 2 // 3 + 1
            self.g_period = float(lc["global_period_s"])
            self.g_eager = lc.get("global_trigger", "periodic") == "eager"
            n_rep = int(lc["bft_replicas"])
            q = 2 * ((n_rep - 1) // 3) + 1
            one_way = float(lc["cloud_rtt_s"]) / 2.0
            # PBFT: pre-prepare (primary signs, replicas verify), then prepare and commit (each replica signs
            # once and verifies a quorum of q messages per phase); three one-way message delays.
            self.t_bft = 3 * one_way + self.c_sign + self.c_verify + 2 * (self.c_sign + q * self.c_verify)
            self.bft_cpu = self.c_sign + n_rep * self.c_verify + 2 * n_rep * (self.c_sign + q * self.c_verify)
        self.request = None
        self.done = True
        self.episode_id = 0

    # ------------------------------------------------------------------ episode setup
    def reset(self, tasks, seed: int = 0):
        """Start an episode. `tasks` is a mapping (or DataFrame) with the columns in TASK_FIELDS, sorted by t_arr."""
        cfg = self.cfg
        self.episode_id += 1
        for f in TASK_FIELDS:
            setattr(self, f, np.asarray(tasks[f]))
        self.slice = self.slice.astype(np.int64)
        self.home = self.home.astype(np.int64)
        self.N = N = len(self.t_arr)
        if N and np.any(np.diff(self.t_arr) < 0):
            raise ValueError("tasks must be sorted by arrival time")
        rng = np.random.default_rng([int(seed), 11])
        self.svc_noise = _mean_one_factors(rng, N, cfg["noise"]["service"])
        self.tx_noise = _mean_one_factors(rng, N, {"dist": "lognormal", "sigma": cfg["noise"]["tx_sigma"]})
        self.drive_noise = _mean_one_factors(rng, N, {"dist": "lognormal", "sigma": cfg["noise"]["drive_sigma"]})
        cell_d = cfg["network"]["cell_radius_km"] * np.sqrt(rng.random(N))
        self.ltx_pred = uplink_time_s(self.payload_bits, self.slice, cell_d, cfg["network"])

        S, C, w = self.S, self.C, self.world
        self.site = w.site0.astype(np.int64).copy()
        self.dest = self.site.copy()
        self.arrive_t = np.zeros(S)
        self.pos = w.pos0.copy()
        self.free = self.cores.copy()
        self.clock = np.zeros(S)
        self.transit = [[] for _ in range(S)]
        self.readyq = [[] for _ in range(S)]
        self.running = [[] for _ in range(S)]
        self.q_cnt = np.zeros((S, N_SLICES), dtype=np.int64)
        self.q_cyc = np.zeros((S, N_SLICES))
        self.run_ef_sum = np.zeros(S)
        self.run_ef = np.zeros(N)
        self.mem_used = np.zeros(S)
        self.battery = w.battery_j.copy()
        self.util = np.zeros(S)
        self.util_t = np.zeros(S)
        mean_cyc = float(np.mean(self.cycles)) if N else 1e9
        self.svc_m1 = mean_cyc / self.f_core
        self.svc_m2 = self.svc_m1 ** 2 * (1.0 + self.kg["cs2_init"])
        self.rep = np.ones(S)
        self.offered = np.zeros((S, N_SLICES))     # EWMA of offered work per slice, core-seconds per second
        self.offered_t = np.zeros(S)
        self.ver = np.zeros(S, dtype=np.int64)
        self.evheap = []
        self.seq = 0
        self.bh_busy = 0.0
        self.shares_c = self.default_shares_c.copy()
        self.shares_r = self.default_shares_r.copy()
        self._recompute_site_cap()
        self.comp_tok = self.shares_c * self.site_cap[:, None] * self.overbook * self.window
        self.radio_tok = self.shares_r * self.airtime * self.window
        self.bucket_t = 0.0
        self.thr = np.zeros(C)
        self.thr_t = np.zeros(C)
        self.calib = ConformalCalibrator(N_SLICES, 3, cfg["mask"]["calibration"]) if self.calibrated else None

        self.server_log = np.full(N, -1, dtype=np.int64)
        self.tier_log = np.full(N, -1, dtype=np.int8)
        self.assign_t = np.full(N, np.nan)
        self.ready_t = np.full(N, np.nan)
        self.start_t = np.full(N, np.nan)
        self.finish_t = np.full(N, np.nan)
        self.latency = np.full(N, np.nan)
        self.on_time = np.zeros(N, dtype=bool)
        self.rejected = np.zeros(N, dtype=bool)
        self.reject_reason = np.full(N, -1, dtype=np.int8)
        self.mask_empty = np.zeros(N, dtype=bool)
        self.empty_cause = np.zeros(N, dtype=np.int8)
        self.best_effort = np.zeros(N, dtype=bool)
        self.auto = np.zeros(N, dtype=bool)
        self.n_options = np.zeros(N, dtype=np.int16)
        self.pred_L = np.full(N, np.nan)
        self.bound = np.full(N, np.nan)
        self.move_pred = np.zeros(N)
        self.relocated = np.zeros(N, dtype=bool)
        self.e_tx = np.zeros(N)
        self.e_comp = np.zeros(N)
        self.e_net = np.zeros(N)
        self.e_move = np.zeros(N)          # driving energy caused by this task's assignment (EdgePlus Eq. 2)
        self.dvfs_lvl = np.ones(N)         # core speed factor the task ran at
        self.remote = np.zeros(N, dtype=bool)  # served by a neighbouring site's mobile over the metro link
        self.metro_busy = np.zeros(C)
        self.distress_cnt = np.zeros(C, dtype=np.int64)
        if self.ledger_on and self.ledger_mode == "blocks":
            self.lg_pend = [[] for _ in range(C)]   # tuples waiting for the next partial block, per cluster
            self.lg_first = np.full(C, INF)          # write time of the oldest waiting tuple
            self.lg_last = np.full(C, -INF)          # write time of the newest waiting tuple
            self.lg_busy = np.zeros(C)               # PoA validation of a cluster's blocks is sequential
            self.lg_heap = []                         # (commit time, seq, cluster, task ids)
            self.lg_seq = 0
            self.lg_count = np.zeros(C)              # tuples written since the last slow step
            self.lg_rate = np.zeros(C)               # write rate per cluster, tuples/s
            self.lg_target = np.full(C, self.blk_min)
            self.lg_write_t = np.full(N, np.nan)
            self.commit_t = np.full(N, np.nan)
            self.final_t = np.full(N, np.nan)
            self.lg_blocks = 0
            self.lg_cpu = 0.0
            self.lg_rounds = 0
            self.lg_last_round = -1
            self.g_start = self.g_end = self.g_next_end = -INF
        self.rep_at = np.full(N, np.nan)
        self.debt_at = np.full(N, np.nan)
        self.km_driven = 0.0
        self.move_j = 0.0
        self.n_moves = 0
        self.n_distress_moves = 0
        self.slow_steps = 0
        self.completions: list[int] = []
        self.i = 0
        self.now = 0.0
        self.next_slow = self.slow_dt if self.slow_dt > 0 else INF
        self.done = False
        self._prepare_next()
        return self.request

    # ------------------------------------------------------------------ public API
    def act(self, a: int) -> None:
        """Apply a dispatch decision for the current task: a server index, or self.REJECT."""
        if self.request is None or self.request[0] != "fast":
            raise RuntimeError("no dispatch decision pending")
        i, t = self.i, self.now
        a = int(a)
        if a == self.REJECT:
            if not self.allow_reject:
                raise ValueError("REJECT is disabled for this system (dispatch.allow_reject)")
            self._reject(i, R_POLICY)
        elif 0 <= a < self.S and self.cur_mask[a]:
            self._assign(i, a, t)
        else:
            self._reject(i, R_INVALID)
        self.i += 1
        self._prepare_next()

    def act_slow(self, action: dict | None = None) -> None:
        """Apply a slow decision. Keys: 'shares_compute' (C, 3), 'shares_radio' (C, 3), 'moves' [(server, site)]."""
        if self.request is None or self.request[0] != "slow":
            raise RuntimeError("no slow decision pending")
        t = self.now
        if action is None:
            self._builtin_slow(t)
        else:
            if action.get("shares_compute") is not None:
                self.shares_c = np.asarray(action["shares_compute"], np.float64).reshape(self.C, N_SLICES)
            if action.get("shares_radio") is not None:
                self.shares_r = np.asarray(action["shares_radio"], np.float64).reshape(self.C, N_SLICES)
            for s, dest in action.get("moves", []) or []:
                if self.is_mob[s] and self.site[s] >= 0 and self.site[s] != dest:
                    self._start_drive(int(s), int(dest), t)
        self._prepare_next()

    def pop_completions(self) -> list[int]:
        """Task indices finished or rejected since the last call (for delayed rewards)."""
        out, self.completions = self.completions, []
        return out

    def site_backlog_s(self) -> np.ndarray:
        """Expected queued work per core at each site, in seconds (parked edge servers only)."""
        t = self.now
        nrun = self.cores - self.free
        work = np.maximum(self.run_ef_sum - nrun * t, 0.0) + self.q_cyc.sum(1) / self.f_core
        parked = self.site >= 0
        sw = np.bincount(self.site[parked], weights=work[parked], minlength=self.C)
        sc = np.bincount(self.site[parked], weights=self.cores[parked], minlength=self.C)
        return sw / np.maximum(sc, 1)

    def utilization(self) -> np.ndarray:
        """Exponentially averaged busy-core fraction per server at the current time."""
        dt = np.maximum(self.now - self.util_t, 0.0)
        a = np.exp(-dt / self.kg["tau_s"])
        busy = (self.cores - self.free) / self.cores
        return self.util * a + (1.0 - a) * busy

    def debt(self) -> np.ndarray:
        """Ledger debt per site: max(0, mean on-time throughput - site on-time throughput) (EdgePlus Eq. 14)."""
        thr = self.thr * np.exp(-np.maximum(self.now - self.thr_t, 0.0) / self.debt_tau)
        return np.maximum(thr.mean() - thr, 0.0)

    # ------------------------------------------------------------------ main loop
    def _prepare_next(self) -> None:
        blocks = self.ledger_on and self.ledger_mode == "blocks"
        while True:
            if self.i >= self.N:
                if blocks:  # drain in 0.25 s steps so ledger blocks see completions in time order
                    while self.evheap:
                        T = self.evheap[0][0] + 0.25
                        self._advance_all(T)
                        self._ledger_until(T)
                self._advance_all(INF)
                if blocks:
                    self._ledger_flush()
                self.done = True
                self.request = None
                return
            t = float(self.t_arr[self.i])
            if t >= self.next_slow:
                ts = self.next_slow
                self.next_slow += self.slow_dt
                self._advance_all(ts)
                if blocks:
                    self._ledger_until(ts)
                    self._ledger_retarget()
                self.now = ts
                self._park(ts)
                self._refill(ts)
                self.slow_steps += 1
                if self.mobility == "external":
                    self.request = ("slow", ts)
                    return
                self._builtin_slow(ts)
                continue
            self._advance_all(t)
            if blocks:
                self._ledger_until(t)
            self.now = t
            self._park(t)
            self._refill(t)
            i = self.i
            self._predict(i, t)
            n_opt = int(self.cur_mask.sum())
            self.n_options[i] = n_opt
            if n_opt == 0:
                self._handle_empty(i, t)
                self.i += 1
                continue
            if self.auto_single and n_opt == 1 and not self.allow_reject:
                self.auto[i] = True
                self._assign(i, int(np.flatnonzero(self.cur_mask)[0]), t)
                self.i += 1
                continue
            self.request = ("fast", i)
            return

    def _advance_all(self, T: float) -> None:
        h = self.evheap
        while h and h[0][0] <= T:
            _, ver, s = heapq.heappop(h)
            if ver != self.ver[s]:
                continue
            self._advance_server(s, T)
            self._schedule(s)

    def _schedule(self, s: int) -> None:
        run, tq = self.running[s], self.transit[s]
        nf = run[0][0] if run else INF
        nt = tq[0][0] if (tq and self.free[s] > 0) else INF
        ne = nf if nf < nt else nt
        self.ver[s] += 1
        if ne < INF:
            heapq.heappush(self.evheap, (ne, int(self.ver[s]), s))

    def _advance_server(self, s: int, T: float) -> None:
        run, tq, rq = self.running[s], self.transit[s], self.readyq[s]
        clock = self.clock[s]
        prio = self.prio_mode
        while True:
            if self.free[s] > 0:
                while tq and tq[0][0] <= clock:
                    r, seq, j = heapq.heappop(tq)
                    heapq.heappush(rq, (int(self.slice[j]), r, seq, j) if prio else (r, seq, j))
                while self.free[s] > 0 and rq:
                    self._start(s, heapq.heappop(rq)[-1], clock)
            nf = run[0][0] if run else INF
            nt = tq[0][0] if (tq and self.free[s] > 0) else INF
            ne = nf if nf < nt else nt
            if ne > T or ne == INF:
                break
            if ne != ne:  # NaN would loop forever
                raise FloatingPointError(f"non-finite event time on server {s}")
            clock = ne
            if nf <= nt:
                fin, j = heapq.heappop(run)
                self._complete(s, j, fin)
        self.clock[s] = T if T < INF else clock

    def _util_touch(self, s: int, now: float) -> None:
        dt = now - self.util_t[s]
        if dt > 0:
            a = math.exp(-dt / self.kg["tau_s"])
            self.util[s] = self.util[s] * a + (1.0 - a) * (self.cores[s] - self.free[s]) / self.cores[s]
            self.util_t[s] = now

    def _start(self, s: int, j: int, now: float) -> None:
        self._util_touch(s, now)
        f = self.f_core[s]
        cyc = self.cycles[j]
        lvl = 1.0
        # DVFS with slack reclamation: run at the lowest core speed that still leaves a safety margin before the
        # deadline, but only when nobody else is waiting and another core stays free, so no other task is delayed.
        if self.dvfs_on and not self.is_cloud[s] and self.free[s] > 1 and not self.readyq[s]:
            slack = self.deadline[j] - now - self.downlink[s]
            base = cyc / f
            for x in self.dvfs_levels:
                if base / x * (1.0 + self.dvfs_margin) <= slack:
                    lvl = x
                    break
        fe = f * lvl
        heapq.heappush(self.running[s], (now + cyc / fe * self.svc_noise[j], j))
        self.free[s] -= 1
        k = self.slice[j]
        self.q_cnt[s, k] -= 1
        self.q_cyc[s, k] -= cyc
        ef = now + cyc / fe
        self.run_ef[j] = ef
        self.run_ef_sum[s] += ef
        self.start_t[j] = now
        self.dvfs_lvl[j] = lvl
        e = self.j_cycle[s] * cyc * self.svc_noise[j] * lvl * lvl  # dynamic energy per cycle ~ f^2
        self.e_comp[j] = e
        if self.is_mob[s]:
            self.battery[s] -= e

    def _complete(self, s: int, j: int, fin: float) -> None:
        self._util_touch(s, fin)
        self.free[s] += 1
        self.run_ef_sum[s] -= self.run_ef[j]
        self.mem_used[s] -= self.mem_mb[j]
        done = fin + self.downlink[s] + (self.metro_delay if self.remote[j] else 0.0)
        lat = done - self.t_arr[j]
        ok = done <= self.deadline[j]
        self.finish_t[j] = done
        self.latency[j] = lat
        self.on_time[j] = ok
        svc = fin - self.start_t[j]
        self.svc_m1[s] += 0.02 * (svc - self.svc_m1[s])
        self.svc_m2[s] += 0.02 * (svc * svc - self.svc_m2[s])
        if self.ledger_on and self.ledger_mode == "blocks":
            self._ledger_write(j, fin)  # reputation and debt change when the block is committed
        else:
            if self.ledger_on:
                self.rep[s] += self.rep_beta * ((1.0 if ok else -self.rep_kappa) - self.rep[s])
            if ok:
                self._thr_add(int(self.home[j]), fin)
        if self.calib is not None:
            self.calib.update(int(self.slice[j] * 3 + self.kind[s]), self.pred_L[j], lat, self.bound[j])
        self.completions.append(j)

    def _thr_add(self, c: int, t: float) -> None:
        dt = t - self.thr_t[c]
        if dt >= 0:
            self.thr[c] = self.thr[c] * math.exp(-dt / self.debt_tau) + 1.0 / self.debt_tau
            self.thr_t[c] = t
        else:
            self.thr[c] += math.exp(dt / self.debt_tau) / self.debt_tau

    # ------------------------------------------------------------------ ledger timing (EdgePlus Sec. III-M)
    def _poa_time(self, b: int) -> tuple[float, float]:
        """Wall time and total CPU time to validate one partial block of b tuples by edge PoA (Eqs. 19-22)."""
        h = self.c_hash0 + self.c_hash1 * b
        leader = h + b * self.c_tag + self.c_sign                     # hash, check tags, sign the block
        validator = self.c_verify + b * self.c_tag + h + self.c_sign  # in parallel at every authority
        collect = self.quorum * self.c_verify                         # leader checks the votes it needs
        wall = leader + self.lan + validator + self.lan + collect
        return wall, leader + self.n_val * validator + collect

    def _ledger_write(self, j: int, t: float) -> None:
        # Completions reach this point server by server, so write times are not globally sorted; a block is
        # never cut before the latest write it contains (lg_last).
        c = int(self.home[j])
        if self.lg_pend[c] and t >= self.lg_first[c] + self.blk_timeout:
            self._ledger_cut(c, self.lg_first[c] + self.blk_timeout)
        if not self.lg_pend[c]:
            self.lg_first[c] = self.lg_last[c] = t
        self.lg_pend[c].append(j)
        self.lg_first[c] = min(self.lg_first[c], t)
        self.lg_last[c] = max(self.lg_last[c], t)
        self.lg_write_t[j] = t
        self.lg_count[c] += 1
        if len(self.lg_pend[c]) >= self.lg_target[c]:
            self._ledger_cut(c, t)

    def _ledger_cut(self, c: int, t: float) -> None:
        t = max(t, self.lg_last[c])
        ids = self.lg_pend[c]
        self.lg_pend[c] = []
        self.lg_first[c] = INF
        self.lg_last[c] = -INF
        wall, cpu = self._poa_time(len(ids))
        done = max(t, self.lg_busy[c]) + wall
        self.lg_busy[c] = done
        self.lg_cpu += cpu
        self.lg_blocks += 1
        heapq.heappush(self.lg_heap, (done, self.lg_seq, c, ids))
        self.lg_seq += 1

    def _ledger_until(self, T: float) -> None:
        """Cut blocks whose timeout passed, then apply every PoA commit up to time T, in time order."""
        if self.blk_timeout < INF and T < INF:
            for c in np.flatnonzero(self.lg_first + self.blk_timeout <= T):
                self._ledger_cut(int(c), self.lg_first[c] + self.blk_timeout)
        h = self.lg_heap
        while h and h[0][0] <= T:
            tc, _, c, ids = heapq.heappop(h)
            for j in ids:
                s = self.server_log[j]
                ok = self.on_time[j]
                self.rep[s] += self.rep_beta * ((1.0 if ok else -self.rep_kappa) - self.rep[s])
                if ok:
                    self._thr_add(c, tc)
            self.commit_t[ids] = tc
            self.final_t[ids] = self._finality(tc + self.wan)

    def _finality(self, arr: float) -> float:
        """Time the cloud BFT round that includes a cluster hash arriving at `arr` finishes (Eqs. 23-25)."""
        if not self.g_eager:  # rounds start every global_period_s
            k = math.ceil(arr / self.g_period)
            if k != self.lg_last_round:
                self.lg_last_round = k
                self.lg_rounds += 1
                self.lg_cpu += self.bft_cpu
            return k * self.g_period + self.t_bft
        # eager: a round starts as soon as a hash is waiting and no round is running (pipelined)
        while self.g_next_end > self.g_end and arr >= self.g_end:
            self.g_start, self.g_end = self.g_end, self.g_next_end
        if arr >= self.g_end:
            self.g_start, self.g_end = arr, arr + self.t_bft
            self.lg_rounds += 1
            self.lg_cpu += self.bft_cpu
            return self.g_end
        if self.g_next_end <= self.g_end:
            self.g_next_end = self.g_end + self.t_bft
            self.lg_rounds += 1
            self.lg_cpu += self.bft_cpu
        return self.g_next_end

    def _ledger_retarget(self) -> None:
        """Adaptive block size per cluster from utilization and write rate (EdgePlus Eq. 21, capped at S_max)."""
        rate = self.lg_count / max(self.slow_dt, 1e-9)
        self.lg_count[:] = 0.0
        self.lg_rate = 0.5 * self.lg_rate + 0.5 * rate
        util = self.utilization()
        edge = (self.site >= 0) & ~self.is_cloud
        n = np.bincount(self.site[edge], minlength=self.C)
        rho = np.bincount(self.site[edge], weights=util[edge], minlength=self.C) / np.maximum(n, 1)
        frac = np.minimum(1.0, rho / self.blk_rho_max + self.lg_rate / self.blk_write_max)
        self.lg_target = self.blk_min + frac * (self.blk_max - self.blk_min)

    def _ledger_flush(self) -> None:
        """End of episode: blocks still filling are cut when their last record was written, then committed."""
        for c in range(self.C):
            if self.lg_pend[c]:
                self._ledger_cut(c, self.lg_last[c])
        self._ledger_until(INF)

    # ------------------------------------------------------------------ decisions
    def _predict(self, i: int, t: float) -> None:
        k, home = int(self.slice[i]), int(self.home[i])
        site, kind = self.site, self.kind
        parked = site >= 0
        driving = site == DRIVING
        tier = np.full(self.S, -1, dtype=np.int64)
        tier[self.is_stat & (site == home)] = T_HOME_STAT
        mob_home = self.is_mob & ((site == home) | (driving & (self.dest == home)))
        tier[mob_home] = T_HOME_MOB
        mob_other = self.is_mob & parked & (site != home)
        if self.pull_on_assign:
            if self.level != "none":
                mob_other &= self.adj[home][np.where(parked, site, 0)]
            tier[mob_other] = T_OTHER_MOB
        tier[self.is_cloud] = T_CLOUD
        phys = (tier >= 0) & (self.mem_used + self.mem_mb[i] <= self.mem_cap)
        phys &= ~self.is_mob | (self.battery >= self.bat_floor)

        ltx = self.ltx_pred[i]
        ready = np.full(self.S, t + ltx + self.lan)
        move = np.zeros(self.S)
        dh = mob_home & driving
        if dh.any():
            move[dh] = np.maximum(self.arrive_t[dh] - t, 0.0)
            ready[dh] = np.maximum(self.arrive_t[dh], t) + ltx
        mo = tier == T_OTHER_MOB
        back = np.zeros(self.S)
        if mo.any():
            if self.serve_remote:
                xfer = max(self.metro_busy[home], t + ltx + self.lan) + self.payload_bits[i] / self.metro_rate
                ready[mo] = xfer + self.metro_delay
                back[mo] = self.metro_delay
            else:
                cx, cy = self.centers[home]
                d_m = np.hypot(self.pos[mo, 0] - cx, self.pos[mo, 1] - cy) * self.road * 1000.0
                move[mo] = d_m / self.v[mo]
                ready[mo] = t + move[mo] + ltx
        bh_done = max(self.bh_busy, t + ltx) + self.payload_bits[i] / self.bh_rate
        ready[self.is_cloud] = bh_done + self.wan

        if self.predictor == "kingman":
            rho = np.minimum(self.utilization(), self.kg["rho_max"])
            m1 = self.svc_m1
            cs2 = np.maximum(self.svc_m2 / (m1 * m1) - 1.0, 0.0)
            c = self.cores
            wq = rho ** (np.sqrt(2.0 * (c + 1)) - 1.0) / (c * (1.0 - rho)) * (self.kg["ca2"] + cs2) / 2.0 * m1
            start = ready + wq
        else:
            if self.prio_mode:
                ahead_cyc = self.q_cyc[:, : k + 1].sum(1)
                ahead_n = self.q_cnt[:, : k + 1].sum(1)
            else:
                ahead_cyc = self.q_cyc.sum(1)
                ahead_n = self.q_cnt.sum(1)
            nrun = self.cores - self.free
            work = np.maximum(self.run_ef_sum - nrun * t, 0.0) + ahead_cyc / self.f_core
            wait = np.where(nrun + ahead_n < self.cores, 0.0, work / self.cores)
            if self.prio_mode and k > 0:
                # Higher-priority work arriving during the wait is served first: inflate by 1 / (1 - rho_high).
                decay = np.exp(-np.maximum(t - self.offered_t, 0.0) / self.kg["tau_s"])
                rho_high = self.offered[:, :k].sum(1) * decay / self.cores
                wait = wait / np.maximum(1.0 - rho_high, 0.1)
            start = np.maximum(ready, t + wait)
        proc = self.cycles[i] / self.f_core
        L = start + proc + self.downlink + back - t
        U = self.calib.bound(L, k * 3 + kind) if self.calib is not None else L

        mask = phys.copy()
        cause = 0
        ok_dl = None
        if self.level == "full":
            ok_dl = (t + U <= self.deadline[i]) if (self.deadline_mode == "all"
                                                    or (self.deadline_mode == "urllc" and k == 0)) else None
            if ok_dl is not None:
                mask &= ok_dl
            ok_b = None
            if self.use_budgets:
                comp_ok = self.comp_tok[home, k] >= self.cycles[i]
                radio_ok = self.radio_tok[home, k] >= ltx
                ok_b = (self.is_cloud | comp_ok) & radio_ok
                mask &= ok_b
            ok_q = None
            if self.use_quarantine and k == 0:
                ok_q = self.rep >= self.rep_urllc
                mask &= ok_q
            if not mask.any():
                if not phys.any():
                    cause = C_PHYSICAL
                else:
                    if ok_dl is not None and not (phys & ok_dl).any():
                        cause |= C_DEADLINE
                    if ok_b is not None and not (phys & ok_b).any():
                        cause |= C_BUDGET
                    if ok_q is not None and not (phys & ok_q).any():
                        cause |= C_QUARANTINE
        elif not mask.any():
            cause = C_PHYSICAL
        self.cur_tier, self.cur_phys, self.cur_mask = tier, phys, mask
        self.cur_L, self.cur_U, self.cur_ready, self.cur_move, self.cur_proc = L, U, ready, move, proc
        self.cur_start = start
        self.cur_cause = cause
        # Point-estimate deadline check (no calibration margin): decides whether a task is hopeless.
        self.cur_deadline_ok = phys & (t + L <= self.deadline[i]) if ok_dl is not None else phys

    def _handle_empty(self, i: int, t: float) -> None:
        """No server passes every check. Servers that can still meet the deadline come first.

        best_effort: the lowest predicted latency among deadline-feasible servers, else among all physical ones.
        reject: drop the task only if no physical server is predicted to meet the deadline (a hopeless task);
                if one is (the mask was emptied by a slice budget or quarantine), use it as in best_effort.
        """
        self.mask_empty[i] = True
        self.empty_cause[i] = self.cur_cause
        phys, dl = self.cur_phys, self.cur_deadline_ok
        if not phys.any():
            self._reject(i, R_NO_RESOURCE)
            return
        if dl.any():
            pool = dl
        elif self.infeasible == "reject":
            self._reject(i, R_INFEASIBLE)
            return
        else:
            pool = phys
        s = int(np.flatnonzero(pool)[np.argmin(self.cur_L[pool])])
        self.best_effort[i] = True
        self._assign(i, s, t)

    def _reject(self, i: int, reason: int) -> None:
        self.rejected[i] = True
        self.reject_reason[i] = reason
        self.completions.append(i)

    def _assign(self, i: int, s: int, t: float) -> None:
        tier = int(self.cur_tier[s])
        k, home, cyc = int(self.slice[i]), int(self.home[i]), self.cycles[i]
        ltx_true = self.ltx_pred[i] * self.tx_noise[i]
        if self.kind[s] == CLOUD:
            start_bh = max(self.bh_busy, t + ltx_true)
            self.bh_busy = start_bh + self.payload_bits[i] / self.bh_rate
            r = self.bh_busy + self.wan
            self.e_net[i] = self.payload_bits[i] * self.j_bit
        elif tier == T_OTHER_MOB and self.serve_remote:
            st = max(self.metro_busy[home], t + ltx_true + self.lan)
            self.metro_busy[home] = st + self.payload_bits[i] / self.metro_rate
            r = self.metro_busy[home] + self.metro_delay
            self.e_net[i] = self.payload_bits[i] * self.j_bit
            self.remote[i] = True
        elif tier == T_OTHER_MOB:
            self.e_move[i] = self._start_drive(s, home, t, self.drive_noise[i])
            self.relocated[i] = True
            r = self.arrive_t[s] + ltx_true
        elif self.site[s] == DRIVING:
            r = max(self.arrive_t[s], t) + ltx_true
        else:
            r = t + ltx_true + self.lan
        if not math.isfinite(r):
            raise FloatingPointError(f"non-finite ready time for task {i} on server {s}")
        self.e_tx[i] = self.p_tx * ltx_true
        heapq.heappush(self.transit[s], (r, self.seq, i))
        self.seq += 1
        self.q_cnt[s, k] += 1
        self.q_cyc[s, k] += cyc
        self.mem_used[s] += self.mem_mb[i]
        tau = self.kg["tau_s"]
        self.offered[s] *= math.exp(-max(t - self.offered_t[s], 0.0) / tau)
        self.offered[s, k] += cyc / self.f_core[s] / tau
        self.offered_t[s] = t
        if self.clock[s] < t:
            self.clock[s] = t
        self._schedule(s)
        if self.use_budgets:
            if self.kind[s] != CLOUD:
                self.comp_tok[home, k] -= cyc
            self.radio_tok[home, k] -= self.ltx_pred[i]
        self.server_log[i] = s
        self.tier_log[i] = tier
        self.assign_t[i] = t
        self.ready_t[i] = r
        self.pred_L[i] = self.cur_L[s]
        self.bound[i] = self.cur_U[s]
        self.move_pred[i] = self.cur_move[s]
        self.rep_at[i] = self.rep[s]

    # ------------------------------------------------------------------ resources
    def _recompute_site_cap(self) -> None:
        parked = self.site >= 0
        self.site_cap = np.bincount(self.site[parked], weights=self.cap[parked], minlength=self.C)

    def _refill(self, t: float) -> None:
        dt = t - self.bucket_t
        if dt <= 0 or not self.use_budgets:
            self.bucket_t = max(self.bucket_t, t)
            return
        rate_c = self.shares_c * (self.site_cap[:, None] * self.overbook)
        self.comp_tok = np.minimum(self.comp_tok + dt * rate_c, rate_c * self.window)
        rate_r = self.shares_r * self.airtime
        self.radio_tok = np.minimum(self.radio_tok + dt * rate_r, rate_r * self.window)
        self.bucket_t = t

    def _park(self, t: float) -> None:
        arrived = (self.site == DRIVING) & (self.arrive_t <= t)
        if arrived.any():
            self.site[arrived] = self.dest[arrived]
            self.pos[arrived] = self.centers[self.dest[arrived]]
            self._recompute_site_cap()

    def _start_drive(self, s: int, dest: int, t: float, noise: float = 1.0) -> float:
        d_m = math.hypot(self.pos[s, 0] - self.centers[dest, 0], self.pos[s, 1] - self.centers[dest, 1]) \
            * self.road * 1000.0
        self.site[s] = DRIVING
        self.dest[s] = dest
        self.arrive_t[s] = t + d_m / self.v[s] * noise
        self.battery[s] -= self.j_drive * d_m
        self.km_driven += d_m / 1000.0
        self.move_j += self.j_drive * d_m
        self.n_moves += 1
        self._recompute_site_cap()
        return self.j_drive * d_m

    def _builtin_slow(self, t: float) -> None:
        if self.mobility == "distress":
            self._distress_moves(t)

    def _distress_moves(self, t: float) -> None:
        """EdgePlus-style redeployment: pull an idle mobile from a calm neighbour into a distressed site."""
        mo = self.cfg["mobility"]
        b = self.site_backlog_s()
        nrun = self.cores - self.free
        work = np.maximum(self.run_ef_sum - nrun * t, 0.0) + self.q_cyc.sum(1) / self.f_core
        # Energy-aware variant: a site must stay distressed for several slow steps before a mobile drives in, and
        # the nearest calm neighbour gives one up (EdgePlus: first distressed step, calmest neighbour).
        self.distress_cnt = np.where(b > mo["distress_hi_s"], self.distress_cnt + 1, 0)
        cx = self.centers
        moved, donors_used = 0, set()
        for c in np.argsort(-b):
            if b[c] <= mo["distress_hi_s"] or moved >= mo["max_moves_per_step"]:
                break
            if self.distress_cnt[c] < self.distress_persist:
                continue
            donors = [n for n in self.neighbors[c] if b[n] < mo["distress_lo_s"] and n not in donors_used]
            key = (lambda x: math.hypot(cx[x, 0] - cx[c, 0], cx[x, 1] - cx[c, 1])) if self.donor_near \
                else (lambda x: b[x])
            for n in sorted(donors, key=key):
                cand = np.flatnonzero(self.is_mob & (self.site == n) & (self.battery >= self.bat_floor))
                if cand.size:
                    m = int(cand[np.argmin(work[cand])])
                    self._start_drive(m, int(c), t)
                    self.n_distress_moves += 1
                    donors_used.add(n)
                    moved += 1
                    break

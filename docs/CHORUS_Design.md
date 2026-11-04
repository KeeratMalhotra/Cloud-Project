# CHORUS: design document

**CHORUS** (working name): Coordinated Hierarchical Orchestration with Reliability-aware, Uncertainty-calibrated Scheduling. It is our multi-agent RL framework for task allocation in 6G edge-cloud systems, designed to fix the limitations we found in EdgePlus.

- Status: direction agreed on 4 Nov 2026. Milestone M1 (data and topology) is done; next is M2, the simulator core (Section 12).
- References like "deep review 5.4.2" point to [`EdgePlus_Deep_Review.md`](./EdgePlus_Deep_Review.md) (section 5.4, item 2).
- Paper target: 15 to 25 pages.

---

## Contents

1. [Goals and claims](#1-goals-and-claims)
2. [System model shared with EdgePlus](#2-system-model-shared-with-edgeplus)
3. [Agents and decisions](#3-agents-and-decisions)
4. [Policy and critic architecture](#4-policy-and-critic-architecture)
5. [Feasibility and calibrated masks](#5-feasibility-and-calibrated-masks)
6. [Learning](#6-learning)
7. [Execution order at servers](#7-execution-order-at-servers)
8. [Datasets](#8-datasets)
9. [Evaluation protocol](#9-evaluation-protocol)
10. [Hypotheses](#10-hypotheses)
11. [Risks and mitigations](#11-risks-and-mitigations)
12. [Build plan](#12-build-plan)
13. [References](#13-references)

---

## 1. Goals and claims

Under conditions that are identical for both methods, we want to show that:

1. CHORUS completes more tasks on time than EdgePlus, especially URLLC tasks and especially under heavy load.
2. Each gain traces back to a specific design choice, shown by ablation.
3. CHORUS keeps its advantage on real data from a different city that neither method was tuned on (AirDelhi, New Delhi).

| EdgePlus limitation (deep review item) | CHORUS mechanism | Section |
|---|---|---|
| Only chooses a server, admits every task, servers run first-come-first-served (5.4.8) | The dispatcher chooses a server or rejects the task; servers run tasks by slice priority and deadline | 3.1, 7 |
| Fixed slice budgets; mobile servers move only in reaction to distress (5.4.6) | Orchestrators re-split slice budgets every few seconds; mobile-server agents move ahead of forecast demand | 3.2, 3.3 |
| Multi-agent in name only: one decision per step, no communication (5.4.7) | Orchestrators and mobile agents act at the same time and exchange messages with neighbouring sites through graph attention | 3.4, 4 |
| Ten separate actors tied to one fixed server list | One shared policy that scores any set of candidate servers, so it runs on a new map without retraining | 4 |
| Hand-set reward weights plus a PSO loop whose acceptance check cannot pass under load (5.3.7) | Per-slice miss-rate targets enforced by Lagrange multipliers that adjust during training | 6.2 |
| The mask trusts one latency estimate (5.4.10) | Calibrated upper bounds from adaptive conformal inference, per slice and server type | 5 |

The building blocks exist in the literature: MAPPO, graph attention, pointer-style action scoring, Lagrangian PPO, and adaptive conformal inference (Section 13). Our contribution is the combination aimed at EdgePlus's failure modes, a head-to-head comparison under equal budgets, and a city-to-city transfer test.

---

## 2. System model shared with EdgePlus

Both methods run in one simulator. Its components follow deep review Sections 2 and 4.

- **Sites and servers.** C edge sites (clusters). Each site has 1 stationary server and 2 mobile servers. There are 2 cloud servers. Neighbouring sites come from a Delaunay triangulation of the site centres.
- **Tasks.** Each task has a slice `p ∈ {URLLC, eMBB, mMTC}`, an absolute deadline `d`, CPU cycles `k`, payload `b`, memory `m`, a location `x`, and a home site `c(i)`.
- **Latency and energy.** `L = L_tx + L_move + L_queue + L_proc` and energy as in EdgePlus Eqs. 1–2 and 30–32.
- **Ground truth.** Queues are simulated explicitly, with random service times. The Kingman formula (EdgePlus Eq. 31) is used only as a predictor.
- **Feasibility.** The same rules as EdgePlus: routing order, slice budgets, capacity, battery, deadline, and the URLLC reputation quarantine.
- **KPIs.** The same as EdgePlus, plus per-slice misses, rejections, and decision time.

Every system feature is a configuration switch, so each method can run with or without it (needed for the comparison tracks in Section 9).

---

## 3. Agents and decisions

There are three agent types. Agents of the same type share parameters and are told apart by their input features.

### 3.1 Dispatcher (fast timescale, one per site)

- **When.** Each time a task arrives at the site.
- **Observation.**
  - Task: slice, slack `d − t`, cycles, payload, memory, priority.
  - Each candidate server: type, speed, backlog per priority class, predicted completion time, calibrated upper bound, link rate, distance and drive time (mobiles), battery, utilization, reputation.
  - Site: load, slice usage against budget, recent miss rate per slice.
  - The site's latest message vector (Section 4).
- **Action.** One server from the EdgePlus candidate set (stationary in `c`, mobiles in `c`, mobiles in neighbouring sites, cloud), or REJECT. Infeasible servers are masked (Section 5).
- **Hopeless-task rule.** If no candidate could meet the deadline even with an empty queue, the task is rejected without asking the policy.
- **Reward.** Given when the task finishes or is rejected (Section 6.1).

### 3.2 Orchestrator (slow timescale, one per site)

- **When.** Every Δ = 5 s of simulated time, all sites together.
- **Observation.** Per-slice arrivals in the last window, an EWMA forecast for the next window, per-slice backlog and miss rates, the current split, and the site's message vector.
- **Action.** The site's slice split, chosen from a fixed grid on the 3-slice simplex (15 splits in steps of 0.1, each share at least 0.1). The split sets the radio and compute budgets of each slice for the next window.
- **Reward.** Team reward over the next window (Section 6.1).

### 3.3 Mobile-server agents (slow timescale, one per mobile server)

- **When.** Every Δ, all mobiles together, except those still driving.
- **Observation.** Own position, battery, current site, load and forecasts of the current and neighbouring sites, and the positions of nearby mobiles.
- **Action.** Stay, move to one of the current site's neighbours, or return home. Driving takes distance / v (v ≤ 120 km/h) and costs energy. A driving server finishes its queue but accepts no new tasks until it arrives.
- **Reward.** Team reward over the next window, minus its own driving energy.

### 3.4 Why this is genuinely multi-agent

Ten orchestrators and twenty mobile agents decide at the same time, so their decisions can conflict, for example two sites pulling the same mobile, or neighbours pushing load onto each other. Coordination comes from messages between neighbouring sites (Section 4) and from centralized critics during training (Section 6). At run time each agent uses only its own observation and its neighbours' messages.

---

## 4. Policy and critic architecture

- **Encoders.** Small MLPs (hidden size 64) for task, server, and site features.
- **Messages.** Two graph-attention layers over the site graph (Delaunay edges plus self-loops) give each site a message vector `h_c`. They are recomputed at every slow step and reused by the dispatcher until the next one.
- **Dispatcher head.** For each candidate `s`, `logit_s = MLP([q_task, e_s, h_c, h_c(s)])`. The reject option has its own head, `logit_reject = MLP([q_task, h_c])`. A masked softmax over the candidates gives the policy. Because the same scorer is applied to each candidate, the policy works for any number of servers and sites.
- **Orchestrator head.** An MLP over `h_c` and the site features gives logits over the share grid.
- **Mobile head.** Attention from the mobile's embedding over its candidate destinations gives logits.
- **Critics (training only).** A graph network over all sites and servers, with mean and max pooling, and one value head per agent type.
- **Size and speed.** About 50k to 100k parameters. A network of this size took 53 µs per decision in our sandbox benchmark when 16 simulations ran side by side.

---

## 5. Feasibility and calibrated masks

1. Start from EdgePlus's mask: routing order, slice budgets, capacity, battery, and the URLLC reputation quarantine.
2. Deadline check. EdgePlus admits server `s` if `t + L̂(i,s) ≤ d`, with a point estimate `L̂`. CHORUS admits it if `t + U(i,s) ≤ d`, with an upper bound `U = L̂ + q_g` for the task's group `g` (slice × server type).
3. `q_g` is the `(1 − α_g)` empirical quantile of recent residuals (realized minus predicted latency) in group `g`, over a sliding window.
4. `α_g` is adapted online (adaptive conformal inference): `α_g ← α_g + γ (α*_g − err)`, with `err = 1[realized latency > U]`. Targets `α* = 0.01` (URLLC), `0.05` (eMBB), `0.10` (mMTC); step `γ = 0.005`.
5. Guarantee. Adaptive conformal inference makes the long-run miss rate among admitted tasks in group `g` approach `α*_g` even when load drifts. We also check this empirically, because the scheduler's own choices decide which residuals are observed.
6. If every server fails the check, the policy may still reject or pick a best-effort server.

---

## 6. Learning

### 6.1 Rewards

For a task `i` of slice `k`:

```
r_i = w_k (1[on time] − 1[missed]) − λ_k · 1[missed] − β_L · L̃_i − β_E · Ẽ_i − β_C · 1[cloud]
```

- A rejected task counts as missed. `L̃` and `Ẽ` are normalized latency and energy.
- Defaults: `w = (3, 2, 1)`, `β_L = 0.1`, `β_E = 0.05`, `β_C = 0.2`.
- Slow agents receive the task rewards earned in their window, divided by the number of arrivals, minus their own action costs (driving energy and a small penalty for changing the split).
- A reward is attached to the decision step when its task finishes. With priority queues the finish time depends on later arrivals, so each rollout waits for its tasks to finish before computing advantages.

### 6.2 Per-slice reliability constraints

- Each slice has a miss-rate target `ε_k`, for example `ε = (0.05, 0.10, 0.15)` under stress. These targets replace EdgePlus's hand-set reward weights and its PSO outer loop.
- After each PPO update: `λ_k ← max(0, λ_k + η_λ (m̂_k − ε_k))`, where `m̂_k` is the slice's miss rate in the latest rollout.
- If a target cannot be met at a given load, `λ_k` keeps growing until it hits a cap. We report the miss rate actually achieved.

### 6.3 Algorithm

- MAPPO: centralized critics, decentralized actors, clipped objective, GAE, entropy bonus, gradient clipping.
- Starting values: learning rate 3e-4, discount 0.99 (fast) and 0.95 (slow), GAE λ 0.95, clip 0.2, entropy 0.01, 4 epochs, minibatch 512, rollouts of 8,192 decisions.
- Staged training:
  - S1: dispatchers only. Slices stay fixed and mobiles follow EdgePlus's distress rule.
  - S2: orchestrators and mobile agents, with dispatchers frozen.
  - S3: all agents together at a lower learning rate.
- Training mixes episodes from all three load levels, so one policy serves every load. EdgePlus instead tunes once at 50k.

---

## 7. Execution order at servers

- FIFO (EdgePlus), or slice priority: URLLC before eMBB before mMTC, earliest deadline first within a slice, without preemption.
- This is a configuration switch, so EdgePlus can also run with priority queues ("EdgePlus+", Section 9).

---

## 8. Datasets

### 8.1 Dataset A: CityPulse pollution (the data EdgePlus used)

- 449 fixed sites, 5 pollutant index values every 5 minutes, 1 Aug to 1 Oct 2014, 7.9M rows. It is located in Aarhus, Denmark, and generated as a bounded random walk (deep review 5.1).
- Rescaled to 10 km × 10 km, as in EdgePlus.
- Split by day: 1 Aug to 9 Sep train, 10 to 19 Sep validation, 20 Sep to 1 Oct test.

### 8.2 Dataset B: AirDelhi (New Delhi, the unseen test)

- **Source.** Chauhan, Ranu, Sen, Patel, Batra, "AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks Track. Portal: <http://www.cse.iitd.ac.in/pollutiondata>, which links to a public Google Drive folder of 91 daily CSV files (1 Nov 2020 to 30 Jan 2021). License: CC BY 4.0.
- **Our copy.** Google Drive throttles bulk downloads, so both datasets are mirrored as checksum-verified Parquet files on this repository's `data-mirror` branch (same rows and values; both licenses allow redistribution with attribution). `scripts/download_data.py` fetches from there by default.
- **Content.** PM1, PM2.5, and PM10 from low-cost sensors with GPS and 4G, mounted on 13 public DIMTS buses. About 12.5M readings over about 559 km² (latitude 28.49–28.72, longitude 77.10–77.32).
- **What we checked** on three sample days (15 Nov, 15 Dec, 12 Jan):
  - 178k to 214k readings per day from 9 to 12 active buses, one reading per bus every 3 s (median);
  - 2,445 distinct 100 m grid cells visited;
  - PM2.5 median 172 µg/m³, 95th percentile 567, maximum 1,236;
  - readings per hour between 17.5k and 30.7k, with peaks around 07:00 and 19:00;
  - 99th percentile of the 1-minute PM2.5 change: 109 µg/m³.
- **Why it is a good final test.** Real measurements with real spikes, moving IoT sources instead of fixed sites, daily peaks in arrivals, and a different city with about 5.6× the area of Dataset A.
- **Geometry.** Kept at its true scale.
- **Split by day.** 1 Nov to 24 Dec for retraining EdgePlus and fine-tuning CHORUS, 25 to 31 Dec validation, 1 to 30 Jan test.

### 8.3 Turning readings into tasks (both datasets)

This makes EdgePlus Algorithm 3 concrete.

1. **Clean.** Drop duplicates, sort by time per sensor, drop non-physical values, winsorize each pollutant at p99.5 (limits from training days).
2. **Severity.** Robust z-score per pollutant (median and MAD from training days), plus a rise flag from the change over Δt (5 min for A, 1 min for B). Severity = max z + 1.0 × rise flag.
3. **Tier.** Two modes (`slice_mode` in `configs/workload.yaml`):
   - `fixed` (default): inside each episode, the 20% most severe readings become Immediate (URLLC), the 30% least severe become Delayed (mMTC), and the rest Routine (eMBB). This keeps EdgePlus Table IV's fixed shares, so load is the only thing that changes between regimes.
   - `natural`: thresholds at the 30th and 80th percentiles of training-day severity, so the mix follows the day's pollution. Used as a stress variant: on the Delhi test day of the sample episode (16 Jan 2021) it gives 31% URLLC, 66% eMBB, and 3.5% mMTC.
4. **Priority.** High, Moderate, or Low from severity, with hysteresis per sensor.
5. **Footprints.** Per-class lognormal payload, cycles, and memory, scaled up with severity. These are our assumptions (EdgePlus gives none) and are calibrated in M2.
6. **Deadline.** Uniform in the tier's band (URLLC 5–15 s, eMBB 30–120 s, mMTC 120–600 s).
7. **Episodes.** N = 50k, 100k, or 150k readings sampled from a window of real time in one split, with the window compressed to 600 s of simulated arrivals. Dataset A's readings arrive in 5-minute batches, so each reading gets a seeded random offset inside its interval. Footprints and deadlines are drawn per episode from the episode seed, so every method sees identical tasks.

### 8.4 Measured properties (milestone M1)

From `results/summary/<dataset>_dataset.json` (sample episodes: test split, seed 0).

| Property | A: CityPulse | B: AirDelhi |
|---|---|---|
| Sensors | 449 fixed sites | 13 buses |
| Readings (train / val / test) | 5.17M / 1.29M / 1.42M | 5.83M / 1.26M / 5.38M |
| Days (train / val / test) | 40 / 10 / 12 | 54 / 7 / 30 |
| Area used | 10 × 10 km (rescaled) | 21.6 × 25.8 km (true scale) |
| Points clustered into sites | 449 sites | 4,815 cells of 100 m |
| Variation of arrivals per second (CV) | 0.05 to 0.10 | 0.31 at every load |
| Busiest site's task share, relative to the mean | 2.1× | 3.3× |

The CityPulse arrival variation matches a Poisson stream (1/√250 ≈ 0.063 at 250 tasks/s), so its load is steady. AirDelhi's stays at 0.31 at all three loads because the rate itself changes over time (bus routes and rush hours), and one Delhi site receives a third of all tasks. These are the conditions the design targets: bursty arrivals and uneven sites.

Figures: `docs/figures/{citypulse,airdelhi}_{topology,arrivals,severity}.png`.

### 8.5 Topology

- Weighted K-means with C = 10 on sensor locations (seed 39). Dataset A uses its 449 sites; Dataset B uses 100 m grid cells weighted by reading count and severity. Neighbours come from a Delaunay triangulation of the centres.
- Both datasets use the same inventory (1 stationary and 2 mobile servers per site, 2 cloud servers), so the data is the only thing that changes.
- A second Delhi map with C = 15 tests a different topology, which EdgePlus cannot handle without a new set of actors.

### 8.6 Optional Dataset C

Telecom Italia's Milan mobile-traffic data (Harvard Dataverse, 62 daily files of about 0.3 GB each) as a cross-domain stress test, if time allows.

---

## 9. Evaluation protocol

### 9.1 Methods compared

- **EdgePlus (our reimplementation):** configurations A1 to A6, its four learned baselines (ensemble PPO, FD3QN, DDPG, SAC), and `ccppo_lagr`.
- **Heuristics:** random feasible, local-first, least-loaded, greedy earliest finish, slack-aware greedy.
- **EdgePlus+:** EdgePlus with the CHORUS system features that need no learning (priority queues, hopeless-task rejection).
- **CHORUS** and its ablations.

### 9.2 Comparison tracks

- **T1, same choices.** FIFO queues, no rejection, fixed slices, EdgePlus's mobile rule. CHORUS's dispatcher against EdgePlus. This isolates the learning method and architecture.
- **T2, full framework.** CHORUS with every feature against EdgePlus and EdgePlus+.
- **T3, transfer to Delhi.** EdgePlus retrained from scratch on the Delhi training days, against CHORUS with no Delhi training and CHORUS fine-tuned with 10% of EdgePlus's training budget. Repeated on the C = 15 map.
- **T4, stress.** Heavy-tailed service times (lognormal σ = 1, Pareto α = 2.5), load surges, and a failed stationary server.

### 9.3 Fairness rules

- Same simulator, inventory, task streams, seeds, and training budget (3M decisions for fast policies) for every learned method.
- Same hyperparameter search budget for EdgePlus and CHORUS: 16 random-search trials on validation days. EdgePlus also keeps its PSO loop, reported separately.
- Code and configurations released.

### 9.4 Metrics and statistics

- Success, miss rate per slice, URLLC reliability, Lp50/Lp95/Lp99, cloud share, locality index, energy, Jain fairness, rejection rate, empty-mask rate, decision time, training wall-clock time.
- 3 training seeds × 20 evaluation seeds for each configuration and load. Means with bootstrap 95% confidence intervals, paired Wilcoxon signed-rank tests (n = 20) with Holm correction, and effect sizes.

### 9.5 Ablations

Remove one piece at a time: rejection, priority queues, orchestrators, mobile agents, messages (replaced by local features only), Lagrange multipliers (fixed weights instead), calibrated masks (point estimate instead), and staged training (joint training from scratch instead).

---

## 10. Hypotheses

| ID | Claim | Test |
|---|---|---|
| H1 | With the same choices, CHORUS matches or beats EdgePlus on success and Lp95 | T1, all loads |
| H2 | The full framework lowers violations and URLLC misses at 100k and 150k compared with EdgePlus and EdgePlus+ | T2 |
| H3 | Calibrated masks keep the admitted URLLC miss rate near its target under heavy tails, and point-estimate masks do not | T4 |
| H4 | Messages and slow orchestration reduce Lp95 and cloud share, more so on bursty data (Delhi) | T2, T3, ablations |
| H5 | CHORUS with no Delhi training comes close to EdgePlus retrained on Delhi, and fine-tuned CHORUS beats it | T3 |

We report the outcome of every test, including the ones that go against us.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| T1 gains are small: EdgePlus and its baselines sit within about 1 pp of each other at 150k | The main claims rest on T2 to T4; T1 is reported as is |
| Two-timescale training is unstable | Staged training, a lower learning rate for slow agents, and heuristic orchestration as a fallback |
| "You beat a weak EdgePlus" | Equal budgets, EdgePlus+ variants, and released code |
| Calibration is biased by the policy's own choices | Report observed coverage per group; adaptive conformal inference keeps its long-run guarantee regardless |
| Gaps in the Delhi data (buses parked at night) | They appear as low-arrival periods; we report them in the dataset section |

---

## 12. Build plan

| Milestone | Content | Output |
|---|---|---|
| M1 (done) | Project skeleton, data pipelines for A and B, topology | labeled readings, maps, dataset summaries |
| M2 | Simulator core (queues, latency, slices, masks, ledger signals, KPIs), heuristic baselines, load calibration | heuristic results at 50k/100k/150k |
| M3 | EdgePlus reproduction: CCPPO, advisory, ledger, PSO, learned baselines, A1–A6 | OBJ1–OBJ5 tables and figures |
| M4 | CHORUS stage S1: dispatcher, rejection, Lagrange multipliers, calibrated masks | T1 and part of T2 |
| M5 | CHORUS stages S2–S3: orchestrators, mobile agents, messages | full T2 and ablations |
| M6 | Delhi transfer and stress tests | T3 and T4 |
| M7 | Paper (15–25 pages) | draft |

Compute: the sandbox benchmark (4 cores, no GPU) puts one 3M-decision training run at about 30 minutes per core, and all final runs at about 25 to 35 core-hours.

---

## 13. References

- C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 Datasets and Benchmarks Track (MAPPO).
- P. Veličković et al., "Graph Attention Networks", ICLR 2018.
- O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks", NeurIPS 2015.
- A. Ray, J. Achiam, D. Amodei, "Benchmarking Safe Exploration in Deep Reinforcement Learning", 2019 (PPO-Lagrangian).
- A. Stooke, J. Achiam, P. Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020.
- I. Gibbs, E. Candès, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021.
- A. N. Angelopoulos, S. Bates, "Conformal Prediction: A Gentle Introduction", Foundations and Trends in Machine Learning, 2023.
- S. Huang, S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", FLAIRS 2022.
- J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 1707.06347, 2017.
- J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016.
- J. F. C. Kingman, "The single server queue in heavy traffic", Mathematical Proceedings of the Cambridge Philosophical Society, 1961.
- S. K. Chauhan, S. Ranu, R. Sen, Z. B. Patel, N. Batra, "AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks Track.
- CityPulse Smart City Datasets, pollution measurements, <http://iot.ee.surrey.ac.uk:8080/datasets.html>.

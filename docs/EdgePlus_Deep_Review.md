# EdgePlus: deep review, reproduction spec, and research directions

Working reference for the project. It covers three jobs:

1. re-implementing the base paper,
2. finding what to improve,
3. choosing a novelty and writing our own paper.

Related files in this folder:

- [`EdgePlus_Explained.md`](./EdgePlus_Explained.md): beginner-friendly walkthrough of the concepts. Read it first if a term below is unfamiliar.
- Base paper: Taher, Luo, Ghrabat, Qiao, "EdgePlus: A Multiagent Reinforcement Learning Framework for Dynamic Task Allocation in 6G Edge Computing", IEEE Internet of Things Journal 13(8), 15 April 2026, DOI 10.1109/JIOT.2026.3656811.
- Background review: Chaurasia, Shukla, Vishwakarma, Tiwari, "Transformative impact of edge computing with 6G network using cloud computing", Journal of Cloud Computing, 2026, DOI 10.1186/s13677-026-00857-x.

Equation, table, figure, and section numbers refer to the EdgePlus paper unless they are marked "Review".

Numbers marked ≈ were read off a plot. Everything else is printed in the paper or was computed by us (the arithmetic is shown).

---

## Contents

0. [Summary of findings](#0-summary-of-findings)
1. [The paper at a glance](#1-the-paper-at-a-glance)
2. [Technical specification (what we must implement)](#2-technical-specification-what-we-must-implement)
3. [Reported results (our reproduction targets)](#3-reported-results-our-reproduction-targets)
4. [What the paper leaves unspecified](#4-what-the-paper-leaves-unspecified)
5. [Issues found](#5-issues-found)
6. [The background review paper](#6-the-background-review-paper)
7. [Improvements](#7-improvements)
8. [Novelty candidates](#8-novelty-candidates)
9. [Roadmap](#9-roadmap)
10. [Decisions (resolved 4 Nov 2026)](#10-decisions-resolved-4-nov-2026)
11. [Sources](#11-sources)

---

## 0. Summary of findings

1. **What EdgePlus is.** An online scheduler that sends each arriving IoT task to a stationary edge server, a mobile edge server, or the cloud, in a simulated 10 km × 10 km city with 10 clusters. The core is PPO with one actor per cluster and a shared critic. A feasibility mask removes servers that would break the routing order, a slice budget, a resource limit, or the deadline. Three add-ons sit on top: logit nudges (Nash, PID, leader), a particle swarm (PSO) outer loop, and a hash-chained ledger whose reputation and debt values feed back into the state, the reward, and the mask.
2. **What we can reproduce.** The scheduling results (OBJ1 to OBJ5) can be rebuilt in Python with NumPy and PyTorch, after we fill in many parameters the paper never gives (Section 4). The PID study (OBJ6) is cheap to redo. The security "defense %" results (OBJ7) cannot be reproduced in a meaningful way because the attack models are not described. We can measure real cryptographic overheads instead.
3. **The dataset is not what the paper says.** The 449-site, Aug–Oct 2014 pollution data on the CityPulse site is located in Aarhus, Denmark, and it is generated: each value follows a bounded random walk that moves by −5 to +5 every 5 minutes. We downloaded the file and confirmed both facts (Section 5.1). The paper calls it "City of Brașov air-pollution traces" and calls the workload bursty.
4. **The gains over baselines are small and statistically untested.** At 150k tasks EdgePlus beats FD3QN by 0.33 percentage points (pp) of success. Each configuration ran 5 episodes. With 5 paired samples the exact two-sided Wilcoxon signed-rank test cannot go below p = 0.0625, so the paper's own test protocol cannot confirm any pairwise difference at α = 0.05.
5. **Several reported numbers contradict each other.** EdgePlus's cloud share at 150k is about 2% in Fig. 11 and 6.1% in Fig. 13. Every overhead in Fig. 19 differs from the same attack's overhead in Tables VIII and IX. Violations go down from 100k to 150k for every method. Section 5.2 lists them with the arithmetic.
6. **Some components do not do what their names suggest.** The "Nash" game has no interaction between players, so it reduces to a closed-form load discount. The PID term as written pushes more work toward overloaded clusters (wrong sign). The PID certified in OBJ6 saturates at ±100, while the one inside the scheduler saturates at ±1. With no adversaries in OBJ1 to OBJ5, ledger reputation is a delayed congestion signal. Section 5.4.
7. **Masks are the paper's main finding, and they are where we should push.** According to Sec. IV-D the baselines inherit the same environment-level masks, so most of the performance comes from masking. The masks trust a deterministic queue formula. Making them calibrated under uncertainty is our recommended novelty (Section 8).
8. **The "6G" part is nominal.** Deadlines are 5 to 600 seconds and median latency at stress is 53 s. The review paper's own 6G target for URLLC is below 1 ms (Review Eq. 8 and Table 3). A millisecond-scale scenario would be a useful extra test for us.

---

## 1. The paper at a glance

**Problem.** Tasks arrive one at a time. Each has a slice (URLLC, eMBB, or mMTC), a deadline, a size, and a home cluster. The scheduler picks one server per task to maximize on-time completion while limiting latency, energy, and cloud use.

**Method (four layers).**

| Layer | What it does | Paper sections |
|---|---|---|
| Masked CCPPO | PPO with one actor per cluster and a centralized critic. Infeasible servers get probability 0. | III-C, III-E, IV-C5b |
| Advisory | Adds Nash, PID, and leader terms to the actor's logits. Never unmasks. | III-F |
| PSO outer loop | Tunes outer parameters. Accepts a change only if success improves and SLA thresholds hold. | III-G |
| Blockchain-lite | Logs completions in hash-chained blocks (edge PoA, cloud BFT). Reputation and debt feed state, reward, and masks. | III-H, III-M |

Mobility (CBS–D* Lite path planning) and a PID robustness study sit beside these layers.

**Headline claims (150k tasks).** 82.77% on-time, 17.23% violations, throughput 206.93 tasks/s, Lp50 53.23 s, Lp95 147.09 s. Ablations: removing masks costs 5.5 to 10.0 pp of violations and 26 to 47 s of Lp95. PID settles in about 2.7 to 2.9 s under ±20% gain drift. Baseline security attacks are defended at 98% to 99%, advanced ones at 70% to 83%. Edge PoA validation takes about 0.39 ms per partial block.

**Evaluation objectives.** OBJ1 stress QoS (150k), OBJ2 light-load sanity (50k), OBJ3 scaling (50k→100k→150k), OBJ4 locality vs cloud and violations, OBJ5 ablations, OBJ6 PID robustness, OBJ7 security.

---

## 2. Technical specification (what we must implement)

### 2.1 Entities and notation (Sec. III-A)

- Clusters `c ∈ {1, …, 10}`. The cloud is cluster 0.
- Servers `S = S_stat ∪ S_mob ∪ S_cloud`.
- Task `i`: arrival `t_i`, absolute deadline `d_i`, home cluster `c(i)`, slice `p_i ∈ {URLLC, eMBB, mMTC}` with weight `w_p > 0`, CPU cycles `k_i`, memory `m_i` (MB), payload `b_i` (bits), location.
- Server `s`: speed `f_s` (cycles/s), memory `M_s`, battery `E_max` (mobiles only), load `ρ_s ∈ [0, 1]`, backlog `Q_s` (seconds).
- Each cluster reserves radio and compute time per slice: `Γ_radio(c, k)` and `Γ_compute(c, k)`.

### 2.2 Latency and energy (Eqs. 1, 2, 30–32)

```
L(i,s)  = L_tx + L_move + L_queue + L_proc                                  (1)
L_tx    = η_OH · b_i / R(i,s)
R(i,s)  = ζ_slice · (W / Φ) · log2(1 + SNR(i,s)) · η_MAC · η_backhaul       (30)
L_move  = ||x_i − x_s(t_i)|| / v_s   for mobiles (v_s ≤ 120 km/h), else 0
L_queue ≈ ρ_s / (1 − ρ_s) · (c_a² + c_s²) / 2 · E[S_s]                      (31, Kingman G/G/1)
L_proc  = k_i / f_s                                                         (32)
E(i,s)  = P_tx · L_tx + χ_s · ||x_i − x_s(t_i)|| + κ_s · k_i                (2)
```

`W` is per-slice bandwidth, `Φ` frequency reuse, `η_MAC` contention, `η_backhaul` cross-cluster/core load. `c_a` comes from the task generator and `c_s` is estimated online. The mobile path length comes from the CBS–D* Lite planner.

### 2.3 Feasibility, routing order, and slices (Eqs. 3–4, Sec. IV-C5b)

Candidates, in routing order: stationary in `c(i)` → mobile in `c(i)` → mobile in a neighbor cluster `N(c(i))` → cloud. A candidate is masked out (`M(i,s) = 0`) if any of these hold:

- the slice's radio or compute budget in the server's cluster would be exceeded,
- the server's capacity or memory would be exceeded, or a mobile's battery is too low,
- `t_i + L(i,s) > d_i` (the text says this is hard "for URLLC when safety gating is enabled"),
- URLLC only: the server's reputation is below `ρ_th^URLLC` (quarantine).

The mask is the only mechanism that removes actions. Advisory terms only reorder the remaining ones.

The paper also states the problem as an integer program (Eq. 3): one server per task, a big-M deadline constraint, `Σ_i x_is k_i/f_s ≤ H` per server, `Σ_i x_is E_is ≤ E_max` per mobile, and per-slice, per-cluster sums of radio time `b_i/R` and compute time `k_i/f_s` bounded by `Γ` (Eq. 4). As written, these budgets are totals over the whole horizon `H`.

### 2.4 Objective and reward (Eqs. 5–7, 14–16)

```
J(x)   = Σ_i [ α w_p 1(late) + β w_p L + γ E + δ 1(cloud) ]                 (5, minimize)
χ_cloud = share of completed tasks run on the cloud
Pr(L ≤ d | URLLC) ≥ 1 − ε_URLLC                                             (6, ε not given)
r_t    = η1 w_p 1[on-time] − η2 w_p 1[late] − η3 L̃ − η4 Ẽ − η5 1[cloud]     (7)
r_t   ← r_t − η6 φ(ρ_rep(a_t)) − η7 d_c(a_t)                                 (15, ledger shaping)
φ(ρ)   = max(0, ρ_th − ρ),  ρ_th = 0.6   (or softplus with β = 10)          (16)
```

### 2.5 Masked MDP and CCPPO (Secs. III-E, IV-B4, IV-E)

- One actor per cluster. The actor of the task's home cluster makes the decision. Actors see local observations. A centralized critic `V_ψ(s)` sees global context (centralized training, decentralized execution).
- Observation: queue depths and headroom (`Q_s`, `ρ_s`, `M_s`) of reachable servers, link-rate estimates and RTTs, deadline slack `d_i − t`, slice availability, neighbor congestion summaries, ledger signals (`ρ_rep`, `d_c`). Standardized with training statistics only. No attack labels.
- Loss (Eq. 8): clipped PPO objective, KL penalty `λ_KL`, entropy bonus `λ_H`. Critic: mean squared error. Advantages from GAE. Gradient-norm clipping. No early stopping.

### 2.6 Advisory layer (Eqs. 9–11)

```
z̃_s = z_s + λ_N · log s_c(s) + λ_P · u_c(s) + λ_L · 1[s = leader of c(s)]   (9)
π(a = s) = M(i,s) · exp(z̃_s) / Σ_j M(i,j) · exp(z̃_j)                       (10)
```

- **Nash sharing.** Each cluster picks `s_c ∈ [s_min, s_max]` to maximize `u_c = −α_N s_c load_c − β_N energy_c − γ_N (s_c − 1)²`, repeating best responses until the change is below ε. Because `u_c` depends only on the cluster's own `s_c`, the best response has a closed form: `s_c = clip(1 − α_N · load_c / (2 γ_N), s_min, s_max)`. One pass is enough.
- **PID.** `e_c = ρ_c − ρ_set`, `u_c = K_p e_c + K_i ∫e_c dt + K_d de_c/dt`, clipped to `[−1, 1]` (Eq. 11).
- **Leader.** In each cluster, the server with the highest `ω1 (1 − load ratio) + ω2 (energy ratio)` gets a bonus.

### 2.7 PSO outer loop (Eqs. 12–13, Alg. 1)

```
w_t      = w_max − (w_max − w_min) · t / I                                  (13a)
v_{t+1}  = k · w_t · v_t + k c1 r1 ⊙ (p − θ_t) + k c2 r2 ⊙ (g − θ_t)        (13b)
θ_{t+1}  = θ_t + v_{t+1}                                                    (13c)
k        = 2 / |2 − φ − sqrt(φ² − 4φ)|,  φ = c1 + c2 > 4                    (13d)
```

With `c1 = c2 = 2.05`, `k ≈ 0.7298`. A candidate is accepted only if success improves and `χ_cloud ≤ 0.07`, fail `≤ 0.05`, and stationary share `≥ 0.80`. PSO runs as pretuning or every `K_PSO` episodes. It is off at evaluation.

### 2.8 Ledger (Secs. III-H, III-M)

- Log tuple per completed task (Eq. 18): `(i, c, s, p, d, t_start, t_end, on-time flag, energy, nonce, AES-GCM tag)`.
- Each cluster batches tuples into partial blocks chained by `h = BLAKE2b(SHA3-256(block ∥ h_prev))` (Eqs. 19–20). Block size adapts to utilization and write rate (Eq. 21).
- Edge validation: Proof of Authority among the cluster's authorities (Eq. 22). Up to 30 validators, split into sub-quorums of at most 10. Forks are tagged and reconciled.
- Cloud finalization: BFT over the ordered cluster hashes, signed with Ed25519 (Eqs. 23–25).
- Sessions: X25519 key agreement and AES-GCM with 96-bit IVs (Eq. 26). Key manager (DKMS) with rotation floors and hysteresis (Eq. 27).
- Scheduler hooks (Eqs. 14, 28–29): reputation `ρ_rep(s) = EMA(1[on-time] − κ · 1[violation])`, debt `d_c = max(0, mean throughput − Θ_c)`. Both enter the state, the reward (Eq. 15), and the URLLC mask.

### 2.9 Mobility (Alg. 4, Eqs. 33–38)

Grid over a risk/obstacle map. Edge cost `w_d · length + w_wait · 1[wait] + w_r · risk(v)`. Conflict-Based Search resolves vertex and swap conflicts between vehicles. D* Lite replans single vehicles. Mobiles are redeployed from nearby clusters toward "distress clusters". `L_move` uses the current path length.

### 2.10 PID robustness (Secs. III-I, V-F)

Plant `G(s) = K e^{−Ls} / (τ s + 1)` discretized at `T_s`. Closed loop `1 + C(z) G(z)` with a discrete PID. Jury's stability test over every corner of the ±20% gain box (Eqs. 17, 40). Then a 90 s unit-step simulation with `u_max = 100` and anti-windup clamp 100. Feasible means settling within ±2% in at most 10 s and overshoot at most 15%. The paper does not report `K`, `τ`, `L`, `T_s`, or the Ziegler–Nichols starting gains.

### 2.11 Workload construction (Sec. IV-B, Alg. 3, Tables III–IV)

- Per reading: robust z-scores per pollutant, rate-of-change flags, tier lookup (Immediate → URLLC, Routine → eMBB, Delayed → mMTC), a footprint model for `(b, k, m)`, a deadline drawn uniformly from the tier's band, and a priority with hysteresis.
- Deadline bands: URLLC 5–15 s, eMBB 30–120 s, mMTC 120–600 s.
- Footprints (Table III): `b` from the pollutant vector plus metadata (gzip estimate), `k` from a light-CNN FLOP count, `m` from a static model plus buffer by class.
- Cleaning: drop duplicates, repair clock skew per site, winsorize at p99.5, interpolate gaps up to 10 min, normalize, and check that more than 99.9% of tasks have at least one feasible server (a static check, not under load).
- Regimes (Table IV): 50,000 / 100,000 / 150,000 tasks with fixed shares 20% URLLC, 50% eMBB, 30% mMTC.
- Seeds {38, 39, 40} for label sampling, clustering, and the simulator.

### 2.12 Topology (Sec. IV-C1, Fig. 2)

100 km² area, 10 clusters from weighted K-means (density plus exceedance weights), neighbors from a Delaunay triangulation of the centroids.

The paper never states how many servers exist. We counted markers in the embedded Fig. 2 image by color: 8 clearly visible stationary servers plus at least one partly hidden behind the legend, 17 visible mobiles plus a few under the legend, and 2 cloud servers. Working assumption: **1 stationary and 2 mobile servers per cluster, plus 2 cloud servers (32 servers).**

### 2.13 Schedulers and ablations (Sec. IV-D, Table VI)

| Name | Description |
|---|---|
| `ccppo_edgeplus` | Full method: masks, advisory, PSO, ledger. |
| `ensemble_ppo` | Bootstrap ensemble of PPO actors. Inherits masks and ledger when enabled. |
| `fd3qn` | Dueling double DQN with n-step targets. Inherits masks and ledger. |
| `ddpg` | Continuous logits, discretized. Inherits masks and ledger. |
| `sac` | Entropy-regularized actor-critic with tuned temperature. Inherits masks and ledger. |
| `ccppo_lagr` | Ablation only: Lagrangian CCPPO, no masks, advisory and PSO off. |

| Config | Masks | Advisory | PSO | Ledger |
|---|---|---|---|---|
| A1 full | ✓ | ✓ | ✓ | ✓ |
| A2 no PSO | ✓ | ✓ | – | ✓ |
| A3 no advisory | ✓ | – | ✓ | ✓ |
| A4 no ledger | ✓ | ✓ | ✓ | – |
| A5 no masks | – | ✓ | ✓ | ✓ |
| A6 plain CCPPO | – | – | – | – |

### 2.14 Protocol and KPIs (Sec. IV, Table V)

- Five episodes per regime with "identical topology, slice budgets, and RNG seeds". Policies frozen at evaluation. PSO off at evaluation. Hyperparameters tuned once at 50k, then frozen.
- Statistics: means with 95% CIs over episodes, Wilcoxon signed-rank with Benjamini–Hochberg correction (α = 0.05), rank-biserial effect size. Higher success wins; Lp95 must be non-inferior as a tie-breaker.
- KPIs: success σ, violation %, Lp50 and Lp95 (end-to-end latency), throughput (completed tasks/s), total energy, locality index LI (% executed in the home cluster), χ_cloud, URLLC tail reliability, Jain fairness over cluster throughput, SEI (defense % / median ms), CAF (ledger work / base work).

---

## 3. Reported results (our reproduction targets)

### OBJ1, stress, 150k tasks (Figs. 4–5)

| Scheduler | Success % | Viol. % | Lp50 (s) | Lp95 (s) | Throughput (tasks/s) | Energy (J) |
|---|---|---|---|---|---|---|
| ccppo_edgeplus | 82.77 | 17.23 | 53.23 | 147.09 | 206.93 | 2699.72 |
| ddpg | 79.40 | 20.60 | 82.34 | 154.81 | 198.50 | 2640.62 |
| ensemble_ppo | 81.05 | 18.95 | 69.28 | 148.58 | 202.63 | 2736.34 |
| fd3qn | 82.44 | 17.56 | 57.61 | 141.58 | 206.09 | 2772.08 |
| sac | 82.07 | 17.93 | 60.58 | 141.93 | 205.18 | 2757.76 |

### OBJ2, light, 50k tasks (Figs. 6–8)

| Scheduler | Success % | Lp50 (s) | Lp95 (s) | Throughput | Energy (J) | LI % (≈, Fig. 8) |
|---|---|---|---|---|---|---|
| ccppo_edgeplus | 100.00 | 3.98 | 10.24 | 160.61 | 1162.00 | ≈ 99.64 |
| ensemble_ppo | 100.00 | 4.04 | 10.88 | 160.32 | 1178.74 | ≈ 99.53 |
| fd3qn | 100.00 | 4.05 | 10.91 | 159.83 | 1182.38 | ≈ 99.52 |
| ddpg | 100.00 | 4.00 | 10.85 | 160.41 | 1181.76 | ≈ 99.52 |
| sac | 100.00 | 4.01 | 10.85 | 160.36 | 1181.53 | ≈ 99.54 |

LI vs energy across the five methods: Pearson r = −0.99, p = 0.0013. The whole LI range is about 0.13 pp.

### OBJ3, medium, 100k tasks (Figs. 9–10, read off the plots)

| Scheduler | Viol. % | Lp95 (s) |
|---|---|---|
| ccppo_edgeplus | ≈ 17.5 | ≈ 152 |
| fd3qn | ≈ 20.7 | ≈ 157 |
| sac | ≈ 20.9 | ≈ 155 |
| ensemble_ppo | ≈ 21.3 | ≈ 162 |
| ddpg | ≈ 22.0 | ≈ 176 |

At 50k all methods have 0% violations and Lp95 ≈ 10–11 s.

### OBJ4, locality (Figs. 11–12)

- LI at 150k: ccppo_edgeplus 87.66, ddpg 87.40, ensemble_ppo 88.43, sac 89.69, fd3qn 90.39.
- χ_cloud at 150k (≈, Fig. 11): fd3qn 1.6, sac 1.75, ccppo_edgeplus 1.95, ensemble_ppo 2.05, ddpg 2.3. At 100k: 2.5 to 3.0. At 50k: about 0.
- Pooled over all 15 method × load points: LI vs χ_cloud Pearson r = −0.995, Spearman ρ = −0.94, Theil–Sen slope −0.175 pp cloud per +1 pp LI. LI vs violations Pearson r = −0.77, slope −0.76.
- At 150k only: LI vs χ_cloud r = −0.997 (slope −0.231). LI vs violations r = −0.52, p = 0.37.

### OBJ5, ablations at 150k (Figs. 13–14)

| Config | Success % | Viol. % | Lp50 (s) | Lp95 (s) | Throughput | Cloud % | ΔLI (pp) |
|---|---|---|---|---|---|---|---|
| A1 full | 82.8 | 17.2 | 53.2 | 147.1 | 206.9 | 6.1 | 0 |
| A2 no PSO | 82.0 | 18.0 | 54.8 | 158.9 | 205.3 | 6.6 | −1.6 |
| A3 no advisory | 80.0 | 20.0 | 55.9 | 164.7 | 201.1 | 7.6 | −3.1 |
| A4 no ledger | 82.8 | 17.2 | 53.8 | 150.0 | 206.9 | 6.2 | −0.4 |
| A5 no masks | 77.3 | 22.7 | 57.5 | 173.6 | 195.5 | 10.1 | −7.0 |
| A6 plain CCPPO | 72.8 | 27.2 | 61.2 | 194.2 | 186.2 | 12.1 | −11.7 |

### OBJ6, PID (Table VII, Figs. 15–17)

Best gains `(K_p, K_i, K_d) = (1.90, 1.20, 0.06)`: settling 2.74 s, overshoot 1.93%, control energy 92.88. "Robust pocket" (the paper's term): `K_p ∈ [1.85, 1.90]`, `K_i ∈ [1.15, 1.20]`, `K_d ∈ [0.06, 0.12]` with settling 2.76–2.88 s and overshoot 1.7%–2.0%. A wider sweep passed 2443 of 4560 triplets (53.6%).

### OBJ7, security (Tables VIII–IX, Figs. 18–19)

| Threat | Defense % | Detection % | Overhead, tables (ms) | Overhead, Fig. 19 (ms) | FPR %, Fig. 18 | ASR %, Fig. 18 |
|---|---|---|---|---|---|---|
| Eavesdropping (MITM) | 99.2 | 99.0 | 0.18 | 0.26 | 0.43 | 0.10 |
| Credential spoofing | 98.9 | 99.3 | 1.12 | 0.47 | 0.65 | 0.35 |
| Data tampering | 99.1 | 99.4 | 0.42 | 0.78 | 0.74 | 0.55 |
| Data poisoning | 80.7 | 82.1 | 12.4 | 1.26 | 2.04 | 4.10 |
| Adversarial evasion | 76.4 | 78.9 | 7.9 | 0.94 | 3.64 | 7.92 |
| FL backdoor | 73.2 | 75.6 | 10.6 | 1.57 | 2.60 | 4.91 |
| Ledger tampering (PoA) | 99.8 | 99.8 | 0.39 | 0.63 | 0.22 | 0.10 |
| Ledger tampering (BFT) | 100.0 | 100.0 | 6.9 | 2.30 | 0.14 | 0.05 |
| Authority/Sybil vs PoA | 97.1 | 98.6 | 0.74 | 0.63 | 0.35 | 0.38 |

Table IX also gives SEI (PoA 255.6, BFT 14.5, Sybil 131.2, poisoning 6.5, evasion 9.7, backdoor 6.9), CAF (PoA 1.62, BFT 2.94), and security tax (PoA 0.63 ms, BFT 10.3 ms).

---

## 4. What the paper leaves unspecified

Everything below has to be chosen by us. The defaults are starting points that we will calibrate, then record in config files.

| Missing item | Where it matters | Proposed default |
|---|---|---|
| Number and type of servers per cluster | everything | 1 stationary + 2 mobile per cluster, 2 cloud (from Fig. 2) |
| CPU speeds `f_s`, memory `M_s` | latency, masks | calibrate so 50k gives about 100% success and 150k gives clear overload, as in the paper |
| Radio: `W`, `Φ`, SNR model, `η_MAC`, `η_backhaul`, `ζ_slice`, `η_OH` | `L_tx` | 6G-style bandwidth per slice, SNR from a distance-based path-loss model, a WAN RTT and a backhaul cap for the cloud |
| Task footprints `b`, `k`, `m` per class | latency | class-dependent distributions, scaled by exceedance |
| How 5-minute readings become arrival times in an episode | load | replay readings in time order inside a fixed arrival window, same window for all regimes (to confirm) |
| Episode horizon `H` | throughput, budgets | 600 s, implied by Fig. 5 (Section 5.2, item 2) |
| Slice budgets `Γ` | masks | per-window shares of capacity, plus the paper's per-horizon form behind a flag |
| Behavior when every server is masked | violations at stress | explicit fallback (smallest predicted latency), logged as "infeasible" |
| Deadline masking for eMBB and mMTC | violations | hard for all classes by default, URLLC-only as a variant |
| Ground-truth queueing | realism, masks | explicit simulated FIFO queues, with Kingman (Eq. 31) as the predictor used by the mask |
| Mobile battery, `χ_s`, speed profile | mobiles | 33.3 m/s (120 km/h), battery sized for about 1 hour of work |
| Reward weights `η1`–`η7`, slice weights `w_p` | learning | `η1 = η2 = 1`, `η3 = 0.1`, `η4 = 0.05`, `η5 = 0.2`, `η6 = 0.5`, `η7 = 0.1`, `w = (3, 2, 1)` |
| PPO hyperparameters, network sizes | learning | lr 3e-4, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.01, two hidden layers of 128 |
| Training budget | fairness of comparison | same number of environment steps for every method |
| Advisory gains `λ_N`, `λ_P`, `λ_L`, Nash `α_N`, `β_N`, `γ_N`, `s` range, `ρ_set`, `ω1`, `ω2` | advisory | small gains (0.1–0.5), `ρ_set = 0.7`, then PSO |
| PSO `P`, `I`, `w_max`, `w_min`, `c1`, `c2`, `K_PSO`, and what it optimizes | PSO | `P = 8`, `I = 10`, `w` 0.9 → 0.4, `c1 = c2 = 2.05`, optimize advisory gains, learning rate, entropy |
| Reputation EMA rate, `κ`, `ρ_th^URLLC`, debt window | ledger | EMA 0.05, `κ = 1`, `ρ_th = 0.6`, `ρ_th^URLLC = 0.7`, throughput over a sliding 30 s window |
| PID plant `K`, `τ`, `L`, `T_s`, Ziegler–Nichols gains | OBJ6 | identify the plant from step tests in our simulator |
| Attack models for OBJ7 | security | out of scope; measure real crypto costs instead |
| Baseline hyperparameters (ensemble size, n-step, SAC temperature) | baselines | standard published defaults |

---

## 5. Issues found

### 5.1 Data provenance

What the paper says: the workload comes from "City of Brașov air-pollution traces (449 sites; August–October 2014)", with a footnote "Modified Brașov Dataset (ZIP archive, GitHub snapshot)" and no URL.

What we found:

- On the [CityPulse dataset page](http://iot.ee.surrey.ac.uk:8080/datasets.html), the section titled "Pollution Measurements for the City of Brasov in Romania" repeats the Aarhus pollution description word for word and links only to the Aarhus files (`citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz`).
- The dataset [readme](http://iot.ee.surrey.ac.uk:8080/datasets/pollution/readme.txt) says the stream is a mock-up. CityPulse simulated one pollution sensor at each traffic-sensor location. Each pollutant starts between 25 and 100 and moves by a random integer in [−5, 5] every 5 minutes, with a push back toward the middle near 20 and 210.
- We downloaded the CSV archive (81 MB) and checked it:
  - 449 files, 7,887,974 rows, from 2014-08-01 00:05 to 2014-10-01 00:00, every 5 minutes.
  - Coordinates: latitude 55.988–56.256, longitude 10.063–10.304. That is Aarhus, Denmark. Brașov is near 45.65°N, 25.6°E.
  - Bounding box about 15 km × 30 km, so fitting it into 100 km² needs rescaling that distorts geometry.
  - In a 20-sensor sample, every particulate-matter step from −5 to +5 occurs about 31,700 times. Steps of size 6 to 10 occur only about 125 to 215 times each (the boundary rule).

Consequences:

1. "Spikes" and rate-of-change flags in Algorithm 3 act on uniform noise.
2. The values are index levels, not concentrations, so WHO concentration thresholds do not apply directly.
3. The data is not bursty, contrary to Sec. VI-D.
4. Class shares are fixed at 20/50/30 anyway, so the data mostly supplies sensor locations and an arrival schedule.

What it means for us: we can use the same file for a faithful reproduction, and our paper should state plainly that it is synthetic. A real dataset (for example OpenAQ or the UCI Beijing multi-site air-quality data) would make a good extra test.

### 5.2 Internal inconsistencies

1. **Cloud share at 150k.** Fig. 11 places every method between about 1.6% and 2.4% cloud at stress, with EdgePlus at about 2%. Fig. 13 gives 6.1% for A1. A1 is the same run as OBJ1: its success, Lp50, Lp95, and throughput match Fig. 4 exactly.
2. **Throughput duplicates success at 150k.** For all five schedulers, success × 150,000 / throughput = 600.0 s, to within 0.03 s. So "highest throughput" and "lowest violations" are one result counted twice. At 50k the same ratio gives 311 to 313 s, so the definition or the horizon changes between regimes.
3. **Ablation throughput vs success.** Using the 600 s relation from item 2, A5's throughput (195.5) implies 78.2% success, but 77.3% is reported. A6 (186.2) implies 74.5% against 72.8% reported, and A3 implies 80.4% against 80.0%.
4. **Scaling runs backwards from 100k to 150k.** Violations and Lp95 fall for every method, for example DDPG ≈ 22.0% → 20.6% and FD3QN ≈ 20.7% → 17.6%. With fixed capacity and fixed class shares, more offered load should not lower the miss rate. The paper credits "tighter feasibility masks" but reports no mask or admission measurement.
5. **Locality story vs the data.** At 150k, EdgePlus has the second-lowest LI (87.66%, against FD3QN 90.39% and SAC 89.69%) and the lowest violations (Fig. 12). The text credits a "locality-aware advisory bias".
6. **Two different PIDs.** The scheduler clips the PID output to [−1, 1] (Eq. 11). OBJ6 certifies a controller with `u_max = 100` and an anti-windup clamp of 100.
7. **Security overheads.** Every overhead in Fig. 19 differs from the tables for the same attack (PoA 0.63 vs 0.39 ms, BFT 2.30 vs 6.9 ms, poisoning 1.26 vs 12.4 ms, and so on; full list in Section 3). The plot title says "mean" while the caption says "median". SEI in Table IX matches the table overheads (99.8 / 0.39 = 255.9), so Fig. 19 is the odd one out.
8. **Defense vs attack success.** Table IX gives evasion defense 76.4%, but Fig. 18 gives evasion ASR 7.92%. FL backdoor is 73.2% vs 4.91%, and poisoning is 80.7% vs 4.10%. Neither metric is defined, and the pairs do not add up to 100%.
9. **Rule of three.** The paper says zero false accepts gives an upper bound "< 1/N" with N evaluation episodes. The rule of three gives 3/N at 95%. With N = 5 episodes that bound is 60%.
10. **OBJ4 p-value.** Pearson r = −0.997 with n = 5 gives p ≈ 0.0002, not 0.0017. The reported p fits r ≈ −0.99. Minor.
11. **Text vs figure.** Sec. V-C says Lp95 ≈ 145 s for EdgePlus at 150k. Fig. 4 shows 147.09 s.
12. **What PSO optimizes.** Sec. III-G says PSO tunes outer hyperparameters Θ. Alg. 1 (lines 4 and 13) runs PSO "on (θ, ψ)", which are the network weights.
13. **Notation and typos.** ρ means load, reputation, the PPO ratio, and Spearman's ρ. γ means the discount factor, an objective weight, and a Nash weight. Θ means throughput and the PSO parameter vector. `EMA(1[on-time] − κ·1[violation])` lies in [−κ, 1], while φ(ρ) assumes ρ ∈ [0, 1]. Eq. 21 can exceed `S_max` because the two ratios can sum to 2. Eq. 17 is missing an "=". The abstract says "lowest 15 percentile latency (Lp50 …)", the conclusion says "fd3gn", Sec. VI-B has "SKeep" and "PStabilize", and the heading of Sec. III-I is duplicated.

### 5.3 Methodology gaps

1. **Statistical power.** Five episodes per configuration. With n = 5 paired samples, the smallest two-sided exact Wilcoxon p-value is 2/32 = 0.0625 (we checked with SciPy). No pairwise claim can pass α = 0.05, even before Benjamini–Hochberg correction. Figs. 4–14 show no confidence intervals and no p-values. The headline margin over FD3QN is 0.33 pp.
2. **Same seeds for every episode.** If the five episodes share workload and simulator seeds, the spread across episodes reflects only policy sampling, not workload variation.
3. **Tuning where nothing can be learned.** Hyperparameters are tuned once at 50k, where every method scores 100%. EdgePlus also gets a PSO search that the baselines do not get, so tuning budgets are unequal.
4. **No simple baselines.** There is no random-feasible, local-first, least-loaded, earliest-finish-time, or deadline-ordered heuristic. All learned baselines share the masks, so a greedy "feasible server with the smallest predicted latency" might come close to them. Without it we cannot see what the learning adds.
5. **Mask effect and learner effect are mixed.** A6 (plain CCPPO, 72.8%) is far below every baseline (79.4% to 82.4%), and Sec. IV-D says the baselines inherit the masks. A3 (masks, PSO, and ledger but no advisory, 80.0%) is below masked FD3QN and SAC. The abstract's "removing masks increases violations by 5.5–10.0 pp" includes A6, which removes everything, not only the masks.
6. **KPIs that are defined but never reported.** URLLC tail reliability (Eq. 6, Table V, Alg. 2), Jain fairness, and per-slice violations. Also missing: how often the mask is empty, the fallback rule, decision time per task, training curves, and training cost.
7. **The PSO gate cannot pass under load.** PSO accepts a change only if fail ≤ 5%. At 100k and 150k every method has at least 17% violations, so the gate never passes there. At 50k it always passes, but there is nothing left to improve. The +0.8 pp credited to PSO at 150k is not explained.
8. **LI vs cloud is partly built in.** A task run on the cloud is never in its home cluster, so LI ≤ 100% − χ_cloud by definition. Pooling all loads also mixes in the load effect (at 50k: LI ≈ 99.5%, cloud ≈ 0, violations 0).
9. **OBJ2's "no penalty from masking or security" is untested.** No unmasked or unsecured run is shown at 50k; the 50k ablations are "omitted for brevity".
10. **OBJ6 evidence.** Figs. 15–17 are 100% pass over the whole plotted area, so they show no boundary. The chosen gains (1.90, 1.20) sit on the right edge of the plotted `K_p` range, which suggests the best point lies outside the grid. Jury's test checks stability only; settling time and overshoot come from the step simulation.
11. **OBJ7 evidence.** A passive eavesdropper cannot read AES-GCM traffic without the key, so a 99.2% "defense" implies a key leak model that is never described. Hash-chain tampering is detected deterministically, so 99.8% detection also needs an explanation. SEI = defense % / milliseconds grows without bound as overhead approaches zero.
12. **Citations.** PPO and GAE are credited to [11] and [39] instead of Schulman et al. Kingman's formula, Clerc–Kennedy constriction, Jury's test, MAPPO and CTDE, and work on invalid action masking (Huang and Ontañón) are not cited. The "masks matter most" result agrees with known results on invalid action masking in policy-gradient methods.
13. **No artifacts.** No code is released and the dataset footnotes have no URL.

### 5.4 Design issues

1. **The Nash layer is not a game.** Each cluster's payoff depends only on its own choice, so the best response is the closed form in Section 2.6. The energy term does not depend on `s_c` and has no effect. The cloud (cluster 0) has no defined `s_c` or `u_c`.
2. **PID sign.** With `e = ρ_c − ρ_set` and `z̃ += λ_P u`, an overloaded cluster gets `e > 0`, `u > 0`, and larger logits, so it receives more work. That is positive feedback. The sign has to flip (or `λ_P < 0`).
3. **PID plant.** The path from a logit bias to cluster utilization is nonlinear, coupled across clusters, and shaped by the learned policy. One first-order-plus-dead-time model with unknown parameters does not describe it, so OBJ6 says little about the scheduler.
4. **Ledger with no adversary.** In a benign simulation, a server misses deadlines because it is overloaded. EMA reputation then acts as a delayed congestion signal and the URLLC quarantine acts as a congestion filter. Tamper-evidence cannot affect OBJ1–OBJ5, which matches A4 ≈ A1.
5. **Debt penalty direction.** `d_c > 0` means cluster c has below-average throughput. The reward penalizes sending work there (`−η7 d_c(a)`), which pushes it further below average. The intended meaning is unclear.
6. **Slice budgets as horizon totals.** With Eq. 4 as written, once a cluster's URLLC budget is spent, no URLLC task can run there for the rest of the episode. A per-window (rate) budget is the usual model.
7. **"Multi-agent" is sequential.** One task per step, decided by its home-cluster actor. Actors never act at the same time and interact only through shared queues. In practice this is one policy split into 10 heads, and centralized training adds little.
8. **Tasks with no feasible server.** The paper does not say what happens when every server is masked (drop, cloud, or least-bad server). At stress this case likely drives most violations.
9. **Deadline masking for eMBB and mMTC.** "Hard for URLLC when safety gating is enabled" suggests the other classes can be sent to servers that will miss their deadlines.
10. **Queue model.** If the realized delay is the Kingman formula itself ("with ρ_s from live load"), the mask predicts every outcome exactly and there is no prediction error to manage. Real queues differ, and the paper's own limitations section says so.

### 5.5 How "6G" the setup is

- EdgePlus deadlines: URLLC 5–15 s, eMBB 30–120 s, mMTC 120–600 s. Measured Lp50: about 4 s at 50k and 53 s at 150k.
- The review paper: URLLC needs `Pr(L ≤ 1 ms) ≥ 1 − 10⁻⁵` (Review Eq. 8), 6G latency below 1 ms (Review Table 3), and a 6G latency budget of roughly 0.5 ms transmission, 0.05 ms propagation, 0.5 ms edge processing, 20 ms cloud processing, and 5 ms backhaul (Review Fig. 22).
- A mobile server at 120 km/h needs about 30 s per km, so a mobile that must drive can never serve a URLLC task at 6G timescales.

---

## 6. The background review paper

**What it is.** A broad narrative survey of 6G with edge and cloud computing. Its "Emptical results" section re-plots figures from another paper (DTDRL-ZSON, its Ref. [77]) and adds a 5G vs 6G latency comparison. It has no experiments of its own.

**Useful for us:**

- 5G vs 6G parameter ranges (Review Table 3): 100–300 GHz carriers, about 10 GHz bandwidth, 500–1000 Gbps peak rate, SNR about 30 dB, about 10⁻¹⁰ J/bit, latency below 1 ms, mobility up to 278 m/s.
- A latency decomposition that adds propagation, backhaul, and retransmission terms to EdgePlus's four (Review Eq. 1).
- A THz path-loss model with molecular absorption (Review Eqs. 9–12).
- An offloading formulation with a 1 ms URLLC constraint (Review Eqs. 13–15).
- Research directions that support our motivation: energy- and carbon-aware networking, multi-agent RL, AI-driven orchestration, trustworthy edge AI, scalable federated learning.

**Do not copy:**

- Review Eq. 5 (M/M/1) is garbled. The standard result is `W = 1/(μ − λ)` with `ρ = λ/μ`.
- Review Eq. 6 uses `E[S]²`. The Pollaczek–Khinchine formula needs the second moment: `W_q = λ E[S²] / (2(1 − ρ))`.
- Review Eq. 16 is not FedAvg. FedAvg sets `θ ← Σ_i w_i θ_i` with weights summing to 1.
- Review Fig. 22 is captioned "Futuristic 6G applications" but shows latency components, and its smoothed curves dip below 0 ms.
- "Achieved data rate will considered as 1TB" mixes up units.

For specific numbers, cite the primary sources the review relies on (its Refs. [109]–[111]) rather than the review itself.

---

## 7. Improvements

Each item names the issue it fixes. The "Fixes" column points to items in Section 5 (for example, 5.3.4 is item 4 of Section 5.3). The first four are the ones we recommend for the "improvements" part of our paper.

| ID | Improvement | Fixes | Expected effect | Cost |
|---|---|---|---|---|
| I-1 | Evaluation that can support claims: 20+ independent seeds (workload and simulator), paired tests, bootstrap CIs, per-slice violations, URLLC reliability, Jain fairness, empty-mask rate, decision time, training curves, equal training budgets | 5.3.1–5.3.3, 5.3.6 | claims we can defend | low |
| I-2 | Heuristic baselines, each with and without masks: random-feasible, local-first, least-loaded, greedy earliest-finish-time, deadline-ordered greedy | 5.3.4, 5.3.5 | separates the mask effect from the learner effect | low |
| I-3 | Correct the advisory layer: flip the PID sign, certify the controller that is actually deployed (±1 clamp) on a plant identified from simulator step tests, and either rename "Nash" as a load discount or replace it with a coupled congestion game between neighboring clusters | 5.2.6, 5.4.1–5.4.3 | an advisory layer that does what it claims | medium |
| I-4 | Explicit handling of hopeless tasks: a defined fallback, plus an admission step that rejects a task when no server can finish it in time. Rejecting a task that would miss anyway costs nothing in success and frees capacity for others | 5.4.8 | fewer cascading misses at 100k–150k | low |
| I-5 | Per-window slice budgets and a load-aware PSO gate (relative thresholds instead of fail ≤ 5%) | 5.4.6, 5.3.7 | slicing and PSO that work under load | low |
| I-6 | Explicit queues as ground truth, with noisy and heavy-tailed service times (lognormal, Pareto) | 5.4.10, paper's Sec. VI-D | tests the paper's own stated limitation | low |
| I-7 | Make the ledger matter: inject faulty or dishonest servers (slowdowns, silent drops, false completion reports) and compare EMA reputation with Beta reputation | 5.4.4 | a measurable job for the ledger | medium |
| I-8 | A millisecond-scale 6G scenario based on the review's latency budget | 5.5 | 6G claims we can back | medium |
| I-9 | Optional system change: slice-priority or deadline-ordered queues at servers, applied to every scheduler | FIFO assumption | likely large URLLC gains | medium |

---

## 8. Novelty candidates

### N1 (recommended): conformal feasibility masks

**The gap.** EdgePlus's advantage rests on its masks, and the masks trust a single latency estimate from a queue formula. When service times are noisy or heavy-tailed, or load shifts, a mask like that lets through servers that will miss the deadline and blocks some that would have made it. The paper lists heavier tails and "formal CMDP safety with online certificates" as open problems (Sec. VI-D, VI-E).

**The idea.** Replace the point estimate with an upper bound `U(i,s)` that is calibrated online by conformal prediction on the scheduler's own completed tasks, separately per slice and server type. Admit server s only if `t_i + U(i,s) ≤ d_i`. Each slice gets a target miss rate `α_k`, for example 1% URLLC, 5% eMBB, 10% mMTC. Adaptive conformal inference (Gibbs and Candès, 2021) keeps the long-run miss rate on target while load drifts. When no server passes, I-4 decides between rejection and best effort.

**What we would claim and test.**

- The miss rate among admitted tasks tracks `α_k` per slice: a distribution-free guarantee under exchangeability, and a long-run average guarantee under adaptive conformal inference.
- Higher success and URLLC reliability than deterministic masks under heavy tails and load shifts, with no loss in the nominal case.
- It drops into any masked scheduler (our CCPPO, the baselines, the heuristics), which shows where the gain comes from.

**Related work (quick check, November 2026).**

- Cohen, Park, Simeone, Popovski, and Shamai ([arXiv 2302.07675](https://arxiv.org/abs/2302.07675)) use online conformal prediction to allocate URLLC radio resources with reliability guarantees.
- FleetGuard ([arXiv 2610.18003](https://arxiv.org/abs/2610.18003), October 2026) is a conformal offloader for local, edge, and cloud execution that resists compromised UAV latency reporters, evaluated with 8 reporters and 4 edge servers.
- Conformal "shields" that filter unsafe actions exist for safe RL in robotics and control (for example [arXiv 2506.17275](https://arxiv.org/abs/2506.17275)).

We did not find conformal feasibility masks inside masked multi-agent RL for multi-slice, city-scale edge scheduling. A full literature search is still needed before we claim novelty.

**Cost.** Low to medium: a few hundred lines on top of the simulator and little extra compute.

### N2 (extension of N1): ledger-certified calibration

Calibrate N1 only on completion records committed to the tamper-evident ledger, weighted by reputation, so a dishonest server cannot shrink its own bound by misreporting latencies. This gives the blockchain a measurable job and pairs with I-7. The threat model overlaps with FleetGuard; the difference is ledger provenance instead of identity caps.

### N3: forecast-driven repositioning of mobile servers

A second policy moves mobiles ahead of predicted demand so driving time leaves the task's critical path. Medium cost. The benefit depends on how much mobiles matter once our simulator is calibrated.

### N4: graph-based shared actor

One permutation-invariant policy over the Delaunay cluster graph, shared by all clusters, which can transfer to other city layouts and cluster counts. Medium cost, and a crowded research area.

**Recommendation.** N1 as the main contribution, I-1 to I-4 as the improvements chapter, and N2 if time allows.

**Decision (4 Nov 2026, revised the same day).** The team first chose a separate multi-agent framework (CHORUS), then dropped it in favour of a narrower extension: EdgePlus-PEC, which keeps EdgePlus's scheduler and adds slice-priority queues (I-9), energy-aware execution and mobility, and faster ledger consensus. See [`HANDOFF.md`](./HANDOFF.md).

---

## 9. Roadmap

The current plan and progress live in [`HANDOFF.md`](./HANDOFF.md).

Compute: we benchmarked the sandbox (AMD EPYC, 4 physical cores with 8 threads, 30 GB RAM, no GPU). A simulator loop handles about 8,900 decisions per second per core; with a small policy network it handles about 1,250 per second one task at a time, or about 6,000 per second when 16 simulations run side by side. A PPO update on 8,192 decisions takes 3.7 s. One 3M-decision training run takes about 30 minutes on one core, and all final runs need about 25 to 35 core-hours. A GPU is not required.

---

## 10. Decisions (resolved 4 Nov 2026)

1. **Timeline and length.** No fixed deadline. The paper should be 15 to 25 pages.
2. **Contribution.** EdgePlus-PEC (EdgePlus plus priority queues, energy-aware execution and mobility, and faster ledger consensus), compared head-to-head with EdgePlus on the same data, then tested on a second dataset.
3. **Reproduction scope.** Mobile routing simplified to grid A* or straight-line driving. No BFT, DKMS, or attack injection. OBJ7 reports measured cryptographic timings instead of "defense %".
4. **Ground-truth queues.** Explicitly simulated queues; the Kingman formula is used only as the mask's predictor.
5. **Compute.** The sandbox is enough (Section 9).
6. **Second dataset.** AirDelhi, New Delhi (real bus-mounted PM sensors; see the README).

---

## 11. Sources

- CityPulse Smart City Datasets, pollution section and readme: <http://iot.ee.surrey.ac.uk:8080/datasets.html>, <http://iot.ee.surrey.ac.uk:8080/datasets/pollution/readme.txt>
- K. M. Cohen, S. Park, O. Simeone, P. Popovski, S. Shamai, "Guaranteed Dynamic Scheduling of Ultra-Reliable Low-Latency Traffic via Conformal Prediction", [arXiv 2302.07675](https://arxiv.org/abs/2302.07675)
- FleetGuard: Identity-Capped Conformal Risk Allocation for Reliable UAV Edge–Cloud Offloading, [arXiv 2610.18003](https://arxiv.org/abs/2610.18003)
- Conformal Safety Shielding for Imperfect-Perception Agents, [arXiv 2506.17275](https://arxiv.org/abs/2506.17275)

Descriptions of the web sources above were paraphrased for compliance with licensing restrictions.

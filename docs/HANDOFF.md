# Handoff: full project state and how to continue (last updated 4 Nov 2026)

This file lets anyone, including a new AI session with no memory of earlier chats, continue the project. It records where everything is, what has been done and measured, what is unfinished, and the exact commands to resume.

**How to trust this file.** Every number below comes from a file in the repository or the `runs-mirror` branch, and the source is named next to it. To recheck run progress and validation results at any time:

```bash
.venv/bin/python scripts/status.py --markdown
```

If this file and the files disagree, the files win.

---

## Contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [Where everything is](#2-where-everything-is)
3. [Setting up a fresh session](#3-setting-up-a-fresh-session)
4. [Status by milestone](#4-status-by-milestone)
5. [Milestone M3 in detail](#5-milestone-m3-in-detail)
6. [Next steps](#6-next-steps)
7. [Operating rules](#7-operating-rules-read-before-running-anything)
8. [Running in another chat, or several in parallel](#8-running-in-another-chat-or-several-in-parallel)
9. [Prompt for a new session](#9-prompt-for-a-new-session)

---

## 1. The project in one paragraph

A research paper of 15 to 25 pages with three parts:

1. **Reproduce the base paper**, EdgePlus (Taher et al., IEEE Internet of Things Journal 13(8), 2026): multi-agent RL (CCPPO) that assigns air-quality IoT tasks to edge, mobile, and cloud servers in a simulated 6G city.
2. **Extend it as EdgePlus-PEC**: slice-priority queues (P), energy-aware execution and mobility (E), and faster ledger consensus (C), with EdgePlus's scheduler unchanged, and compare the two under identical conditions. An earlier plan for a separate framework (CHORUS) was dropped on 4 Nov 2026.
3. **Test both on a second, real dataset from New Delhi** (AirDelhi) that neither was tuned on.

The base paper and a background review are in `docs/`. The paper's numbers, gaps, and errors are analysed in `docs/EdgePlus_Deep_Review.md`.

## 2. Where everything is

GitHub repository: <https://github.com/KeeratMalhotra/Cloud-Project>

### Branches

| Branch | Content | State |
|---|---|---|
| `main` | Docs, data pipeline (M1), simulator (M2) | PRs #1–#4 merged |
| `feat/m3-edgeplus` | EdgePlus reproduction, EdgePlus-PEC, training and evaluation scripts, the paper, this file | pushed, draft PR #5 |
| `data-mirror` | Parquet copies of both datasets with SHA-256 checksums (210 MB, separate history) | complete |
| `runs-mirror` | Training runs: checkpoints, logs, PSO records, evaluations (separate history) | updated after each batch |

### Documents

| File | Content |
|---|---|
| `docs/EdgePlus_Explained.md` | Plain-language walkthrough of the base paper |
| `docs/EdgePlus_Deep_Review.md` | Full spec of EdgePlus, every reported number, issues found, decisions |
| `paper/EdgePlus-PEC.md` | The paper (Keerat Malhotra, Bhavya Maheshwari, Bhumika Aswal): reproduction, EdgePlus-PEC, results on CityPulse and AirDelhi |
| `results/summary/pec_comparison.md` | EdgePlus vs EdgePlus-PEC tables with CIs and paired tests (written by `scripts/compare.py`) |
| `docs/Simulator.md` | Simulator model, how we read EdgePlus where it is silent, validation, calibration, rule-based results |
| `docs/HANDOFF.md` | This file |

### Code map

| Path | Content |
|---|---|
| `configs/` | `datasets/*.yaml`, `workload.yaml` (tasks), `topology.yaml` (servers), `sim.yaml` (simulator), `systems.yaml` (system presets), `methods.yaml` (learned methods for M3) |
| `src/edgesim/data/` | loaders (`citypulse.py`, `airdelhi.py`), `mirror.py`, `labeling.py`, `episodes.py` |
| `src/edgesim/sim/` | `env.py` (event-driven simulator), `world.py`, `radio.py`, `calibration.py` (conformal), `metrics.py`, `config.py` |
| `src/edgesim/agents/` | `heuristics.py` (5 rules), `features.py` (observations), `rl.py` (rewards, delayed-reward buffer, parallel envs), `ccppo.py` (EdgePlus learner and trainer), `advisory.py` (Nash/PID/leader), `pso.py` |
| `src/edgesim/` | `runner.py`, `experiments.py` |
| `scripts/` | `download_data.py`, `prepare_data.py`, `validate_simulator.py`, `calibrate_capacity.py`, `run_heuristics.py`, `train.py`, `launch.py`, `evaluate.py`, `compare.py` (statistics and figures), `bench_crypto.py`, `smoke_systems.py`, `status.py`, `sync_runs.sh`, `build_data_mirror.py` |
| `results/summary/` | committed summaries: datasets, validation, calibration, rule-based results, crypto timings, EdgePlus vs EdgePlus-PEC |
| `runs/` | training runs; ignored by git, saved to `runs-mirror` |

## 3. Setting up a fresh session

Takes about 5–10 minutes on a 4-core sandbox. Python 3.11. No GPU needed.

```bash
git clone --single-branch --branch feat/m3-edgeplus https://github.com/KeeratMalhotra/Cloud-Project.git
cd Cloud-Project
uv venv -p 3.11 .venv                                  # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python tabulate      # for status.py --markdown
.venv/bin/python scripts/download_data.py --dataset citypulse   # from data-mirror, seconds
.venv/bin/python scripts/download_data.py --dataset airdelhi
.venv/bin/python scripts/prepare_data.py --dataset citypulse    # about 15 s
.venv/bin/python scripts/prepare_data.py --dataset airdelhi     # about 25 s
bash scripts/sync_runs.sh pull                                  # restore all training runs
.venv/bin/python scripts/status.py                              # what is done and what is not
```

Versions used here: Python 3.11.15, torch 2.15.0+cpu, numpy 2.4.6, pandas 3.1.0 (3.1.1 also works). In some sandboxes `pip install --user` fails because the home directory is read-only; the `.venv` route above works.

Run `status.py` only after `sync_runs.sh pull` has finished; started at the same time, it prints "no runs found". `prepare_data.py` rewrites `results/summary/*_dataset.json`, changing only `runtime_s` and the last digit of one float; restore them with `git checkout -- results/summary/` rather than committing the change. Sandboxes can have 8 cores instead of 4 (`nproc`). With 5 jobs in one `launch.py` call on 8 cores, each run did about 0.5M decisions in 7 minutes.

**Checks.** `prepare_data.py` is deterministic. The training split must have 5,172,031 rows for CityPulse and 5,826,328 for AirDelhi (`results/summary/*_dataset.json`).

## 4. Status by milestone

| Milestone | Content | State |
|---|---|---|
| M1 | Data pipelines, labeling, sites, episodes for both datasets | done (PR #3) |
| M2 | Shared simulator, validation, capacity calibration, rule-based baselines | done (PR #4) |
| M3 | EdgePlus reproduction: CCPPO, advisory, PSO, ledger signals (now timed), ablations | full EdgePlus (A1) done with 3 seeds under the current protocol; A2–A6 only under the earlier protocol; FD3QN, DDPG, SAC not implemented |
| M4 | EdgePlus-PEC: priority queues, energy-aware execution and mobility, fast consensus | done: 3 seeds, plus 1 seed per single change |
| M5 | Test days on CityPulse and AirDelhi (no retraining), statistics, figures | done at 150k (Section 5, first subsection) |
| M6 | Paper | written: `paper/EdgePlus-PEC.md` |

### Key facts from M1–M2

- **Datasets.**
  - CityPulse (EdgePlus's data): 449 sites, 7.9M rows, generated random-walk data located in Aarhus.
  - AirDelhi: 13 buses, 12.5M rows, real data, CC BY 4.0. Source: Chauhan et al., NeurIPS 2023 Datasets and Benchmarks.
  - Splits are by day; see `configs/datasets/*.yaml`.
- **Tasks.** 50k / 100k / 150k per 600 s episode (light / medium / stress), with slice shares 20 / 50 / 30% (URLLC / eMBB / mMTC). Seed k uses the k-th window of a fixed shuffle of days.
- **Simulator** (`docs/Simulator.md`).
  - Multi-core servers: 1 stationary per site (cores proportional to the site's demand), 2 mobiles per site, 2 cloud.
  - Queues are FIFO or slice-priority.
  - Validated against Erlang C, Pollaczek–Khinchine, and Cobham: every case within 5% (`results/summary/sim_validation.json`).
- **Capacity.** `capacity.scale = 0.38`, calibrated on CityPulse validation days (`results/summary/calibration_citypulse.json`).
- **Rule-based results** (test days, 5 seeds; `results/summary/heuristics_*_test.json`), success at 150k:

| Dataset | Preset | Greedy EFT | Local-first |
|---|---|---|---|
| CityPulse | `edgeplus` | 80.6 ± 1.7% (URLLC 46.3%) | 77.1 ± 1.7% |
| CityPulse | `edgeplus_plus` | 89.9 ± 1.7% (URLLC 100%) | 90.4 ± 0.7% |
| AirDelhi | `edgeplus` | 54.9 ± 2.7% (URLLC 24.9%) | 54.7 ± 2.4% |
| AirDelhi | `edgeplus_plus` | 71.9 ± 3.1% (URLLC 96.2%) | 72.1 ± 3.0% |

At 50k every method gets about 100%. EdgePlus reports 79.4–82.8% at 150k for its methods (deep review, Section 3).

## 5. Milestone M3 in detail

### Main result: EdgePlus vs EdgePlus-PEC (4 Nov 2026)

- **Runs.** Nine CCPPO runs of 1.5M decisions each, all under the current protocol (constant imitation weight 0.3), all trained on CityPulse training days:
  - `edgeplus_s0`–`s2`: full EdgePlus (A1);
  - `edgeplus_pec_s0`–`s2`: EdgePlus-PEC;
  - one seed of each single change: `edgeplus_pq_s0`, `edgeplus_energy_s0`, `edgeplus_cons_s0`.

  All nine are on `runs-mirror`, with test evaluations at 150k tasks, test seeds 0–4, in `eval_citypulse_test.csv` and `eval_airdelhi_test.csv`.
- **Systems** (`configs/systems.yaml`):
  - P: slice-priority queues;
  - E: a neighbouring site's mobile serves in place over a metro link instead of driving over, redeployment is damped, and edge cores use slack-based DVFS;
  - C: partial blocks are cut within 0.5 s and cloud BFT runs every 0.5 s.

  The ledger is now timed (edge PoA commit, periodic cloud BFT, crypto costs from `results/summary/crypto_bench.json`). Reputation and debt change when a block commits.
- **Results** (success at 150k; from `results/summary/pec_comparison.md`):

  | System | CityPulse | AirDelhi (no retraining) |
  |---|---|---|
  | EdgePlus (3 seeds × 5 episodes) | 78.8% (URLLC 44.1%) | 54.9% (URLLC 24.9%) |
  | EdgePlus-PEC (3 × 5) | 99.3% (URLLC 100%) | 76.3% (URLLC 98.6%) |
  | + P / + E / + C alone (1 × 5) | 86.0 / 97.6 / 78.8% | 63.0 / 66.4 / 54.8% |
  | greedy EFT on the EdgePlus / PEC system | 79.9 / 99.2% | 54.9 / 76.6% |

- **Caveats stated in the paper:**
  - Most of the gain comes from serve-in-place (part of E).
  - The trained scheduler matches greedy EFT on both systems, so the gain comes from the system, not from learning.
  - Costs: mMTC success falls 2.3 pp (CityPulse) and 12.5 pp (Delhi); Lp95 on Delhi rises from 152 s to 308 s; computing plus network energy rises 67%, though total energy falls because driving stops; ledger CPU time rises about 10×.
  - C alone does not change success.
  - EdgePlus's PSO gate rejected all 18 rounds.
- **Provenance.** 30 of the 45 AirDelhi episodes (the six main runs) were written at 22:07–22:10 UTC by an evaluation call missing from the session transcript. Two of them were recomputed and matched exactly, so they were kept.

### What is built (branch `feat/m3-edgeplus`)

- **CCPPO** (`src/edgesim/agents/ccppo.py`). One actor per site, implemented as stacked weights. Logits over all 32 servers, plus advisory biases, then a masked softmax. One centralized critic over local and global features. PPO with GAE, entropy bonus, optional KL penalty, and gradient clipping.
- **Delayed rewards** (`rl.py`). A decision's reward is known only when its task finishes. Each environment keeps a ring buffer. Steps are used once their reward is known, with GAE truncated at the first unresolved successor. Old log-probabilities are recomputed and stale samples get a truncated importance weight.
- **Reward** (EdgePlus Eq. 7 and 15): ±w per task (w = 3/2/1 by slice), −0.5 × latency/deadline (`eta_lat`), −0.05 × energy, −0.2 × cloud, −0.5 × max(0, 0.6 − reputation), −0.1 × debt, all scaled by 0.1.
- **Advisory** (`advisory.py`). Closed-form Nash sharing. PID on site utilization with the corrected sign (bias −λ_P·u). Leader bonus.
- **PSO** (`pso.py`). Constricted PSO over (λ_N, λ_P, λ_L, ρ_set) with 4 particles and 2 iterations. Each particle is scored on a 200 s training-day stress episode, in parallel worker processes. Two rounds per run, at 50% and 85% of the budget. Gates: `paper` (the paper's SLA thresholds) or `relative` (our fix I-5).
- **Variants** in `configs/methods.yaml`: A1_full, A2_no_pso, A3_no_advisory, A4_no_ledger, A5_no_masks, A6_plain_ccppo, ccppo_lagr, ensemble_ppo, and A1_relgate (our fixed PSO gate).
- **Scripts.** `train.py` (resumable, time-boxed), `launch.py` (parallel jobs with hard time limits), `evaluate.py` (frozen greedy policy; appends `eval_<dataset>_<split>.csv` to the run), `status.py`, `sync_runs.sh`.

### Shared training protocol (every learned method)

- **Budget:** 1.5M decisions per run.
- **Episodes:** 600 s training-day episodes with loads 50k : 100k : 150k sampled 1 : 1 : 2, 8 parallel environments.
- **PPO settings:** lr 3e-4, γ = 0.9, 1,024 steps per environment per update, entropy 0.003.
- **Warm start** (our addition, applied identically to every method, and it must be reported in the paper):
  - for the first 300k decisions the environments follow greedy EFT and the actor imitates it;
  - after that PPO takes over, with an imitation term of constant weight 0.3 (`ppo.bc_coef: 0.3`, `bc_decay_frac: 1000000.0`). This was adopted on 4 Nov 2026 after the pilot in finding 3. Earlier runs used a weight fading from 1 to 0 over half of the remaining budget.
- **Speed:** about 1,700–2,100 decisions per second per run, 4 runs in parallel. One run takes about 12–14 minutes, plus about 2.4 minutes per PSO round.

### Tuning history

All rows are A2-style (advisory on, no PSO), CityPulse validation days, 2 episodes each, success at 150k; numbers from `scripts/status.py`. Greedy EFT on the same two episodes gets 77.9% and 82.3% (mean 80.1%).

| Round | Change | Decisions | Success at 150k |
|---|---|---|---|
| Smoke | 150 s training episodes | 0.54M | about 60% |
| 1 | 600 s episodes: default / γ 0.9 / γ 0.9 + cheaper reputation + costlier cloud | 1.15M | 62.9 / 61.5 / 61.4% |
| Diagnosis (smoke run, one 150k episode) | The policy picked the fastest feasible server only 6% of the time (greedy: 99%) and sent 55% of its decisions to the cloud | | |
| 2 | Added "fastest feasible" flag and latency-rank features; four variants | 0.85–0.93M | 60.9–66.3% (best: `eta_lat` 0.5, entropy 0.003) |
| 3 | Warm start (`w1_bc`, lr 3e-4) | 1.05M | **79.0%** |
| 3 | Warm start with lr 1e-4 (`w2`–`w4`) | 1.05M | 70.8–73.0% |

### Results under the earlier protocol (imitation fading to 0; seed 0, CityPulse validation days, 2 episodes, success at 150k)

All runs trained to 1.5M decisions. Numbers from `runs/citypulse/<run>/eval_citypulse_val.csv`; the greedy EFT row is from `results/summary/heuristics_citypulse_val.json` (same two episodes).

| Run | PSO rounds | Success at 150k | URLLC success | Cloud share |
|---|---|---|---|---|
| greedy EFT (rule) | | 80.1% (77.9 / 82.3) | 46.5% | 14.1% |
| `A1_full_s0` | 2, both rejected | 67.9% | 25.8% | 16.0% |
| `A1_relgate_s0` | 2, both accepted | 68.7% | 22.2% | 15.3% |
| `A2_no_pso_s0` | none | 69.1% | 26.3% | 15.2% |
| `A3_no_advisory_s0` | none | 72.4% | 32.2% | 14.9% |
| `A4_no_ledger_s0` | 2, both rejected | 70.4% | 29.7% | 16.2% |
| `A5_no_masks_s0` | 2, both rejected | 64.4% | 11.8% | 19.0% |
| `A6_plain_ccppo_s0` | none | 72.5% | 23.4% | 17.1% |
| `ccppo_lagr_s0` | none | 68.9% | 16.3% | 17.4% |
| `ensemble_ppo_s0` | none | 71.0% | 31.5% | 15.1% |

Every learned method is 7.6–15.7 pp below greedy EFT. These runs stay on `runs-mirror` whatever is decided on open issue 3.

### Findings and open issues

1. **The paper's PSO gate never passes.** It requires failure ≤ 5% and stationary share ≥ 80%. On the PSO episodes, failures were about 3.5% but the stationary share was 39–40%, so every candidate was rejected (`runs/citypulse/A1_full_s0/pso.jsonl`). A5 was rejected too: in round 1 no particle beat the baseline, and in round 2 the stationary share was 41% (`runs/citypulse/A5_no_masks_s0/pso.jsonl`).
   - **Our relative gate (`A1_relgate`) accepted both rounds.** On the 200 s PSO episodes, success rose from 96.4% to 96.8% (round 1) and from 89.3% to 90.9% (round 2), with cloud share down 0.5 and 1.1 pp (`runs/citypulse/A1_relgate_s0/pso.jsonl`).
   - On validation this gave 68.7% against 67.9% for `A1_full`. That is within the spread of two episodes from one seed, so the gate change does not yet show a measurable effect.
2. **The add-ons do not help so far.** Removing the advisory layer (A3, 72.4%) or both masks and ledger (A6, 72.5%) gave the best seed-0 results. Removing only the masks (A5) gave the worst, 64.4%. These are 2 episodes from one training seed, so this is not yet reliable.
3. **PPO fine-tuning erodes the warm start.** `w1_bc` had 79.0% at 1.05M decisions; A1–A6 and the baselines have 64–73% at 1.5M. In A2 and A3, agreement with greedy EFT falls from about 1.0 after the warm start to 0.72 and 0.62 at the end (`agree` in `log.jsonl`).
   - **Pilot (4 Nov 2026).** Four A2-style runs (seed 0, 1.5M decisions, everything else as in the protocol) that keep the imitation term longer. Numbers from each run's `eval_citypulse_val.csv` and the last line of `log.jsonl`:

     | Run | Imitation weight after the 300k warm start | Weight at the end | Success at 150k | URLLC |
     |---|---|---|---|---|
     | `A2_no_pso_s0` (current protocol) | fades 1 → 0 over half the remaining budget | 0 | 69.1% | 26.3% |
     | `p3a_bcfade_full` (`ppo.bc_decay_frac=1.0`) | fades 1 → 0 over the whole remaining budget | 0 | 72.3% | 34.3% |
     | `p3b_bcfade_x2` (`ppo.bc_decay_frac=2.0`) | fades 1 → 0.5 | 0.50 | 78.9% | 45.2% |
     | `p3c_bc_const03` (`ppo.bc_coef=0.3`, no decay) | 0.3, constant | 0.30 | 79.3% | 46.0% |
     | `p3d_bc_const1` (`ppo.bc_coef=1.0`, no decay) | 1.0, constant | 1.0 | 78.8% | 44.4% |
     | greedy EFT (rule) | | | 80.1% | 46.5% |

   - Any imitation weight that stays above 0 keeps the policy at greedy EFT's level (78.8–79.3%), and the three such runs end with agreement 0.99–1.00. Once the weight reaches 0, success drops even if that happens only near the end (p3a). None of the runs beat greedy EFT.
   - `agree` is measured on the first 4,096 samples of each update, about half of the 8 environments, so treat it as a rough signal. p3a ends at 0.99 agreement but 72.3% success.
   - **Options (decided 4 Nov 2026: constant weight 0.3, now the default in `configs/methods.yaml`):**
     - keep a constant imitation weight (for example `ppo.bc_coef: 0.3` with `bc_decay_frac` effectively infinite). Learned methods then stay close to greedy EFT, and differences between ablations will likely shrink;
     - keep the current protocol and report that PPO without the anchor loses 8–16 pp to greedy EFT;
     - keep the best checkpoint by validation (not implemented).
   - Whatever is chosen, the existing seed-0 runs under the fading protocol can be reported as a sensitivity result.
   - **Run names.** Changing `configs/methods.yaml` defaults makes `train.py` refuse to resume the existing `<method>_s0` directories (config mismatch). New-protocol runs need new names (for example a `--run-name` suffix), and `KEEP_CKPT` in `sync_runs.sh` must match them, or the old runs must be renamed first.
4. **Not implemented yet:**
   - the FD3QN, DDPG, and SAC baselines (EdgePlus compares against them);
   - the OBJ6 PID-robustness script;
   - the OBJ7 attack experiments (only crypto timings are measured, `scripts/bench_crypto.py`).

   Done since: paired statistics and bootstrap CIs (`scripts/compare.py`).

## 6. Next steps

Run each command from the repo root. Each `launch.py` call stops itself after `--minutes`; repeat it until `status.py` shows 100%.

1. ~~**Seed-0 batch, `A1_relgate`, imitation-weight pilot.**~~ Done 4 Nov 2026 (Section 5).
2. ~~**EdgePlus vs EdgePlus-PEC.**~~ Done 4 Nov 2026: trained, tested on both datasets, statistics, and figures (Section 5, first subsection).
3. ~~**Write the paper.**~~ Done 4 Nov 2026: `paper/EdgePlus-PEC.md`. The team decided to run no further experiments; the paper reports only work that was done.
4. **Not planned (team decision, 4 Nov 2026).** If the work is ever extended, the most valuable additions are, in order:
   - Vary the metro link that serve-in-place relies on. Add presets in `configs/systems.yaml` with, for example, `network: {metro_gbps: 0.1}`, and run greedy EFT on them with `run_heuristics.py --systems ... --tag _metro`.
   - Train seeds 1 and 2 for `edgeplus_pq`, `edgeplus_energy`, and `edgeplus_cons`.
   - Evaluate the learned policies at 50k and 100k (`evaluate.py --regimes light medium`).
   - Test DVFS alone, and deadline-aware ordering within slices to reduce the mMTC and tail-latency costs.

   Each `launch.py` or `evaluate.py` call is one step: wrap it in `timeout --kill-after=10 600`, use `--minutes 7`, and repeat it until done (Section 7).
5. **Also not planned:** FD3QN, DDPG, and SAC baselines; the OBJ6 PID script; A2–A6 retrained under the current protocol. Give the retrained runs names ending in `_bc_s<seed>`, so `sync_runs.sh` keeps their checkpoints.
6. **Merge:** mark draft PR #5 ready and merge it into `main`.

## 7. Operating rules (read before running anything)

- **Commands are capped at 30 minutes, and background processes die when the command returns.** Long work must be resumable. Use `launch.py --minutes 7` (at most 10) and repeat. Never use `nohup` or `&` for training.
- **One step per command, always with a hard cap.** Never chain training, evaluation, and syncing in one call. Wrap each command in `timeout --kill-after=10 <seconds>`, with seconds = minutes × 60 + 180.
  - **What the caps guarantee.** `train.py` stops at the first update after `--minutes`. It defers a due PSO round to the next call when less than 4 minutes remain. `launch.py` kills jobs at `--minutes` + 3. `evaluate.py --minutes 7` starts no new episode after 7 minutes, saves every episode as it finishes, and skips saved episodes on the next call.
  - **So a 7-minute step should end within about 10 minutes.** If one runs past 15 minutes, interrupt it. At most about 1 minute of work is lost.
  - **Then clean up and resume.** Check with `ps -eo args | grep -E "train.py|launch.py|evaluate.py"`, run `pkill -f train.py; pkill -f evaluate.py`, check `scripts/status.py`, and rerun the same command.
- **A message sent while a command runs interrupts it.** Checkpoints survive (saved every 5 updates and at the end), so just rerun the same command. Tell the user how long each step takes before starting it.
- **Save after every batch:** `bash scripts/sync_runs.sh push "what changed"`. Commit code changes to the branch and push.
- **Never commit `runs/`, `data/`, or `.venv/`** (they are in `.gitignore`). Runs go to `runs-mirror`, data to `data-mirror`.
- **Resuming a run** works only with the same method and `--set` overrides (`config.json` is checked). For a new configuration, use a new `--run-name`.
- **YAML parses `1e-4` as text.** `train.py` converts it, but `1.0e-4` is safer in YAML files.
- **Google Drive throttles the original AirDelhi download.** Use the mirror (the default in `download_data.py`).
- **Report honestly.** Where our results contradict EdgePlus or our hypotheses, say so and show the numbers. Don't promise outcomes before they are measured.

## 8. Running in another chat, or several in parallel

- **A new chat cannot see this sandbox.** It sees only GitHub. Everything needed is there: code on `feat/m3-edgeplus`, data on `data-mirror`, runs on `runs-mirror`.
- **To continue in a new chat,** paste the prompt from Section 9. The new session sets itself up (Section 3), restores the runs, and continues from Section 6.
- **To use two chats at once** (each gets its own 4-core sandbox):
  - split the work so no run is trained twice, for example chat A trains seed 1 and chat B trains seed 2, or split by method;
  - both may run `sync_runs.sh push`. It never overwrites a run that is further along. If two pushes collide, rerun the push;
  - only one chat should change code at a time; otherwise each must use its own code branch;
  - tell each chat exactly which seeds or methods it owns.
- **Autonomous mode** (chosen when starting a session) suits long unattended training better than this interactive mode. The 30-minute command limit still applies.
- **Your own computer also works:** follow Section 3. With more cores, add more jobs to one `launch.py` call (one job per core).

## 9. Prompt for a new session

Copy everything between the lines:

---
Continue my research project in the GitHub repo KeeratMalhotra/Cloud-Project, branch `feat/m3-edgeplus`. First read `docs/HANDOFF.md` completely and follow its setup (Section 3) and operating rules (Section 7) exactly. Then run `bash scripts/sync_runs.sh pull` and `python scripts/status.py`, tell me what state you find, and continue from the first unfinished item in Section 6. Keep every command under 10 minutes using `scripts/launch.py --minutes 7`, save runs with `bash scripts/sync_runs.sh push` after each batch, commit and push code changes, and report results with the numbers from the files. [If running in parallel with another chat: you own seed(s) ___ / method(s) ___ only.]
---

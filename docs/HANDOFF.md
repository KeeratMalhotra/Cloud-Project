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
2. **Propose our own framework, CHORUS**, and show it beats EdgePlus under identical conditions.
3. **Test both on a second, real dataset from New Delhi** (AirDelhi) that neither was tuned on.

The base paper and a background review are in `docs/`. The paper's numbers, gaps, and errors are analysed in `docs/EdgePlus_Deep_Review.md`.

## 2. Where everything is

GitHub repository: <https://github.com/KeeratMalhotra/Cloud-Project>

### Branches

| Branch | Content | State |
|---|---|---|
| `main` | Docs, data pipeline (M1), simulator (M2) | PRs #1–#4 merged |
| `feat/m3-edgeplus` | M3 work in progress: learners, warm start, PSO, training scripts, this file | pushed, draft PR #5 |
| `data-mirror` | Parquet copies of both datasets with SHA-256 checksums (210 MB, separate history) | complete |
| `runs-mirror` | Training runs: checkpoints, logs, PSO records, evaluations (separate history) | updated after each batch |

### Documents

| File | Content |
|---|---|
| `docs/EdgePlus_Explained.md` | Plain-language walkthrough of the base paper |
| `docs/EdgePlus_Deep_Review.md` | Full spec of EdgePlus, every reported number, issues found, decisions |
| `docs/CHORUS_Design.md` | Our framework, datasets, evaluation protocol, milestones M1–M7 |
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
| `scripts/` | `download_data.py`, `prepare_data.py`, `validate_simulator.py`, `calibrate_capacity.py`, `run_heuristics.py`, `train.py`, `launch.py`, `evaluate.py`, `status.py`, `sync_runs.sh`, `build_data_mirror.py` |
| `results/summary/` | committed JSON summaries: datasets, validation, calibration, rule-based results |
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

Versions used here: Python 3.11.15, torch 2.15.0+cpu, numpy 2.4.6, pandas 3.1.0. In some sandboxes `pip install --user` fails because the home directory is read-only; the `.venv` route above works.

**Checks.** `prepare_data.py` is deterministic. The training split must have 5,172,031 rows for CityPulse and 5,826,328 for AirDelhi (`results/summary/*_dataset.json`).

## 4. Status by milestone

| Milestone | Content | State |
|---|---|---|
| M1 | Data pipelines, labeling, sites, episodes for both datasets | done (PR #3) |
| M2 | Shared simulator, validation, capacity calibration, rule-based baselines | done (PR #4) |
| M3 | EdgePlus reproduction: CCPPO, advisory, PSO, ledger signals, baselines, ablations | in progress, see Section 5 |
| M4 | CHORUS stage 1: dispatcher, rejection, Lagrange multipliers, calibrated masks | not started |
| M5 | CHORUS stages 2–3: orchestrators, mobile agents, graph messages | not started |
| M6 | Delhi transfer and stress tests | not started |
| M7 | Paper (15–25 pages) | not started |

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
  - after that PPO takes over, and the imitation weight fades from 1 to 0 over half of the remaining budget.
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

### Results of the current protocol (seed 0, CityPulse validation days, 2 episodes, success at 150k)

| Run | PSO rounds | Success at 150k |
|---|---|---|
| `A1_full_s0` | 2, both rejected | 67.9% |
| `A2_no_pso_s0` | none | 69.1% |
| `A3_no_advisory_s0` | none | 72.4% |
| `A4_no_ledger_s0` | 2, both rejected | 70.4% |
| `A5_no_masks_s0` | | training, 50% |
| `A6_plain_ccppo_s0` | | training, 62% |
| `ccppo_lagr_s0` | | training, 63% |
| `ensemble_ppo_s0` | | training, 64% |

### Findings and open issues

1. **The paper's PSO gate never passes.** It requires failure ≤ 5% and stationary share ≥ 80%. On the PSO episodes, failures were about 3.5% but the stationary share was 39–40%, so every candidate was rejected (`runs/citypulse/A1_full_s0/pso.jsonl`). `A1_relgate` tests our relative gate. Not yet trained.
2. **The add-ons do not help so far.** Removing the advisory layer gave the best seed-0 result. These are 2 episodes from one training seed, so this is not yet reliable.
3. **PPO fine-tuning erodes the warm start.** `w1_bc` had 79.0% at 1.05M decisions; A1–A4 have 68–72% at 1.5M. In A2 and A3, agreement with greedy EFT falls from about 1.0 after the warm start to 0.72 and 0.62 at the end (`agree` in `log.jsonl`). Options to decide on:
   - keep a small imitation weight (`ppo.bc_coef` with `bc_decay_frac: 1.0`);
   - report as is;
   - keep the best checkpoint by validation (not implemented, and it would change the protocol for every method).
4. **Not implemented yet:**
   - the FD3QN, DDPG, and SAC baselines (EdgePlus compares against them);
   - the OBJ6 PID-robustness script;
   - the OBJ7 crypto-timing benchmark;
   - a statistics script (paired tests and bootstrap CIs across test episodes);
   - `docs/EdgePlus_Reproduction.md`, which `configs/methods.yaml` already mentions.

## 6. Next steps

Run each command from the repo root. Each `launch.py` call stops itself after `--minutes`; repeat it until `status.py` shows 100%.

1. **Finish the seed-0 batch**, then save:
   ```bash
   .venv/bin/python scripts/launch.py --minutes 7 -- "--method A5_no_masks --seed 0" "--method A6_plain_ccppo --seed 0" "--method ccppo_lagr --seed 0" "--method ensemble_ppo --seed 0"
   bash scripts/sync_runs.sh push "seed-0 batch 2 progress"
   ```
2. **Train `A1_relgate --seed 0`.**
3. **Decide on open issue 3** (imitation weight). If the protocol changes, every method must be retrained with it.
4. **Implement FD3QN, DDPG, and SAC** in `src/edgesim/agents/`:
   - reuse `features.py`, the masks, the reward, and the warm start, so the comparison stays fair;
   - add them to `methods.yaml` (`sync_runs.sh` already keeps their checkpoints).
5. **Train seeds 1 and 2** for every method: the same commands with `--seed 1` and `--seed 2`.
6. **Evaluate on test days**, 5 seeds, stress first:
   ```bash
   .venv/bin/python scripts/evaluate.py --runs runs/citypulse/A1_full_s0 runs/citypulse/A2_no_pso_s0 --split test --regimes stress --seeds 0 1 2 3 4
   ```
   One 150k episode with a learned policy takes about 45 s. Four run in parallel.
7. **Statistics:** paired Wilcoxon tests and bootstrap CIs against greedy EFT and between methods. Use the same test episodes for every method; they are paired by seed.
8. **Remaining pieces:** OBJ6 PID script, OBJ7 crypto timings, `docs/EdgePlus_Reproduction.md`. Then open the M3 PR.
9. **M4–M7:** follow `docs/CHORUS_Design.md`, Section 12.

## 7. Operating rules (read before running anything)

- **Commands are capped at 30 minutes, and background processes die when the command returns.** Long work must be resumable. Use `launch.py --minutes 7` (at most 10) and repeat. Never use `nohup` or `&` for training.
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

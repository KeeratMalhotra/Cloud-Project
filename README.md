# Cloud-Project: multi-agent RL for 6G edge-cloud task allocation

Research project with three parts:

1. Re-implement the base paper, **EdgePlus** (Taher et al., IEEE Internet of Things Journal, 2026).
2. Extend it with three system improvements, **EdgePlus-PEC**: slice-priority queues, energy-aware execution and mobility (DVFS, serving from neighbouring mobiles without driving, damped redeployment), and faster ledger consensus (time-bounded blocks, short BFT rounds). The scheduler itself is unchanged.
3. Compare both under identical conditions on CityPulse and on a second, real dataset from New Delhi (AirDelhi) that neither was tuned on.

## Documents

| File | Content |
|---|---|
| [`docs/EdgePlus_Explained.md`](docs/EdgePlus_Explained.md) | Beginner-friendly walkthrough of the base paper |
| [`docs/EdgePlus_Deep_Review.md`](docs/EdgePlus_Deep_Review.md) | Full technical spec, reported numbers, issues found, decisions |
| [`docs/Simulator.md`](docs/Simulator.md) | Shared simulator: model, EdgePlus interpretations, validation, calibration, rule-based baselines |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Current state, how to resume, and the next steps |

## Setup

Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -e .          # add ".[rl]" for PyTorch (needed from milestone M3)
```

Or with plain pip: `python -m venv .venv && .venv/bin/pip install -e .`

## Data

| Dataset | Role | Content |
|---|---|---|
| A: CityPulse pollution | Reproduction and main comparison (the data EdgePlus used) | 449 sites, 1 Aug to 1 Oct 2014, 5-minute steps, 7.9M rows. Generated data located in Aarhus, Denmark. |
| B: AirDelhi | Unseen-city test | PM sensors on 13 public buses in New Delhi, 1 Nov 2020 to 30 Jan 2021, about 3 s per reading, 12.5M rows. Real data. |

```bash
.venv/bin/python scripts/download_data.py --dataset citypulse    # about 50 MB
.venv/bin/python scripts/download_data.py --dataset airdelhi     # about 165 MB
.venv/bin/python scripts/prepare_data.py --dataset citypulse     # labels, topology, sample episodes (~15 s)
.venv/bin/python scripts/prepare_data.py --dataset airdelhi      # (~25 s)
```

Downloads come from the repository's `data-mirror` branch: Parquet copies with SHA-256 checksums, because the original hosts are slow or throttled. To rebuild the mirror from the original sources, use `--source original` and then `scripts/build_data_mirror.py`.

Outputs:

- `data/processed/<dataset>/`: labeled readings, label parameters, topology, server inventory. Regenerated, not committed.
- `results/summary/<dataset>_dataset.json`: dataset statistics.
- `docs/figures/<dataset>_*.png`: topology, arrivals, and severity figures.

## Simulator and baselines

```bash
.venv/bin/python scripts/validate_simulator.py                  # queue engine vs Erlang C, Pollaczek-Khinchine, Cobham
.venv/bin/python scripts/calibrate_capacity.py --dataset citypulse
.venv/bin/python scripts/run_heuristics.py --dataset citypulse --regimes stress --seeds 0 1 2 3 4
```

Details and results: [`docs/Simulator.md`](docs/Simulator.md).

## Layout

```
configs/            YAML settings: datasets, workload, topology, simulator (sim.yaml), system presets (systems.yaml)
src/edgesim/        shared package
  data/             loaders (citypulse, airdelhi), mirror, labeling, episode builder
  sim/              event-driven simulator: world, radio, calibration, env (core), metrics
  agents/           rule-based dispatchers, EdgePlus learner (CCPPO, advisory, PSO), features, rewards
  topology.py       city frame, weighted K-means sites, Delaunay neighbours, server inventory
  runner.py         episode loading and running; experiments.py runs grids in parallel
  plots.py          figures
scripts/            command-line entry points
docs/               papers, reviews, design, simulator, figures
results/summary/    committed result summaries
```

## Status

Data pipelines, the shared simulator, the EdgePlus reproduction, and EdgePlus-PEC are implemented, trained, and evaluated on both datasets. The paper is [`paper/EdgePlus-PEC.pdf`](paper/EdgePlus-PEC.pdf) (LaTeX source in `paper/latex/`, ready-to-upload Overleaf project in `paper/EdgePlus-PEC_overleaf.zip`, Markdown version in `paper/EdgePlus-PEC.md`); results so far and the next steps are in [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Data sources and licenses

Both datasets are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- CityPulse Smart City Datasets, pollution measurements: <http://iot.ee.surrey.ac.uk:8080/datasets.html>
- S. K. Chauhan, S. Ranu, R. Sen, Z. B. Patel, N. Batra, "AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks Track. Data portal: <http://www.cse.iitd.ac.in/pollutiondata>

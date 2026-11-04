# Cloud-Project: multi-agent RL for 6G edge-cloud task allocation

Research project with three parts:

1. Re-implement the base paper, **EdgePlus** (Taher et al., IEEE Internet of Things Journal, 2026).
2. Propose our own multi-agent RL framework, **CHORUS**, and compare it with EdgePlus under identical conditions.
3. Test both on a second, real dataset from New Delhi that neither method was tuned on.

## Documents

| File | Content |
|---|---|
| [`docs/EdgePlus_Explained.md`](docs/EdgePlus_Explained.md) | Beginner-friendly walkthrough of the base paper |
| [`docs/EdgePlus_Deep_Review.md`](docs/EdgePlus_Deep_Review.md) | Full technical spec, reported numbers, issues found, decisions |
| [`docs/CHORUS_Design.md`](docs/CHORUS_Design.md) | Our framework: agents, learning, datasets, evaluation protocol, build plan |

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

## Layout

```
configs/            YAML settings: datasets, workload (task generation), topology (sites and servers)
src/edgesim/        shared package
  data/             loaders (citypulse, airdelhi), mirror, labeling, episode builder
  topology.py       city frame, weighted K-means sites, Delaunay neighbours, server inventory
  plots.py          figures
scripts/            command-line entry points
docs/               papers, reviews, design, figures
results/summary/    committed result summaries
```

## Status

Milestone M1 (data pipelines and topology) is done. Next is M2: the simulator core and heuristic baselines. See the build plan in [`docs/CHORUS_Design.md`](docs/CHORUS_Design.md#12-build-plan).

## Data sources and licenses

Both datasets are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- CityPulse Smart City Datasets, pollution measurements: <http://iot.ee.surrey.ac.uk:8080/datasets.html>
- S. K. Chauhan, S. Ranu, R. Sen, Z. B. Patel, N. Batra, "AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks Track. Data portal: <http://www.cse.iitd.ac.in/pollutiondata>

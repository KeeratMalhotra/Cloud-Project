"""Print the current state of every training run from its files (no memory or guesswork involved).

Usage: python scripts/status.py [--dataset citypulse] [--markdown]

For each run: method, seed, decisions trained out of the budget, PSO rounds and whether they were accepted,
and the mean validation/test results of any evaluations saved next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {50_000: "light", 100_000: "medium", 150_000: "stress"}


def run_rows(dataset: str) -> list[dict]:
    rows = []
    for run in sorted((ROOT / "runs" / dataset).glob("*/")):
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        rec = json.loads(cfg_path.read_text())
        budget = int(rec["config"]["budget_decisions"])
        log = run / "log.jsonl"
        last = json.loads(log.read_text().strip().splitlines()[-1]) if log.exists() and log.stat().st_size else {}
        pso = [json.loads(x) for x in (run / "pso.jsonl").read_text().splitlines()] if (run / "pso.jsonl").exists() else []
        row = {"run": run.name, "method": rec["method"], "seed": rec["seed"],
               "decisions": last.get("decisions", 0), "budget": budget,
               "done": last.get("decisions", 0) >= budget, "checkpoint": (run / "checkpoint.pt").exists(),
               "pso": " ".join(f"r{p['round']}:{'acc' if p['accepted'] else 'rej'}" for p in pso) or "-"}
        for ev in sorted(run.glob("eval_*.csv")):
            df = pd.read_csv(ev)
            split = ev.stem.split("_")[-1]
            for n, g in df.groupby("n_tasks"):
                row[f"{split}_{REGIMES.get(n, n)}"] = f"{100 * g['success'].mean():.1f}% (n={len(g)})"
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="citypulse")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    df = pd.DataFrame(run_rows(args.dataset)).fillna("")
    if df.empty:
        print("no runs found (restore them with scripts/sync_runs.sh pull)")
        return
    df["progress"] = (100 * df["decisions"] / df["budget"]).round().astype(int).astype(str) + "%"
    cols = ["run", "method", "seed", "progress", "checkpoint", "pso"] + [c for c in df.columns if c.startswith(("val_", "test_"))]
    df = df[cols]
    print(df.to_markdown(index=False) if args.markdown else df.to_string(index=False))


if __name__ == "__main__":
    main()

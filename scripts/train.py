"""Train one learned method (configs/methods.yaml). Resumable: rerun the same command to continue.

Usage:
    python scripts/train.py --method A1_full --dataset citypulse --seed 0 --minutes 25
    python scripts/train.py --method A1_full --seed 0 --run-name tune_lr1e-4 --set ppo.lr=1e-4

Each call trains until the method's decision budget is reached or `--minutes` run out, saving a checkpoint
every few updates and at the end. Output: runs/<dataset>/<run-name>/ (checkpoint.pt, log.jsonl, config.json).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from edgesim.agents.ccppo import CCPPOTrainer
from edgesim.config import CONFIG_DIR, load_yaml
from edgesim.sim.config import deep_merge
from edgesim.sim.world import load_world

ROOT = Path(__file__).resolve().parents[1]


def method_config(method: str, sets: list[str]) -> dict:
    spec = load_yaml(CONFIG_DIR / "methods.yaml")
    if method not in spec["methods"]:
        raise SystemExit(f"unknown method {method!r}; known: {sorted(spec['methods'])}")
    cfg = deep_merge(spec["defaults"], spec["methods"][method])
    for item in sets:  # e.g. ppo.lr=1e-4, reward.eta_cloud=0.5, budget_decisions=3000000
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(val)
    return cfg


def parse_value(text: str):
    """YAML value, with plain floats like 1e-4 (which YAML 1.1 reads as strings) converted to float."""
    val = yaml.safe_load(text)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", default="citypulse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. ppo.lr=1e-4")
    args = ap.parse_args()
    torch.set_num_threads(1)
    cfg = method_config(args.method, args.set)
    name = args.run_name or f"{args.method}_s{args.seed}"
    run_dir = ROOT / "runs" / args.dataset / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    record = {"method": args.method, "dataset": args.dataset, "seed": args.seed, "config": cfg}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old["config"] != json.loads(json.dumps(cfg)):  # compare in JSON form (dict keys become strings)
            raise SystemExit(f"{run_dir} was started with a different config; use a new --run-name")
    else:
        cfg_path.write_text(json.dumps(record, indent=2))
    advisory = None if cfg.get("advisory") is None else copy.deepcopy(cfg["advisory"])
    loads = {int(k): float(v) for k, v in cfg["loads"].items()}
    trainer = CCPPOTrainer(load_world(args.dataset), run_dir, cfg["system"], cfg.get("sim"), args.dataset,
                           args.seed, ppo=cfg.get("ppo"), reward=cfg.get("reward"), advisory=advisory, loads=loads,
                           pso=cfg.get("pso"))
    state = trainer.train(int(cfg["budget_decisions"]), args.minutes)
    done = state["decisions"] >= int(cfg["budget_decisions"])
    print(json.dumps({"run": str(run_dir), "finished": done, **state}), flush=True)


if __name__ == "__main__":
    main()

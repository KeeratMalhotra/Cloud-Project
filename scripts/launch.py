"""Run several training commands in parallel worker processes, each with a hard time limit.

Usage:
    python scripts/launch.py --minutes 14 -- \
        "--method A1_full --seed 0 --run-name tuneA" \
        "--method A1_full --seed 0 --run-name tuneB --set ppo.lr=1e-4"

Every job is `python scripts/train.py <args> --minutes M`; its output goes to runs/logs/<run-name>.log.
Prints one status line per job when it ends. Jobs that overrun are killed (their last checkpoint survives).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args()
    log_dir = ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for job in args.jobs:
        toks = shlex.split(job)
        name = toks[toks.index("--run-name") + 1] if "--run-name" in toks else "_".join(toks[1:4:2])
        log = open(log_dir / f"{name}.log", "a")
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), *toks, "--minutes", str(args.minutes)]
        procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log))
        print(f"started {name}", flush=True)
    deadline = time.time() + 60 * args.minutes + 180
    for name, p, log in procs:
        try:
            p.wait(timeout=max(deadline - time.time(), 1))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        log.close()
        tail = (log_dir / f"{name}.log").read_text().strip().splitlines()[-1:] or [""]
        print(f"{name}: exit {p.returncode} | {tail[0][:200]}", flush=True)


if __name__ == "__main__":
    main()

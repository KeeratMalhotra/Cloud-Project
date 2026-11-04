---
inclusion: always
---

# Cloud-Project: context for every session

Research project: reproduce EdgePlus (multi-agent RL for 6G edge task allocation), propose our framework CHORUS, compare both on CityPulse and on the real AirDelhi dataset, then write a 15–25 page paper.

- Before doing anything, read `docs/HANDOFF.md`. It has the current state, the setup commands, and the next steps. Then run `python scripts/status.py` to see run progress from the files.
- Don't rely on memory for numbers. Take them from `results/summary/*.json`, `runs/*/eval_*.csv`, and `scripts/status.py`, and say which file each comes from.
- Commands time out after 30 minutes, and background processes die when the command returns. Train with `python scripts/launch.py --minutes 7 -- "<train.py args>" ...` and repeat until done. Tell the user how long each step will take; a message sent mid-command interrupts it, and checkpoints let the same command resume.
- After every training batch, run `bash scripts/sync_runs.sh push "message"` (runs go to the `runs-mirror` branch). Commit and push code changes. Never commit `runs/`, `data/`, or `.venv/`.
- Every learned method uses the same protocol: `configs/methods.yaml` defaults, which include the greedy-EFT warm start. Keep comparisons fair: same budget, same episodes, same features.
- Report results that contradict EdgePlus or our hypotheses as they are.

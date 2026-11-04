---
inclusion: always
---

# Cloud-Project: context for every session

Research project: reproduce EdgePlus (multi-agent RL for 6G edge task allocation), propose our framework CHORUS, compare both on CityPulse and on the real AirDelhi dataset, then write a 15–25 page paper.

- Before doing anything, read `docs/HANDOFF.md`. It has the current state, the setup commands, and the next steps. Then run `python scripts/status.py` to see run progress from the files.
- Don't rely on memory for numbers. Take them from `results/summary/*.json`, `runs/*/eval_*.csv`, and `scripts/status.py`, and say which file each comes from.
- Commands time out after 30 minutes, and background processes die when the command returns. Train with `python scripts/launch.py --minutes 7 -- "<train.py args>" ...` and evaluate with `python scripts/evaluate.py --minutes 7 ...`; repeat each until done. Run one step per command, never chained, and wrap it in `timeout --kill-after=10 600`. A 7-minute step should end within about 10 minutes. Tell the user how long each step will take; a message sent mid-command interrupts it, and checkpoints and saved episodes let the same command resume.
- After every training batch, run `bash scripts/sync_runs.sh push "message"` (runs go to the `runs-mirror` branch). Commit and push code changes. Never commit `runs/`, `data/`, or `.venv/`.
- Every learned method uses the same protocol: `configs/methods.yaml` defaults, which include the greedy-EFT warm start. Keep comparisons fair: same budget, same episodes, same features.
- Report results that contradict EdgePlus or our hypotheses as they are.

#!/usr/bin/env bash
# Save or restore training runs (checkpoints, logs, evaluations) through the orphan branch `runs-mirror`.
#
#   bash scripts/sync_runs.sh push "short message"   # copy runs/ to the branch, commit, push
#   bash scripts/sync_runs.sh pull                   # copy the branch's runs into runs/
#
# A run is copied only if the source copy is at least as far along as the destination copy (more lines in
# log.jsonl, i.e. more PPO updates), so two sessions sharing the branch never overwrite newer progress.
# Checkpoints are kept for the main runs (A1-A6, baselines, CHORUS) and for the warm-start reference run;
# tuning runs keep only their small files (config, logs, evaluations). Run from anywhere inside the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH=runs-mirror
WT="$(dirname "$ROOT")/.runs-mirror-worktree"
KEEP_CKPT='^(A[0-9]_[a-z_]+|A1_relgate|ccppo_lagr|ensemble_ppo|fd3qn|ddpg|sac|edgeplus[a-z0-9_]*)_s[0-9]+$|^w1_bc$'

cd "$ROOT"
cleanup() { git worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"; git worktree prune; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

# copy_runs SRC_RUNS DST_RUNS FILTER_CKPT(0/1): per-run copy, skipping runs that are further along in DST
copy_runs() {
  local src="$1" dst="$2" filter="$3" ds run name
  for dataset_dir in "$src"/*/; do
    [ -d "$dataset_dir" ] || continue
    ds="$(basename "$dataset_dir")"
    if [ "$ds" = "logs" ]; then mkdir -p "$dst/logs"; cp -a "$dataset_dir". "$dst/logs/"; continue; fi
    for run_dir in "$dataset_dir"*/; do
      [ -d "$run_dir" ] || continue
      run="$(basename "$run_dir")"
      local target="$dst/$ds/$run"
      if [ "$(lines "$target/log.jsonl")" -gt "$(lines "$run_dir/log.jsonl")" ]; then
        echo "  skip $ds/$run (the other copy is further along)"
        continue
      fi
      mkdir -p "$target"
      for f in "$run_dir"*; do
        name="$(basename "$f")"
        case "$name" in *.tmp) continue ;; esac
        if [ "$filter" = 1 ] && [ "$name" = "checkpoint.pt" ] && ! [[ "$run" =~ $KEEP_CKPT ]]; then continue; fi
        cp -a "$f" "$target/"
      done
    done
  done
}

fetch_branch() { git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"; }  # works in --single-branch clones

case "${1:-}" in
  push)
    msg="${2:-Update training runs}"
    cleanup
    if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
      fetch_branch
      git worktree add -q -B "$BRANCH" "$WT" "origin/$BRANCH"
    else
      git worktree add -q --detach "$WT"
      (cd "$WT" && git switch -q --orphan "$BRANCH")
      find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    fi
    mkdir -p "$WT/runs"
    copy_runs runs "$WT/runs" 1
    cat > "$WT/README.md" <<'EOF'
# Training runs (branch `runs-mirror`)

Checkpoints, training logs, PSO records, and evaluation CSVs from `runs/` on the code branches. This branch
has its own history; restore with `bash scripts/sync_runs.sh pull` from a code branch. See `docs/HANDOFF.md`.
EOF
    (cd "$WT" && git add -A && { git diff --cached --quiet && echo "nothing new to save"; } || \
       (git commit -q -m "$msg" && git push -q -u origin "$BRANCH" && echo "pushed to $BRANCH: $(git log --oneline -n 1)"))
    cleanup
    ;;
  pull)
    cleanup
    fetch_branch
    git worktree add -q --detach "$WT" "origin/$BRANCH"
    mkdir -p runs
    copy_runs "$WT/runs" runs 0
    echo "restored runs from $BRANCH ($(cd "$WT" && git log --oneline -n 1))"
    cleanup
    ;;
  *)
    echo "usage: $0 push \"message\" | pull" >&2
    exit 2
    ;;
esac

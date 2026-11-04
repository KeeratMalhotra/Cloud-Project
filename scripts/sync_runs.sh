#!/usr/bin/env bash
# Save or restore training runs (checkpoints, logs, evaluations) through the orphan branch `runs-mirror`.
#
#   scripts/sync_runs.sh push "short message"   # copy runs/ to the branch, commit, push
#   scripts/sync_runs.sh pull                   # copy the branch's runs into runs/ (keeps newer local files)
#
# Checkpoints are kept for the main runs (A1-A6, baselines, CHORUS) and for the warm-start reference run;
# tuning runs keep only their small files (config, logs, evaluations). Run from anywhere inside the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH=runs-mirror
WT="$(dirname "$ROOT")/.runs-mirror-worktree"
KEEP_CKPT='^(A[0-9]_[a-z_]+|A1_relgate|ccppo_lagr|ensemble_ppo|fd3qn|ddpg|sac|chorus[a-z0-9_]*)_s[0-9]+$|^w1_bc$'

cd "$ROOT"
cleanup() { git worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"; git worktree prune; }

case "${1:-}" in
  push)
    msg="${2:-Update training runs}"
    cleanup
    if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
      git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
      git worktree add -q -B "$BRANCH" "$WT" "origin/$BRANCH"
    else
      git worktree add -q --detach "$WT"
      (cd "$WT" && git switch -q --orphan "$BRANCH" && git rm -rq --cached . >/dev/null 2>&1 || true)
      find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    fi
    for dataset_dir in runs/*/; do
      ds="$(basename "$dataset_dir")"
      [ "$ds" = "logs" ] && { mkdir -p "$WT/runs/logs"; cp -a runs/logs/. "$WT/runs/logs/"; continue; }
      for run_dir in "$dataset_dir"*/; do
        run="$(basename "$run_dir")"
        dst="$WT/runs/$ds/$run"
        mkdir -p "$dst"
        for f in "$run_dir"*; do
          name="$(basename "$f")"
          if [ "$name" = "checkpoint.pt" ] && ! [[ "$run" =~ $KEEP_CKPT ]]; then continue; fi
          case "$name" in *.tmp) continue ;; esac
          cp -a "$f" "$dst/"
        done
      done
    done
    cat > "$WT/README.md" <<'EOF'
# Training runs (branch `runs-mirror`)

Checkpoints, training logs, PSO records, and evaluation CSVs from `runs/` on the code branches. This branch
has its own history; restore with `scripts/sync_runs.sh pull` from a code branch. See `docs/HANDOFF.md`.
EOF
    (cd "$WT" && git add -A && { git diff --cached --quiet && echo "nothing new to save"; } || \
       (git commit -q -m "$msg" && git push -q -u origin "$BRANCH" && echo "pushed to $BRANCH: $(git log --oneline -n 1)"))
    cleanup
    ;;
  pull)
    cleanup
    # explicit refspec: also works in clones made with --single-branch
    git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
    git worktree add -q --detach "$WT" "origin/$BRANCH"
    mkdir -p runs
    cp -a -n "$WT/runs/." runs/ 2>/dev/null || cp -a "$WT/runs/." runs/
    echo "restored runs from $BRANCH ($(cd "$WT" && git log --oneline -n 1))"
    cleanup
    ;;
  *)
    echo "usage: $0 push \"message\" | pull" >&2
    exit 2
    ;;
esac

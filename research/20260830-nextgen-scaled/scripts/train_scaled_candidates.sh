#!/usr/bin/env bash
set -euo pipefail

root_dir="research/20260830-nextgen-scaled/workspace"
python="./.venv-pathagon-gnn/bin/python"
targets="$root_dir/targets-14000-selective-depth8.jsonl"

[[ -f "$targets" ]] || { echo "missing assembled targets: $targets" >&2; exit 2; }

"$python" research/20260829-action-transition-policy/scripts/train_transition_policy.py \
    --targets "$targets" \
    --output-dir "$root_dir/model-next-xent-hidden32" \
    --epochs 60 --hidden 32 --seed 1

"$python" research/20260829-action-transition-policy/scripts/train_transition_policy.py \
    --targets "$targets" \
    --output-dir "$root_dir/model-next-rank-hidden32" \
    --epochs 60 --hidden 32 --seed 1 --rank-weight 0.5 --rank-margin 0.25

"$python" research/20260829-action-transition-policy/scripts/train_transition_policy.py \
    --targets "$targets" \
    --output-dir "$root_dir/model-next-virtual-hidden32" \
    --epochs 60 --hidden 32 --seed 1 --virtual-source

echo "scaled candidate training complete"

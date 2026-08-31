#!/bin/sh
set -eu

ROOT="research/20260829-nextgen-action-transition/workspace"
while :; do
    ready=1
    for path in "$ROOT"/targets-6000-0.jsonl "$ROOT"/targets-6000-1.jsonl "$ROOT"/targets-6000-2.jsonl "$ROOT"/targets-6000-3.jsonl "$ROOT"/targets-6000-4.jsonl "$ROOT"/targets-6000-5.jsonl "$ROOT"/targets-4000-0.jsonl "$ROOT"/targets-4000-1.jsonl "$ROOT"/targets-4000-2.jsonl "$ROOT"/targets-4000-3.jsonl; do
        [ -f "$path" ] || { ready=0; break; }
        [ "$(wc -l < "$path" | tr -d ' ')" -eq 1000 ] || { ready=0; break; }
    done
    [ "$ready" -eq 1 ] && break
    sleep 30
done

python3 research/20260829-nextgen-action-transition/scripts/audit_targets.py \
    --targets "$ROOT/targets-6000-*.jsonl" \
    --roots "$ROOT/roots-6000.jsonl" \
    --excluded-roots research/20260829-superdeep-contextual-evaluator/workspace/roots-turn-balanced-1920.jsonl \
    --expected 6000 > "$ROOT/audit-6000.json"
python3 research/20260829-nextgen-action-transition/scripts/audit_targets.py \
    --targets "$ROOT/targets-4000-*.jsonl" \
    --roots "$ROOT/roots-4000.jsonl" \
    --excluded-roots "$ROOT/exclude-roots-old-plus-6000.jsonl" \
    --expected 4000 > "$ROOT/audit-4000.json"
python3 research/20260829-nextgen-action-transition/scripts/combine_target_lanes.py \
    --lane "$ROOT/targets-6000-*.jsonl" \
    --lane "$ROOT/targets-4000-*.jsonl" \
    --output "$ROOT/targets-10000.jsonl" \
    --expected 10000 > "$ROOT/combine-10000.json"
python3 research/20260829-nextgen-action-transition/scripts/compare_teacher_depths.py \
    --shallow "$ROOT/targets-6000-*.jsonl" \
    --deep "$ROOT/targets-depth8-balanced.jsonl" > "$ROOT/depth8-vs-depth7.json"

PYTHON="./.venv-pathagon-gnn/bin/python"
for spec in \
    "xent:--seed 1" \
    "rank:--seed 1 --rank-weight 0.5 --rank-margin 0.25" \
    "rank-seed2:--seed 2 --rank-weight 0.5 --rank-margin 0.25" \
    "deep-only:--seed 1 --min-completed-depth 7"; do
    name=${spec%%:*}
    args=${spec#*:}
    # shellcheck disable=SC2086
    "$PYTHON" research/20260829-action-transition-policy/scripts/train_transition_policy.py \
        --targets "$ROOT/targets-10000.jsonl" \
        --output-dir "$ROOT/model-v3-$name" \
        --epochs 50 --hidden 24 $args
done

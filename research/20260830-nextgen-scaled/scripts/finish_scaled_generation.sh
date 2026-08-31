#!/usr/bin/env bash
set -euo pipefail

root_dir="research/20260830-nextgen-scaled/workspace"
prior_dir="research/20260829-nextgen-action-transition/workspace"
while :; do
    ready=1
    for path in "$root_dir"/targets-4000-0.jsonl "$root_dir"/targets-4000-1.jsonl "$root_dir"/targets-4000-2.jsonl "$root_dir"/targets-4000-3.jsonl; do
        [[ -f "$path" ]] || { ready=0; break; }
        [[ "$(wc -l < "$path" | tr -d ' ')" -eq 1000 ]] || { ready=0; break; }
    done
    [[ "$ready" -eq 1 ]] && break
    sleep 30
done

python3 research/20260829-nextgen-action-transition/scripts/audit_targets.py \
    --targets "$root_dir/targets-4000-[0-3].jsonl" \
    --roots "$root_dir/roots-4000.jsonl" \
    --excluded-roots "$root_dir/exclude-roots-old-plus-10000.jsonl" \
    --expected 4000 > "$root_dir/audit-4000.json"

python3 research/20260829-nextgen-action-transition/scripts/combine_target_lanes.py \
    --lane "$root_dir/targets-4000-[0-3].jsonl" \
    --output "$root_dir/targets-4000.jsonl" \
    --expected 4000 > "$root_dir/combine-4000.json"

python3 research/20260830-nextgen-scaled/scripts/filter_targets.py \
    --targets "$root_dir/targets-4000.jsonl" \
    --roots "$root_dir/roots-calibration-256.jsonl" \
    --output "$root_dir/targets-calibration-shallow-256.jsonl"

while :; do
    ready=1
    for path in "$root_dir"/targets-calibration-depth8-0.jsonl \
        "$root_dir"/targets-calibration-depth8-1.jsonl \
        "$root_dir"/targets-calibration-depth8-2.jsonl \
        "$root_dir"/targets-calibration-depth8-3.jsonl; do
        [[ -f "$path" ]] || { ready=0; break; }
        [[ "$(wc -l < "$path" | tr -d ' ')" -eq 64 ]] || { ready=0; break; }
    done
    [[ "$ready" -eq 1 ]] && break
    sleep 30
done

python3 research/20260829-nextgen-action-transition/scripts/combine_target_lanes.py \
    --lane "$root_dir/targets-calibration-depth8-[0-3].jsonl" \
    --output "$root_dir/targets-calibration-depth8-256.jsonl" \
    --expected 256 \
    --teacher-depth 8 \
    --teacher-nodes 2000000 > "$root_dir/combine-calibration-depth8.json"

python3 research/20260829-nextgen-action-transition/scripts/select_disagreement_depth8.py \
    --base "$root_dir/targets-calibration-shallow-256.jsonl" \
    --deep "$root_dir/targets-calibration-depth8-256.jsonl" \
    --roots "$root_dir/roots-calibration-256.jsonl" \
    --output-targets "$root_dir/targets-depth8-disagreement.jsonl" \
    --output-roots "$root_dir/roots-depth8-disagreement.jsonl" \
    --output-summary "$root_dir/depth8-disagreement-summary.json"

python3 research/20260829-nextgen-action-transition/scripts/merge_depth8_targets.py \
    --base "$root_dir/targets-4000.jsonl" \
    --depth8 "$root_dir/targets-depth8-disagreement.jsonl" \
    --output "$root_dir/targets-4000-selective-depth8.jsonl"

python3 research/20260829-nextgen-action-transition/scripts/combine_target_lanes.py \
    --lane "$prior_dir/targets-10000.jsonl" \
    --lane "$root_dir/targets-4000-selective-depth8.jsonl" \
    --output "$root_dir/targets-14000-selective-depth8.jsonl" \
    --expected 14000 \
    --allow-mixed-teacher > "$root_dir/combine-14000.json"

echo "scaled target assembly complete: $root_dir/targets-14000-selective-depth8.jsonl"

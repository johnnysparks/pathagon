# Small sorter pool

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

Limit the relative-regret sorter to eight heuristic actions, matching the
smallest practical training neighborhood and minimizing disruption to native
alpha-beta ordering.

## Outcome

Candidate/control points were 45.8%/54.2% at 2k, 49.0%/47.9% at 8k, and
51.0%/58.3% at 32k. The model still regressed at the low and high budgets;
all nine candidate logs passed replay audits.

## Decision

Retire this deployment-only ablation. Smaller intervention did not establish a
strength gain.

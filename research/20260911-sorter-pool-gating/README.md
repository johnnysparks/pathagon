# Sorter pool gating

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

The relative-regret model was trained on a 16-action neighborhood, while the
first arena asked it to rank every legal action. This path kept the model fixed
and limited inference to the existing heuristic pool to test for action-space
extrapolation damage.

## Outcome

Across 144 candidate games, candidate/control points were 45.8%/54.2% at 2k,
49.0%/47.9% at 8k, and 47.9%/58.3% at 32k. The 8k improvement was only 1.0
percentage point and the other budgets regressed. All nine candidate logs
passed native replay audits. Pool restriction alone is insufficient.

## Decision

Retain the result as a deployment integration ablation. Do not promote the
model or spend more arena budget on pool-size variants without a new target or
teacher.

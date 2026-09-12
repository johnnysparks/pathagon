# Sorter confidence gating

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

Test whether a learned reorder should be accepted only when its top Q value is
at least 0.2 above the heuristic first action.

## Outcome

The gate changed no aggregate result at 2k or 8k and only reduced the 32k loss
slightly. Candidate/control points were 45.8%/54.2% at 2k, 36.5%/47.9% at
8k, and 50.0%/58.3% at 32k. All nine candidate logs passed native replay
audits. The model's confidence is not calibrated well enough to be a safety
gate.

## Decision

Keep the gate as a negative diagnostic. The candidate and v4 transition-policy
default remain research-only and unchanged.

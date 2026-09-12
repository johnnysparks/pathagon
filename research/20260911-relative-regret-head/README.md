# Relative regret action head

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

Absolute search scores contained ±1e9 terminal sentinels, so the earlier
`tanh(score / 5000)` target was not a useful action scale. This path replaced
it with a within-position median/MAD regret target and froze the board/value
encoder while fitting only policy, Q, and advantage heads.

## Outcome

The 192-root, game-disjoint set was preserved exactly: 96 neutral and 96
decisive roots, split 96/96 between train and heldout. Heldout top-1 teacher
agreement rose from 6.3% to 42.7% and pairwise agreement from 53.4% to 65.5%.
Value MAE stayed exactly 0.3494 because the value head was frozen.

The paired 288-game arena failed at every budget: candidate/control points were
50.0%/54.2% at 2k, 36.5%/47.9% at 8k, and 50.0%/58.3% at 32k. All 18 logs
passed native replay audits with no duplicate sequences. The action target is
learnable on heldout roots but is not a safe deployed ordering signal.

## Project impact and decision

The exporter, robust target transform, and action-head-only trainer remain
research infrastructure. The checkpoint and ONNX model stay under ignored
`workspace/`; the transition-policy v4 default is unchanged. The result
motivated deployment-pool and teacher-envelope diagnostics rather than
promotion.

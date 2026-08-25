# 7x7 Q/advantage pilot

This pilot is the first replay batch with per-action root Q targets. It was
generated on 2026-08-24 from the re-evaluated 30k GNN and CNN checkpoints,
using CPU self-play with 32 MCTS simulations per move.

The JSONL archives are intentionally ignored by git. The manifest and this
report are the durable provenance for the local batch.

## Target semantics

Each move contains:

- `actionValues`: root-player-perspective Q estimates aligned to the state's
  legal-action order;
- `actionVisits`: the root-child visit count aligned to the same actions;
- `actionValueSource`: `mcts-root-q-v1`.

For a visited child, `actionValues` is the sign-corrected child mean value.
For an unvisited child, it is the sign-corrected heuristic afterstate seed.
`actionVisits` is therefore required when training: unvisited values should be
masked or down-weighted rather than treated as equally strong Q supervision.

The pilot does not serialize a separate advantage array. The initial
advantage target is derived from Q, for example
`A(s,a) = Q(s,a) - mean_legal_actions(Q)`, with the exact baseline to be fixed
when the action-value head is implemented.

## Audit

| Measure | Result |
| --- | ---: |
| Games / positions | 32 / 2,213 |
| Q-target coverage | 100% |
| Simulations per root | 32 |
| Mean legal actions | 116.62 |
| Mean visited actions | 12.32 |
| Mean visited-action fraction | 22.26% |
| Q range | -0.998 to 1.000 |
| Mean Q spread per position | 0.325 |
| Median Q spread per position | 0.295 |
| Selected action was Q-max | 11.75% |

All records passed the Python contract validator and the shared JSON Schema.
All 32 games ended by path completion; this is a target-quality pilot, not an
arena result or a training-ready scale batch.

# 4x4 five-piece endgame audit

The audit uses a 4x4 board with exactly five pieces per side, zero reserves,
and no overlapping pieces. It evaluates 24 positions: eight symmetry
transforms each of an immediate-win, forced-block, and forced-fork fixture.
Every position has 30 root actions.

Run it with:

```bash
.venv-pathagon-gnn/bin/python scripts/evaluate-4x4-endgame.py \
  --checkpoint training/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt \
  --budgets 0,32,128 \
  --solver-horizon 3
```

## Solver labels

`learning/gnn/solver.py` provides a generic legal-move AND/OR search with:

- a transposition table with exact, lower-bound, and upper-bound entries;
- repetition-count signatures in the cache key, so threefold repetition is
  not confused with a first occurrence of the same position;
- terminal wins, no-legal-action draws, the ply cap, and a finite proof
  horizon;
- no named block, fork, or one-away predicates.

The audit uses a three-ply solver horizon. A non-terminal position at that
horizon is treated as draw/unknown, which is sufficient to label these
one-move tactical fixtures without pretending this is a complete 4x4
tablebase. The solver labels are the correctness target; `tactical_root` is
still run only as a diagnostic baseline for the optional MCTS guard.

## Result

`policyAccuracy` is the action selected from the returned policy
probabilities. `visitAccuracy` is the most-visited root action. Both are
scored against solver-optimal actions.

| Family | Unguarded, 32 sims | Unguarded, 128 sims | Guarded diagnostic, 32 sims | Guarded diagnostic, 128 sims |
| --- | ---: | ---: | ---: | ---: |
| Immediate win | 8/8 | 8/8 | 8/8 | 8/8 |
| Forced block | 0/8 | 0/8 | 8/8 | 8/8 |
| True forced fork | 0/8 | 0/8 | 8/8 | 8/8 |

The current unguarded stack finds immediate wins but does not discover the
forced blocks or forks at these budgets. The diagnostic guard selects the
solver-optimal actions, but its visit counts still do not: it changes the
returned policy after search and does not rewrite the tree statistics.

This establishes a useful next experiment: use the GNN to order actions and
provide values, then add generic proof/search extensions and measure whether
the search—not a tactical rule list—recovers the solver labels.

# 4x4 five-piece tactical audit

The toy endgame suite uses a 4x4 board with exactly five pieces per side,
zero reserves, and no overlapping pieces. It exhaustively evaluates the root
actions, every opponent reply, and the next winning replies needed to prove an
immediate win, a forced block, or a true forced fork.

Run it with:

```bash
.venv-pathagon-gnn/bin/python scripts/evaluate-4x4-endgame.py \
  --checkpoint training/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt \
  --budgets 0,32,128
```

The suite contains 24 positions: eight transforms each of the three tactical
families. Every position has 30 root actions. The exhaustive tree contains
768 root/reply edges for immediate-win positions, 900 for forced blocks, and
684 for forced forks.

## Result

`policyAccuracy` is the action selected from the probabilities returned by
PUCT. The guarded mode applies the exact small-board tactical priority after
search; it does not rewrite the underlying visit counts.

| Family | Unguarded, 32 sims | Unguarded, 128 sims | Guarded, 32 sims | Guarded, 128 sims |
| --- | ---: | ---: | ---: | ---: |
| Immediate win | 8/8 | 8/8 | 8/8 | 8/8 |
| Forced block | 0/8 | 0/8 | 8/8 | 8/8 |
| True forced fork | 0/8 | 0/8 | 8/8 | 8/8 |

The exact tactical layer is therefore sufficient to solve the toy problem.
The guard is opt-in and restricted to boards of size 4 or smaller, so the
existing 7x7 search behavior is unchanged.

## Implementation boundary

`learning/gnn/tactics.py` uses graph distance for general path estimates and
caches simple 4x4 goal-path masks as a one-away prefilter. The authoritative
checks remain action-aware: captures, relocation restrictions, opponent
replies, and forced forks are evaluated by the legal move graph rather than by
a scalar board score.

This is a tactical regression suite, not a complete 4x4 tablebase. Full
win-distance solving still needs a transposition-table search with repetition
handling. The tactical features are the intended transition targets for the
separated Q/advantage experiment.

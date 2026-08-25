# 7x7 Q/Advantage backfill

This archive preserves the 3,000 games from the remote neural re-evaluation
run and adds complete `mcts-root-q-v1` action-value targets to every move.
The source JSONL files were not modified.

## Coverage

- 3,000 games
- 170,399 positions
- 170,399 positions with `actionValues` and `actionVisits`
- 32 root-search simulations per position
- deterministic root search without Dirichlet noise
- original checkpoint selected from each record's model hash

The three compressed JSONL files are directly discoverable by
`scripts/build-7x7-benchmark.py`; use `--require-action-values` to construct a
QAdvantage-only benchmark. The resulting verification build contained 3,096
unique Q-complete games / 177,221 positions, including the earlier QAdvantage
pilot.

To reproduce the derived benchmark on another machine:

```bash
./.venv-pathagon-gnn/bin/python scripts/build-7x7-benchmark.py \
  --root research/runs/gnn \
  --output work/benchmark-qadv \
  --require-action-values \
  --heldout-fraction 0.2 \
  --seed 20260825
```

The backfill command is available at
`scripts/backfill-7x7-action-values.py` for future archives.

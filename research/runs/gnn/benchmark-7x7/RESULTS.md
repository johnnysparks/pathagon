# 7x7 CNN/GNN benchmark

This is a deduplicated, seed-grouped benchmark over the archived 7x7 games.

| Split | Game records | Positions |
| --- | ---: | ---: |
| Raw 7x7 input | 3,251 | 158,109 |
| Unique records | 2,037 | 123,691 |
| Train | 1,621 | 97,940 |
| Held out | 416 | 25,751 |

The split removed 1,214 duplicate full-game records and keeps seed groups
disjoint between train and held out. The held-out records contain 225 distinct
seed groups because some archived files reused seeds for multiple records.

## Matched warm-start

Both models used 10,000 random replay updates, seed `20260825`, CPU training,
and D4 symmetry augmentation.

| Model | Parameters | Train policy loss | Train value loss |
| --- | ---: | ---: | ---: |
| CNN, 32 channels × 4 blocks | 87,395 | 2.502 | 0.634 |
| GNN, 64 channels × 8 layers | 100,227 | 2.435 | 0.638 |
| Compact GNN, 32 channels × 4 layers | 17,475 | 2.544 | 0.634 |

## Held-out positions

Metrics are exact selected-action prediction from the replay records. NLL is
lower; top-1 and top-5 are higher; value MSE is lower.

| Phase | Model | Policy NLL | Top-1 | Top-5 | Value MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| All | CNN | 2.291 | 48.9% | 75.6% | 0.603 |
| All | GNN | 2.112 | 49.6% | 75.8% | 0.609 |
| Placement | CNN | 2.250 | 47.3% | 73.2% | 0.773 |
| Placement | GNN | 2.024 | 49.9% | 73.4% | 0.780 |
| Relocation | CNN | 2.353 | 51.5% | 79.3% | 0.342 |
| Relocation | GNN | 2.248 | 49.2% | 79.7% | 0.345 |
| All | Compact GNN | 2.215 | 48.3% | 75.2% | 0.607 |
| Placement | Compact GNN | 2.128 | 49.6% | 73.0% | 0.780 |
| Relocation | Compact GNN | 2.350 | 46.2% | 78.6% | 0.341 |

The GNN currently learns a better-calibrated policy, while value quality is
effectively tied. This is a warm-start comparison, not yet a final playing
strength result.

## Small arena check

With four PUCT simulations per move and alternating colors over 40 games:

| Model | Wins | Losses | Draws | Average plies |
| --- | ---: | ---: | ---: | ---: |
| CNN vs seeded random | 8 | 5 | 27 | 161.7 |
| GNN vs seeded random | 5 | 6 | 29 | 165.1 |
| Compact GNN vs seeded random | 7 | 5 | 28 | 164.3 |

The arena is intentionally recorded as a smoke check; four simulations and a
draw-heavy ruleset are not enough to establish a meaningful strength ordering.

## Speed check

Single-threaded CPU `policy_value` timing over 100 repeated calls on one
placement state and one relocation state:

| Model | Parameters | Placement | Relocation |
| --- | ---: | ---: | ---: |
| CNN | 87,395 | 1.324 ms | 4.061 ms |
| GNN | 100,227 | 1.572 ms | 4.739 ms |
| Compact GNN | 17,475 | 1.198 ms | 3.688 ms |

The compact GNN is therefore the best current bulk-data candidate. Its policy
quality is close enough to the larger models to be useful, while its lower
capacity should also reduce self-play cost. It should generate diverse replay
with low-simulation PUCT, root noise, and a long temperature window; the large
GNN remains the learner being improved by that replay.

## Reproduce

```sh
./.venv-pathagon-gnn/bin/python scripts/build-7x7-benchmark.py \
  --root research/runs/gnn --output research/runs/gnn/benchmark-7x7 \
  --heldout-fraction 0.2 --seed 20260824
```

The checkpoint files are `cnn-warmstart.pt` and `gnn-warmstart.pt` in this
directory. The held-out scorer is `scripts/evaluate-7x7-checkpoint.py`.

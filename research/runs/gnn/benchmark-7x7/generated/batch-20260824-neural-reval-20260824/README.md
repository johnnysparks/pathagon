# Re-evaluated neural self-play batch

This batch was generated after the heuristic root-afterstate scoring and
terminal ply handling changes. It keeps the historical
`batch-20260824-neural-mix` archive intact while providing policy-aware replay
for the next training pass.

| Setting | Value |
| --- | --- |
| Board | 7x7, 14 reserves per player |
| Games | 3,000 total; 1,000 each Scout, Learner, CNN |
| Simulations | 4 per move |
| Temperature window | 32 plies |
| Ply cap | 196 |
| Workers/device | 8 CPU workers, CPU inference |
| Positions | 170,399 |
| Policy targets | 170,399 / 170,399 moves |

## Outcomes

| Agent | Wins | Draws | Draw rate | Positions | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Scout | 988 | 12 | 1.2% | 56,165 | `35d47237f1f1ef63878f5d28545857c83bd2ffc1eed6ac5012165dcd5adaeec1` |
| Learner | 975 | 25 | 2.5% | 60,837 | `c6498b0eaad98972f279680bac146f971c82c3480ac7a4e0fb002d0a6e21b64a` |
| CNN | 984 | 16 | 1.6% | 53,397 | `9467172114252c2a2e070c16e1f28278f3d2692fcae422d7747b6b3bf931b113` |

The historical 3,000-game neural mix had 1,723 max-ply draws (57.4%) and no
stored policy targets. Every record in this batch passed contract validation,
full replay reconstruction, and policy/legal-action alignment checks.

## Training smoke check

An 80/20 game-grouped split produced 2,443 training games / 139,815 positions
and 557 held-out games / 30,584 positions. The split was generated with:

```text
.venv-pathagon-gnn/bin/python scripts/build-7x7-benchmark.py \
  --root research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824 \
  --output <split-directory> --heldout-fraction 0.2 --seed 20260824
```

Compact 32-channel GNN and CNN warm-starts were trained for 2,000 CPU steps.
Final losses were GNN policy 2.3431 / value 0.9628 and CNN policy 2.2993 /
value 0.9691. On the held-out split, the new GNN reached policy top-1 51.8%
and top-5 78.7%; the new CNN reached 51.6% and 79.9%. Existing compact GNN
and CNN checkpoints scored 60.9% / 83.9% and 61.0% / 84.5%, respectively,
on the same split. These are smoke-check results; longer training and an
arena evaluation are still needed before replacing the active checkpoints.

## 10,000-update follow-up

Fresh compact models were trained for 10,000 single-example CPU updates on
the same 139,815-position training split. The final training losses were GNN
policy 2.0166 / value 0.9490 and CNN policy 1.9638 / value 0.9494.

| Model | Policy NLL | Top-1 | Top-5 | Value MSE |
| --- | ---: | ---: | ---: | ---: |
| Revaluated GNN, 10k | 1.637 | 57.9% | 84.3% | 0.937 |
| Existing compact GNN | 1.633 | 60.9% | 83.9% | 0.941 |
| Revaluated CNN, 10k | 1.604 | 57.3% | 85.3% | 0.938 |
| Existing CNN | 1.675 | 61.0% | 84.5% | 0.940 |

The longer pass closed most of the earlier gap and improved top-5 accuracy
for both models. The held-out scorer measures the selected replay action;
it does not yet score the full stored MCTS policy distribution.

In a direct color-balanced smoke arena at 4 PUCT simulations and a 196-ply
cap, the revaluated GNN tied the existing GNN 10-10, while the revaluated CNN
beat the existing CNN 20-0. All 40 games were decisive path wins with no
draws. The arena is deterministic and small, so it supports—but does not
prove—a checkpoint replacement decision.

The resulting checkpoints are preserved beside this batch:

- `reval-gnn-10k.pt` — SHA-256 `93f35fd7a54229a9e35fe4509f2536d55cbb42ec83f84bbb71dc8fd07db00541`
- `reval-cnn-10k.pt` — SHA-256 `6436649111995a738af0a499cfd165a8ffef1c1a87bec2e796b83232431e41dc`

## 30,000-update follow-up

Fresh compact models were also trained for 30,000 single-example CPU updates
on the same split. Final training losses were GNN policy 1.8772 / value
0.9454 and CNN policy 1.8054 / value 0.9444.

| Model | Policy NLL | Top-1 | Top-5 | Value MSE |
| --- | ---: | ---: | ---: | ---: |
| Revaluated GNN, 30k | 1.521 | 59.3% | 86.4% | 0.938 |
| Existing compact GNN | 1.633 | 60.9% | 83.9% | 0.941 |
| Revaluated CNN, 30k | 1.490 | 58.8% | 87.4% | 0.943 |
| Existing CNN | 1.675 | 61.0% | 84.5% | 0.940 |

At 30k, both revaluated models beat the existing checkpoints on policy NLL
and top-5 accuracy. Top-1 selected-action accuracy remains slightly higher
for the existing checkpoints. The value head is still weak and near the
zero-prediction baseline.

The 30k direct arena used 20 color-balanced games per matchup, 4 simulations,
and a 196-ply cap. Both GNN and CNN matchups ended 10-10 with no draws.
However, every game was won by the second player, so this deterministic
low-simulation arena is dominated by a turn-order artifact and should not be
used as a model ranking. A stochastic or higher-budget arena is required for
a meaningful playing-strength decision.

The 30k checkpoints are:

- `reval-gnn-30k.pt` — SHA-256 `7f41dbe135649d1116a0914bc0358a5ffae95749195ec20dd972035bbbea52f7`
- `reval-cnn-30k.pt` — SHA-256 `7439239ff343dcc422cb918560b1b72433f2e13628580427343a446f349c6f5a`

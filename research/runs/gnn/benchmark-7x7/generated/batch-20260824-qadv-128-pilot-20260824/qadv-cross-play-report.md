# The Q-Arbiter cross-play report

The Q/advantage pilot was evaluated with the checkpoint
`qadv-arbiter-7x7-v0.1.0.pt` against color-balanced 7×7 opponents. The 1,000
promotion-reference games are split into separate reproducible archives so
each opponent budget remains visible in the leaderboard import history.

| Opponent | Games | Q-Arbiter record | Average plies |
| --- | ---: | ---: | ---: |
| The Pathfinder | 700 | 0–700–0 | 13.5 |
| The Surveyor | 100 | 0–100–0 | 41.5 |
| Re-evaluated GNN 30k | 100 | 50–50–0 | 56.0 |
| Re-evaluated CNN 30k | 100 | 0–100–0 | 66.5 |
| **Total** | **1,000** | **50–950–0** | — |

The result is a valid reference run, not a promotion result: the direct Q
selector is currently substantially weaker than Pathfinder and Surveyor and
should return to training with broader action coverage and stronger targets.

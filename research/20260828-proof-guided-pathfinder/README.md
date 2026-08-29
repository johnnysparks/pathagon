# 20260828 Proof-guided Pathfinder

Status: `completed · not promoted`

## Idea

A small rule-grounded proof extension may improve the promoted tactical-safe
Pathfinder on positions where a one-move win or forced reply is visible, while
ordinary positions continue to use the same bounded Pathfinder search. The
extension should only answer questions that can be proved from legal moves and
must never silently increase the ordinary search budget.

## Starting point

The current supported baseline is `rust-pathfinder-v0.4.0-tactical-filter` on
`pathagon-rules-v1`, 7×7, 14 reserves per player. Its latest paired screen beat
unmodified depth-5 Pathfinder 629–169–2 over 800 games. Learned sorters and
seeded-position curricula improved selected offline metrics but did not exceed
the baseline in promotion-grade play.

This path uses the tactical-safe filter as the frozen control. The first
candidate is a pure-Rust proof-guided variant with an explicit proof horizon,
proof node budget, and whole-game match accounting. Any candidate that does not
beat the control on paired colors and held-out tactical fixtures remains
unpromoted.

## What happened

The proof logic passed its focused 7×7 fixture, selecting the bounded winning
relocation at the expected root. Three 40-game same-seed pilots were then run
against the frozen tactical-filter control:

- horizon 3, 15,000 proof nodes: 17–23–0;
- horizon 2, 15,000 proof nodes: 17–23–0;
- horizon 3, 15,000 proof nodes, trusting only forced-win proofs: 18–22–0.

The proof branch appeared in ordinary games, but it did not produce a repeatable
strength gain and added variable endgame work. The candidate was therefore not
promoted. Its Rust implementation remains as a tested research hook for a
future, separately controlled endgame study.

## Data and artifacts

Disposable games, reports, logs, and proof traces belong in this path's ignored
`workspace/`. Reusable game records or labels will be promoted only through the
canonical `data/` corpus policy. No checkpoint or implementation-shaped tensor
will be promoted unless it has durable value.

## Project impact

This path did not change the supported opponent or deployed assets. It added a
focused Rust regression fixture and preserved the negative result so the proof
layer is not retried as an unqualified strength improvement.

## Next decision

Decision: do not promote. Continue with the budgeted evaluator search in the
sibling path, keeping the tactical-safe filter as the control.

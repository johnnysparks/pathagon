# 20260828 Budgeted Pathfinder

Status: `completed · promoted`

## Idea

The next higher-ranked opponent may be a larger, still explainable Pathfinder
configuration rather than a learned model. The tactical-safe root filter is
already the strongest promoted behavior; increasing search depth or node budget
could improve ordinary move selection if the gain is repeatable and the browser
runtime remains acceptable.

## Starting point

The frozen control is `rust-pathfinder-v0.4.0-tactical-filter` on 7×7 boards
with 14 reserves per player. Recent evidence supports the filter over the
unmodified depth-5 search, while learned sorters and seeded curricula remain
below the parent. The proof-guided experiment in the sibling research path was
negative at 40 games (17–23 and 18–22 after a stricter win-only rule), so this
path keeps the proof layer out of the budget ladder.

## Protocol

Use paired colors, two randomized opening plies, identical seeds, 7×7/14
reserve rules, and a 120-ply cap for the pilot. Compare one variable at a time:
depth 5 versus depth 4 at the same node budget, then a larger node budget only
if the fixed-budget candidate is promising. A promotion screen must use a
larger held-out ladder, record node and latency cost, pass the tactical fixture
suite, and include representative replay inspection.

## What happened

The larger-search pilots did not clear the bar: depth 5 versus the depth-4
control scored 15–25–0 in 40 games, and a 5,000-node depth-4 candidate versus
the 2,000-node control scored 7–12–1 in 20 games. Both were rejected because
extra compute alone did not improve whole-game strength.

Filter-aware deterministic evaluator evolution was then run for two
generations, population 8, four training pairs, and twelve held-out evaluation
pairs. The promoted weights are:

`path=241, material=112, capture=887, structure=40, threat=154, edge=74`

The final 120-game held-out arena used paired colors, two randomized opening
plies, the same depth-4/2,000-node/beam-8 envelope, and a 120-ply cap. The
candidate scored 70 wins, 47 losses, and 3 draws against
`rust-pathfinder-v0.4.0-tactical-filter` (59.6% game points; 5,926,568 total
nodes; 78.9 seconds). The 8-game post-build smoke confirmed the distinct
stable Rust identity and paired-color wiring.

Replay audit covered all 120 games: all rule-valid, zero capture mismatches,
zero illegal records, two threefold repetitions, one max-ply draw, and one
duplicate trajectory. The reviewed set included short tactical wins, a
capture-heavy 94-ply win, a control win, and the max-ply draw; no illegal or
capture-corrupt behavior was found.

## Data and artifacts

Generated games, summaries, and logs belong in this path's ignored `workspace/`.
Only canonical replay-bearing games or other reusable labels may be promoted to
`data/`; temporary reports and repeated archives will not be committed.

## Project impact

The promoted Rust identity is
`rust-pathfinder-v0.5.0-trained-evaluator`, with its deployable manifest in
`pathagon/opponents/pathfinder-v0.5.0-trained-evaluator.json`. The browser
catalog, cross-play roster, lab model list, WASM engine assets, and focused web
tests now expose the opponent as The Pathfinder · Trained. Generated games and
training reports remain in this path's ignored `workspace/` as experiment
evidence rather than being copied into canonical data.

## Next decision

Decision: promote The Pathfinder · Trained with a provisional rating. Keep the
original tactical-safe Pathfinder available as the default control and rerun a
larger ladder after production telemetry accumulates.

# 20260829 Turn-balanced contextual evaluator

Status: inconclusive

## Idea

The first million-node contextual-evaluator path used roots sampled at mostly
even plies, leaving 424 Light-to-move positions and only 56 Dark-to-move
positions. That made its player-conditioned behavior under-trained for Dark.
This path keeps the strongest parts of that design—source-disjoint roots,
super-deep Pathfinder labels, phase-specific features, and held-out evaluation—
while enforcing an exactly balanced Light/Dark root sample and fitting separate
weights for the player to move.

## Starting point

The incumbent is Pathfinder v0.5 with weights
`path=241, material=112, capture=887, structure=40, threat=154, edge=74`
and a tactical-safe root filter. The parent path is
`research/20260829-superdeep-contextual-evaluator/`; it showed that one-million-
node labels are feasible but its Light-heavy roots and single shared evaluator
did not clear the arena gate. The durable source corpus is
`data/corpora/games-v1/manifest.json` (38,547 games, 128,953 observations).

## What happened

The corrected generator sampled one eligible 7x7 game per root and alternated
even and odd source plies. The pilot produced 480 roots (240/240), then the
scaled run produced 1,920 roots (960/960) across opening, placement, movement,
and late-game phases. Every root was labeled with a depth-7, 1,000,000-node,
beam-32 Pathfinder teacher, split by player to move, and fit with conservative
phase-conditioned vectors.

The 480-root player-conditioned screen was legal but lost overall (7–8–1 in 20;
4/10 as Light and 3/10 as Dark). The 1,920-root fit did not improve held-out
teacher agreement: with 1,000 iterations Light was 41/158 versus 45/158 for
v0.5 and Dark tied at 56/160; stronger shrinkage only tied Light and improved
Dark by one root (57/160). Up-weighting the completed-depth-7 rows returned to
baseline held-out agreement. A separate depth-7-only 20-game screen was 9–10–1
and did not clear a strength gate. All arenas replayed legally, but no variant
earned a 100-game promotion run.

## Data and artifacts

Generated roots, 1M-node labels, training reports, and arena logs live in the
ignored `workspace/` directory for this path. A Lambda one-game worker was
packaged for a possible fan-out, but deployment was blocked by the project's
managed SCP (`lambda:GetFunction`); no cloud resources were created. The
research executable and worker are not durable opponents. Only a candidate
that passes the final gate may be promoted into versioned `data/` and Rust
runtime code; otherwise the evidence remains narrative here.

## Project impact

No opponent, app roster, or durable corpus has been changed. The path tested
whether correcting player-turn distribution and substantially increasing the
million-node training sample unlocks value without overfitting the incumbent;
the larger sample did not produce a reliable strength signal.

## Next decision

Retire this avenue for the current cycle and retain its labels as the best
available supervision. Do not promote the contextual evaluator or spend more
compute on this six-feature linear family unless a new feature representation
or stronger teacher objective is introduced. The incumbent v0.5 remains the
supported strength line.

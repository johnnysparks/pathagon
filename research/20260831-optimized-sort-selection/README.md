# 20260831 Optimized sort selection for a d5/b256/500k teacher

Status: completed — no promotion

## Idea

The direct Rust Pathfinder search is now the strength reference. The next learner should spend inference capacity choosing the small root beam that Pathfinder searches, so the training set should be labeled by the proposed practical control: depth 5, beam width 256, and a 500,000-node ceiling. The working hypothesis is that source-disjoint action-transition labels plus a narrow-candidate replay gate can retain most of the direct-search strength at lower runtime cost.

## Starting point

The current default transition-policy model is `pathfinder-action-transition-v4-xent`. It was trained from earlier d7/1M/b32 labels and is already the user-facing default. Existing sorter experiments were mixed and remain research-only. The canonical `data/corpora/games-v1` corpus provides replayable, content-addressed games; Rust exposes the exact transition feature encoder used by the learner.

## What happened

A research-only Rust emitter replays canonical games, selects deterministic roots at paired even/odd checkpoints, keeps game-level train/heldout separation, computes legality/tactical safety/action-transition features in Rust, and records d5/b256/500k teacher labels. A first sharding attempt exposed duplicate roots at merge time; the selector was corrected to use one fixed global selection before slicing. A first schedule exposed Light-only coverage; paired checkpoints and a fail-closed turn audit corrected that as well.

The final set contains 1,920 roots from 162 source games: 1,536 train and 384 heldout, with 965 Light and 955 Dark positions. It contains opening, placement, movement, and late rows. The teacher consumed 510,422,928 nodes; 1,319 roots completed depth 5 and 601 exhausted at an earlier completed depth.

Three 16-unit explicit-source scorers were trained for 60 epochs. Heldout top-1 was 12.24% for all three seeds, versus 22.92% for the native heuristic; the best heldout top-3/top-8/top-16/top-32 rates were 26.30%/43.23%/58.59%/83.59%, versus 45.31%/70.05%/83.07%/93.75% for the heuristic. Three wider/rank-aware variants were also screened and were no better. Training only fully completed labels removed movement and late coverage, so it was not retained as a candidate.

The best new seed was tested as a real top-32 root-limited Rust sorter for 20 paired games against direct d5/b256/500k: 6 wins, 10 losses, 4 draws, 40% points, with mean candidate telemetry of 206,129 nodes and completed depth 4.47. This is negative promotion evidence, not a production change.

## Data and artifacts

Disposable labels, reports, and checkpoints belong under this path's ignored `workspace/`. The emitter is `src/main.rs`; the existing transition-policy trainer remains the Python orchestration layer. A durable target sidecar will only be promoted to `data/` after it passes replay, legality, source-disjointness, and heldout strength gates.

## Project impact

This path produced a reusable research emitter, merge tool, audit, and a Rust narrow-root agent, but no supported learner. Rust still owns legality and alpha-beta authority. The current v4 model remains the user-facing default and rollback/control; none of the research models or root limits were promoted.

## Next decision

Keep v4 active. The next useful experiment is a larger or more informative target signal (for example, completed-depth-only labels with deliberate movement coverage, or teacher regret/rank targets) before revisiting a narrowed root beam. The ignored workspace retains the generated labels, reports, checkpoints, and arena archive for reproducibility during this research cycle.

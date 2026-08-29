# 20260828 Random midgame and late-game datasets

Status: completed

## Idea

Create two deterministic synthetic position families for the 7×7, 14-reserve
ruleset:

- midgame roots with unexpected random occupancy and either color to move;
- late-game roots containing a nearly complete path for the side to move,
  with two path pieces removed so a one-move win is guaranteed.

The late-game family is intended to create compact, known-value terminal
examples that can be ingested by the existing historyless golden-table
builder. The midgame family is intended for model coverage and ranking
experiments, not as a claim that the positions were reached by legal play.

## Starting point

The supported rules authority is `pathagon/engine-rs`, with the Python GNN
lab's `GameState.seeded` adapter used for exploratory position construction.
Seeded roots must preserve the 14-piece inventory per color, contain no active
winning path, and have at least one legal action. Existing exact proof code
can exhaustively label small boards, while the current 7×7 tactical boundary
can prove immediate wins.

## What happened

`build-random-phase-datasets.py` was added with reproducible seed and count
controls. A midgame sample defaults to 14 Light and 14 Dark pieces on random
open squares, with the turn selected from a shuffled color-balanced schedule.
The optional `--none-count` removes that many pieces from board inventory and
places them in reserves, which is the rule-valid interpretation of capture
imbalance. For example, `--none-count 14` preserves 28 total pieces across
the two colors but makes their board counts and reserve counts vary.

Late-game samples create a connected seven-cell path spanning the target
color's goal edges, remove two distinct path cells, then fill both colors'
remaining board inventory on squares outside the complete path. Candidates are
accepted only when the root is non-terminal and exhaustive one-ply legal
action inspection finds an action that wins for the target color. A matching
one-move replay record is emitted for each accepted sample.

The generated smoke batch used seed `2026082801`, 100 midgame roots, 100
late-game roots, and a second 100-root midgame batch with 14 missing inventory
pieces. All late-game roots had at least one immediate win; both turn colors
were represented. The exact counts and hashes remain in the ignored
`workspace/` report for the run.

The generator does not claim that a shallow heuristic or arbitrary deeper
search will choose a particular winning action. Its durable guarantee is the
stronger, checkable one-ply property: at least one legal action from the
declared side to move transitions to a terminal path win. This is the right
boundary for producing terminal lookup-table inputs without treating a model
prediction as truth.

## Data and artifacts

The reusable code is retained here. Generated JSONL datasets, replay records,
reports, and any locally built golden shards belong under this path's ignored
`workspace/` directory. Nothing is promoted into `data/` in this exploration.

The late replay stream follows the existing replay contract closely enough for
`scripts/build-golden-terminal-table.py` to ingest it when passed as a scoped
research root. That command should write to a deliberate review location;
refreshing the canonical `data/golden/` table is a separate promotion decision.

## Project impact

No production Rust, browser, or canonical data behavior changed. The path
adds a focused synthetic-data generator and tests its inventory, non-terminal,
path-gap, immediate-win, replay-contract, and reproducibility invariants.

## Next decision

Retain the generator as research infrastructure. Before promotion, compare
these roots against held-out reachable positions and ordinary whole-game
strength. If the midgame set is useful, promote only a small versioned fixture
or a canonical replay/label sidecar with explicit provenance. If late-game
terminal coverage is useful, review canonical-key duplication and table size
before merging its replay records into the golden source inventory.

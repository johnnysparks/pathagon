# Learning evaluation archives

This directory contains JSON/JSONL records from offline learner evaluations,
pairwise matches, and historical leagues. It is not the source code for the
league runner; see [`learning/gnn/league.py`](../../../learning/gnn/league.py)
and [`docs/WORKFLOWS.md`](../../../docs/WORKFLOWS.md).

## Record requirements

Every archive should include or be accompanied by:

- board size and reserve configuration;
- versioned agent IDs and checkpoint hashes;
- search/PUCT budgets;
- seed and color-assignment policy;
- winner, draw reason, and complete move list;
- whether the games were used for training or held out for evaluation.

Ratings start at 1000 and use online Elo updates with `K=24` in the existing
league tools. They are provisional comparison signals, not human Elo.

## Archive families

- `league-*` files are historical 7x7 checkpoint league snapshots.
- `scout-policy-*` files are focused pairwise experiments for learned players.
- `rust-*.jsonl` and machine-labelled JSONL files are offline replay archives
  suitable for validation and dataset construction.
- The 7x7 records are the canonical strength evidence. Smaller-board curriculum
  and regression material lives in the dedicated self-play and evaluation paths.

Large archives should eventually move to the external/ignored data path with a
tracked manifest and hash. The public copies under `public/lab/` are currently
static mirrors and should be generated rather than hand-edited.

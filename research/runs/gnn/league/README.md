# Learning evaluation archives

This directory contains JSON/JSONL records from 7x7 learner evaluations,
pairwise matches, and league snapshots. It is not the source code for the league
runner; see [`research/gnn/league.py`](../../../gnn/league.py) and
[`docs/WORKFLOWS.md`](../../../../docs/WORKFLOWS.md).

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
- These 7x7 records are the canonical strength evidence; curriculum and
  regression material belongs in the dedicated self-play and evaluation paths.

Large archives belong in the external or ignored data path with a tracked
manifest and hash. The public copies under `public/lab/` are static mirrors for
the read-only lab and should be generated rather than hand-edited.

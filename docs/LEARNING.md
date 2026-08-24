# Learning laboratory

The learning code is a research pipeline, not an automatic browser-opponent
promotion system. The canonical comparison target is 7x7 with 14 reserves per
player.

## Learner families

| Family | Role | Status |
| --- | --- | --- |
| Rust tabular book | Exact-state replay baseline | Historical diagnostic |
| Compact GNN | Fast neural data generator and learner candidate | Active research |
| Full GNN | Higher-capacity graph policy/value model | Active research |
| CNN | Fixed 7x7 comparison and browser export candidate | Active research |

The graph implementation can exercise smaller boards for curriculum and
regression, but strength comparisons should use the canonical 7x7 distribution.

## Current artifacts

- [`training/gnn/benchmark-7x7/`](../training/gnn/benchmark-7x7/) contains the
  earlier deduplicated benchmark and held-out report.
- [`training/gnn/benchmark-7x7-full-20260824/`](../training/gnn/benchmark-7x7-full-20260824/)
  contains the full-data split manifest and ignored replay payloads.
- [`training/gnn/round-full-20260824/`](../training/gnn/round-full-20260824/)
  contains compact, CNN, and full-GNN checkpoints with scoring snapshots.
- [`training/rust-v1/learned-100/`](../training/rust-v1/learned-100/) contains
  the historical exact-state learner.

## Dataset rules

Use imported/offline records with contract validation. Deduplicate complete
games, split by game or seed group, preserve color balance, and record the
source manifest. Human games should be included only when their consent and
privacy status is explicit.

Symmetry augmentation is a training transform, not a substitute for held-out
games. Draws remain draws and should not be silently converted into losses.

## Training and evaluation

The command-level workflow is in [`WORKFLOWS.md`](WORKFLOWS.md). Every serious
run should retain:

- dataset manifest and split seed;
- model architecture and checkpoint hash;
- optimizer/device/seed configuration;
- held-out policy/value metrics;
- color-balanced pairwise games against named opponents.

Held-out prediction metrics diagnose learning. Pairwise results determine
whether a candidate is strong enough to enter the promotion conversation.

The proposed transition-focused alternative to a scalar board value is
documented in
[`SEPARATED-VALUE-ACTION-POLICY.md`](SEPARATED-VALUE-ACTION-POLICY.md). It is
currently a design experiment, not an active model contract.

The first exact transition-focused regression suite is documented in
[`4X4-ENDGAME-TACTICAL-AUDIT.md`](4X4-ENDGAME-TACTICAL-AUDIT.md).

## Browser boundary

The GNN and CNN remain research candidates until they pass the evaluation gates
and their runtime/export path passes parity checks. A browser model must carry a
stable agent ID, model hash, board configuration, and search budget.

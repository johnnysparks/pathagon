# Learning laboratory

The learning code is a research pipeline, not an automatic browser-opponent
promotion system. The canonical comparison target is 7x7 with 14 reserves per
player.

## Learner families

| Family | Role | Status |
| --- | --- | --- |
| Rust tabular book | Exact-state replay baseline | Historical diagnostic |
| Compact GNN | Historical model family; retained only when a Rust experiment selects an artifact | Supporting |
| Full GNN | Historical higher-capacity comparison | Historical |
| CNN | Current browser inference artifact | Product support |
| QAdv-guided search | Historical Rust-search experiment | Historical |

The graph implementation can exercise smaller boards for curriculum and
regression, but strength comparisons should use the canonical 7x7 distribution.

## Current artifacts

- [`research/corpora/games-v1/`](../research/corpora/games-v1/) is the durable,
  content-addressed game and observation corpus.
- [`research/experiments/20260827-pathfinder-rust-sorter/`](../research/experiments/20260827-pathfinder-rust-sorter/)
  is the active Pathfinder lineage record and contains its selected small ONNX
  sorter.
- [`research/fixtures/`](../research/fixtures/) contains curated universal
  tactical and parity evidence.

`research/runs/` and `training/` are ignored workspaces, not artifact indexes.

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

Store these as a durable experiment record following
[`EXPERIMENTS.md`](EXPERIMENTS.md), including the parent agent/experiment,
canonical game membership, outcome even when the attempt fails, and hashes for
large artifacts kept outside Git.

Held-out prediction metrics diagnose learning. Pairwise results determine
whether a candidate is strong enough to enter the promotion conversation.

The proposed transition-focused alternative to a scalar board value is
documented in
[`SEPARATED-VALUE-ACTION-POLICY.md`](SEPARATED-VALUE-ACTION-POLICY.md). It is
implemented as an isolated pilot. The direct Q-max selector is retained for
diagnosis; the current playing path uses QAdv to narrow candidates and then
checks the best replies before selecting a move. The guided selector is an
evaluation wrapper, not yet a browser opponent or promotion claim.

The Rust PUCT path now has an opt-in `use_action_value_seeds` mode. When
enabled by the QAdv-guided player, the QAdv action head seeds unvisited
children at every expanded node, not only at the root. This is search guidance
and move ordering—not a tactical proof: rule-grounded proof extensions remain
the next experiment. Ordinary policy/value and browser/WASM paths leave the
flag disabled so their prior inference cost and behavior are unchanged. The
native QAdv CLI enables it by default; pass `--no-qadv-tree-seeds` to reproduce
the root-only baseline.

Native QAdv play also has a selective rule-grounded proof extension. It checks
roots with an immediate win, an opponent's immediate threat, or a configured
capture burst; QAdv values order the bounded proof search, while the final
labels come only from legal rule transitions. The default native experiment is
proof-off; pass `--tactical-proof-horizon 3` (and optionally
`--tactical-proof-nodes 50000`) to enable a three-ply experiment. The proof path
is intentionally not enabled for browser/WASM play yet.

The first exact transition-focused regression suite is documented in
[`4X4-ENDGAME-TACTICAL-AUDIT.md`](4X4-ENDGAME-TACTICAL-AUDIT.md).

The current Pathfinder iteration includes a research-only compact root
sorter. `research/gnn/league.py` exposes `SorterOnlyPathfinderAgent` (policy
ordering only) and `SorterPathfinderAgent` (the same ordering plus a
transposition-aware table and one-ply tactical extension). Both preserve the
Pathfinder evaluator and root beam. Use
`scripts/benchmark-pathfinder-sorter.py` for matched, color-balanced screens;
these variants are not promoted to browser play until the longer 7x7 gate is
positive.

For the native screen, `--sorter-all-actions` expands the ONNX scoring pool
from Pathfinder's heuristic head to every legal root action. This can improve
candidate recall but costs more inference time, so both pool choices are
recorded in the agent parameters and benchmark reports.

The production-shaped native adapter is in `engine-rs`: build with the
`inference` feature and pass `--sorter-onnx` to use an ONNX policy as the root
sorter, or `--sorter-qadv-onnx` to use QAdv action values. In both cases Rust
owns rules, alpha-beta search, node budgets, and the final move; ONNX only
orders Pathfinder's bounded root candidates. The benchmark wrapper is
`scripts/benchmark-rust-pathfinder-sorter.py`, and its current screens remain
research evidence rather than a promotion claim.

## Browser boundary

The GNN and CNN remain research candidates until they pass the evaluation gates
and their runtime/export path passes parity checks. A browser model must carry a
stable agent ID, model hash, board configuration, and search budget.

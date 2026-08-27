# Engine

[`engine-rs/`](../engine-rs/) is the native Pathagon rules, evaluator, search,
self-play, and contract implementation. It uses bitboards and a size-aware
kernel for development across several board sizes; the product and learning
target is 7x7 with 14 reserves.

## Verify rules and search

```bash
npm run rust:test
npm run test:parity
```

The parity suite compares TypeScript, Rust, and Python behavior using shared
fixtures and generated positions. Run it after changing rules, action encoding,
captures, win resolution, or replay normalization.

## Run native matches

```bash
npm run rust:selfplay -- \
  --games 100 --seed 20260823 --depth 4 --nodes 90000 --beam 40
```

Use `--jsonl` for complete machine-readable records and `--opponent search` for
search-versus-search matches. For a provenance-stamped archive, use
`scripts/run-rust-archive.sh` as described in [`WORKFLOWS.md`](WORKFLOWS.md).

The inference feature includes a native GNN parity harness for the current QAdv
checkpoint. Export the shared policy/value artifact or the full QAdv artifact
with `research.gnn.export_gnn`, then use `--eval-only` to cross-check numerical
outputs before running matches. The guided mode carries the same temperature,
opening-mix, and Pathfinder blend controls as Python; `--qadv-onnx` exercises
the direct action-value head. QAdv-guided Rust PUCT also opts into action-value
seeds for unvisited children at every expanded node; the plain policy/value and
browser/WASM paths keep this mode disabled for a clean baseline comparison. For
an explicit QAdv A/B baseline, pass `--no-qadv-tree-seeds`. On tactical roots,
the native QAdv player can also run a bounded rule proof (for example, three
plies and 50,000 nodes); pass `--tactical-proof-horizon 3` to enable it, or
sweep it with `scripts/benchmark-rust-qadv-ablation.py`. Proof is disabled by
default because it is an experimental latency trade-off. Set
`--tactical-simulations` equal to `--simulations` when you want a strict
fixed-budget ablation; the historical default remains 512 tactical simulations
for self-play generation.

The native Pathfinder sorter path is also ONNX-backed and keeps the game loop
in Rust. Pass `--sorter-onnx <policy-value.onnx>` to run alpha-beta Pathfinder
with the ONNX policy used only to reorder the bounded root beam; tune the
number of reordered candidates with `--sorter-top-k`. The optional
`--sorter-root-limit` caps the root candidates (zero defaults to twice the
Pathfinder beam), while `--sorter-min-margin` and
`--sorter-max-heuristic-gap` can require a confident, evaluator-compatible
reorder. Add
`--sorter-all-actions` to score the complete legal root set before taking that
top-k hint (slower, but it can discover moves outside the heuristic head). Use
`--opponent sorter` to place the same candidate on the other side of a match.
This path requires the inference feature (the Rust `tract-onnx` runtime) and
does not depend on Python at play time.

The same root-ordering adapter accepts a QAdv artifact through
`--sorter-qadv-onnx <qadv.onnx>`. In that mode the native engine uses the
artifact's action-value head as the sorter signal while Pathfinder's
alpha-beta evaluator remains authoritative. Use `--opponent qadv-sorter` for
an explicit QAdv-sorter opponent, or run
`python3 scripts/benchmark-rust-pathfinder-sorter.py --sorter-kind qadv` for
the matched wrapper. This is an experiment, not a promoted strength claim.

For a repeatable release-mode timing sample using the full guided recipe, run
`python3 scripts/benchmark-rust-qadv.py`. It reports both engine time and wall
time, along with the exact game/node totals, so engine changes can be compared
on the same seeds and simulation budget.

For the opt-in small-board tactical proof mode, add
`--tactical-proof-horizon 3`. It applies only to boards up to 4x4, searches
the full legal action set, and records the mode in the agent parameters. The
default 7x7 heuristic search is unchanged.

## Curated corpus

```bash
npm run rust:corpus -- --games 100 --seed 20260823 --opponent search
```

This command writes legacy compact output into the ignored run workspace. After
a serious run completes, normalize its games into the reviewable canonical
corpus under [`research/corpora/games-v1/`](../research/corpora/games-v1/) and
link the resulting game keys from an experiment record. Large intermediate
archives remain outside Git.

## Evaluator training

```bash
npm run rust:train -- \
  --generations 3 --population 6 \
  --training-pairs 6 --evaluation-pairs 12
```

Candidates are selected on one split and tested on a disjoint split. The
handcrafted incumbent remains the champion unless the promotion gate is met.

## Browser/WASM boundary

The Rust/WASM adapter and CNN inference path are active integration work. The
build command emits browser artifacts under `public/engine/` and
`public/engine-inference/`:

```bash
npm run build:engine
```

The browser engine should not be switched by default until the WASM boundary
passes contract and generated parity tests. This keeps a faster implementation
from changing game semantics or leaderboard agent identity unexpectedly.

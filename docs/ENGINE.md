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
the direct action-value head.

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

The compact corpus under [`research/corpora/rust-v1/`](../research/corpora/rust-v1/) is reviewable
knowledge, not disposable output. Large experimental archives should remain
outside Git until they earn promotion into a curated corpus.

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

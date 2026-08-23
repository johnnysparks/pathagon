# Rust headless engine

`engine-rs` is a dependency-free native implementation of the Pathagon rules, evaluator, iterative-deepening search, and deterministic self-play harness. It uses two 49-bit bitboards inside `u64` values and is intended for high-volume training and eventual WebAssembly use.

## Verify rules and search

```bash
cargo test --manifest-path engine-rs/Cargo.toml --all-targets --release
```

Both engines consume `fixtures/rules-parity.tsv`. The fixture covers exact A-B-A capture, rejected A-B-B-A capture, simultaneous four-direction capture, both winning axes, diagonal non-wins, the one-reply capture-hole prohibition, and movement-phase restrictions.

## Run headless matches

```bash
cargo run --release --manifest-path engine-rs/Cargo.toml --bin pathagon-selfplay -- \
  --games 100 --seed 20260823 --depth 4 --nodes 90000 --beam 40
```

Use `--opponent search` for search-versus-search games and `--jsonl` to emit complete machine-readable game records before the aggregate summary.

The Rust and TypeScript harnesses use the same Mulberry32 seed algorithm, color alternation, random-opening convention, threefold-repetition rule, maximum-ply draw, action encoding, and schema-v2 move diagnostics.

## Boundary

The Rust engine is currently native/headless. It does not yet replace the browser engine. Before compiling it to WebAssembly, parity will be extended from curated fixtures to generated state/action corpora. That prevents a fast engine with subtly different rules from training the wrong game.


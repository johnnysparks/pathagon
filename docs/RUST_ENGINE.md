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

## Build a persistent strategy corpus

```bash
npm run rust:corpus -- --games 100 --seed 20260823 --opponent search
```

This writes three deterministic, diffable files under `corpus/rust-v1/`:

- `games.tsv` stores one replayable game per line. Every move is exactly two base64url characters (12 bits); captures and intermediate boards are recovered by replaying the rules.
- `positions.tsv` is an exact position/action book keyed by agent. It stores the deepest completed answer, its score and prior node cost, plus observed wins, losses, and draws.
- `manifest.json` declares the encoding and corpus counts.

The next run loads `positions.tsv` before play. A cached action is reused only when it came from the same versioned agent, matches the exact game state, remains legal, and was searched at least as deeply as the current request. Corpus writes deduplicate identical games and sort both files, so rerunning a seed does not create Git noise.

The tracked corpus is knowledge, not disposable output: keep curated batches small enough to review in Git. Large experimental runs should remain external until they earn promotion into the canonical corpus.

The Rust and TypeScript harnesses use the same Mulberry32 seed algorithm, color alternation, random-opening convention, threefold-repetition rule, maximum-ply draw, action encoding, and schema-v2 move diagnostics.

## Boundary

The Rust engine is currently native/headless. It does not yet replace the browser engine. Before compiling it to WebAssembly, parity will be extended from curated fixtures to generated state/action corpora. That prevents a fast engine with subtly different rules from training the wrong game.

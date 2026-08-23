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

## Learn a tabular candidate

The Rust engine also has a deliberately modest replay learner. It builds an
exact-state action book from completed games, scores each observed action by
the mover's empirical win/draw rate, and falls back to normal search for
unseen states. This is useful for testing the learning pipeline on a small
archive; it is not a neural network or a generally valid policy model.

Convert archived schema-v2 JSONL into the Rust replay format, then build the
book:

```bash
python3 scripts/jsonl-to-rust-games.py \
  --input /tmp/pathagon-rust-selfplay.jsonl \
  --output /tmp/pathagon-rust-selfplay.games.tsv
npm run rust:learn -- \
  --games /tmp/pathagon-rust-selfplay.games.tsv \
  --out training/rust-v1/learned-100
```

Evaluate it without changing the browser opponent:

```bash
npm run rust:selfplay -- \
  --learned training/rust-v1/learned-100/learned.tsv \
  --learned-min-visits 1 \
  --opponent search --games 20 --seed 20261001 --jsonl
```

`--learned-min-visits 2` is the safer default. Keep the candidate local until
it is tested on a disjoint evaluation set and beats the incumbent repeatedly;
the current 100-game sample has too few repeated exact positions to justify
promoting it to the web game.

## Evolve evaluation weights

```bash
npm run rust:train -- --generations 3 --population 6 --training-pairs 6 --evaluation-pairs 12
```

Each candidate mutates the six interpretable evaluation weights and plays paired-color games against the incumbent. Every candidate sees the same training openings within a generation. Only the best training candidate sees the disjoint evaluation seed range, preventing direct selection on the promotion games.

A candidate is promoted only when it beats the incumbent on both splits and earns at least 55% of held-out points. The output keeps training and evaluation histories in separate replayable corpora alongside `report.json` and `champion.json`. This gate reduces overfitting; it does not by itself establish statistical significance or human Elo.

The Rust and TypeScript harnesses use the same Mulberry32 seed algorithm, color alternation, random-opening convention, threefold-repetition rule, maximum-ply draw, action encoding, and schema-v2 move diagnostics.

## Boundary

The Rust engine is currently native/headless. It does not yet replace the browser engine. Before compiling it to WebAssembly, parity will be extended from curated fixtures to generated state/action corpora. That prevents a fast engine with subtly different rules from training the wrong game.

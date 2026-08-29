# Rust engine

This crate is the authoritative implementation of Pathagon rules, legal move
generation, captures, winner detection, search, native self-play, corpus
replay, training utilities, and browser WASM adapters.

```bash
cargo test --manifest-path pathagon/engine-rs/Cargo.toml --all-targets --release -j1
cargo run --release --manifest-path pathagon/engine-rs/Cargo.toml --bin pathagon-selfplay -- --games 2
npm run build:engine
```

Changes to rules, records, or opponents require focused unit coverage,
cross-runtime fixtures where applicable, and inspection of representative game
outputs. Stable interchange types live in [`../contracts/`](../contracts/);
durable fixtures and corpora live in [`../../data/`](../../data/).

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

The promoted Pathfinder profile is depth 4 with a 2,000-node budget and beam
width 8. The browser and native defaults use this same envelope; larger
profiles must be requested explicitly.

The runtime boundary keeps `apply_action_json` as a position-only compatibility
API and also exposes `apply_action_transition_json` (and the corresponding
WASM export) for auditable moves. The transition result includes the acting
player, action, captured squares, and the complete post-move position. Run
`npm run build:engine` to regenerate the checked-in browser bundle after
consumers adopt that endpoint.

The browser adapter also exposes a deadline-bounded tactical search export.
It checks the browser clock during iterative deepening and returns the last
fully completed iteration, preserving a legal fallback move when the deadline
is reached. The web app runs this search in a cancelable Worker; the historical
depth-4 / 2,000-node / beam-8 profile remains the supported default until a
later strength study clears the three-second research gate.

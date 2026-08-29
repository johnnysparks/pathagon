# Pathagon core

This directory is the supported game system. Code promoted here must be
documented, tested, and compatible with the Rust engine boundary.

| Path | Responsibility |
| --- | --- |
| [`engine-rs/`](engine-rs/) | Authoritative rules, search, self-play, training utilities, native runners, and WASM adapters. |
| [`opponents/`](opponents/) | Stable opponent identities and the artifacts/configuration required to play them. |
| [`contracts/`](contracts/) | Versioned cross-runtime interchange schemas and fixtures. |

Research implementations do not become shared dependencies in place. Promote
useful behavior into Rust, add coverage and documentation, then reference the
originating dated research path from the change.

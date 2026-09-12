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

The supported runtime is also the learning boundary: Rust owns legal action
generation, tactical safety, search, and replay verification. Experimental
learners may propose action ordering or value estimates, but a promoted
opponent must keep native search authoritative and document matched-budget
strength and cost evidence.

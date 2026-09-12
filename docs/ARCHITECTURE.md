# Architecture

Pathagon is organized by lifecycle and ownership rather than programming
language.

| Layer | Contract |
| --- | --- |
| `apps/*` | Independently deployable products. Each app owns its infrastructure and UI details. |
| `pathagon/*` | Supported game/runtime code. Rust is the authority for promoted rules, search, and opponents. |
| `data/*` | Small, strict, reusable datasets and fixtures with stable formats. |
| `research/YYYYMMDD-*` | Historical questions, exploratory code, narratives, and disposable workspace artifacts. |
| `scripts/*` | Shared tools that still serve more than one current subsystem. |

## Dependency direction

Apps may consume `pathagon` and `data`. Core code may consume contracts and
fixtures in `data`, but must not depend on an archived research implementation.
Research may import anything while exploring. If research succeeds, port the
behavior into Rust and promote only the data/artifacts that have durable value.

## Opponent learning boundary

The learning loop has four owners. Rust defines the rules, legal actions,
tactical safety, search envelope, and native replay audit. Research paths create
source-disjoint trajectories and teacher targets, then train action and value
signals as separate hypotheses. The runtime integrates a learned signal only
as an advisory root-order hint that preserves native legality and search
ordering. Promotion requires held-out action ranking, value calibration,
paired whole-game strength, and node/depth/latency evidence at the intended
budget. A better fit to a teacher or a single favorable arena is not enough.

The browser retains a TypeScript rules adapter for UI state and parity checks,
but the Rust engine is the supported high-throughput and opponent runtime. A
second app should use a shared Rust/WASM or contract boundary rather than import
files from `apps/web`.

## Monorepo conventions

- The root package orchestrates workspaces and cross-project checks.
- Deployment-specific configuration stays inside its app.
- README files live at ownership boundaries and explain local decisions.
- Generated dependencies, build output, and research workspaces stay ignored.
- A feature with no current owner, documentation, or meaningful coverage is a
  deletion candidate even if a recent experiment once used it.

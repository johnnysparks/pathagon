# Pathagon

Digital preservation of Mark Fuchs's two-player wooden strategy game: a
mobile web client, deterministic rules engines, playable opponents, and a
reproducible 7x7 learning laboratory.

## Start here

```bash
# browser development
npm run dev

# full JavaScript, Python, Rust, and parity regression suite
npm test

# Rust engine tests
npm run rust:test

# Python learner tests
./scripts/test-python.sh
```

For the project map, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For
the commands used to generate, archive, train, evaluate, and deploy, read
[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md).

## Repository map

| Path | Responsibility |
| --- | --- |
| [`app/`](app/) | Browser product, game UI, reference/coaching behavior |
| [`engine-rs/`](engine-rs/) | Rust rules, search, self-play, training, and WASM adapters |
| [`learning/`](learning/) | Python GNN/CNN research code and model tooling |
| [`training/`](training/) | Checkpoints, datasets, league archives, and experiment reports |
| [`selfplay/`](selfplay/) | TypeScript offline arena and league runner |
| [`scripts/`](scripts/) | Reusable build, archive, parity, and evaluation commands |
| [`contracts/`](contracts/) | Versioned cross-runtime interchange contract |
| [`corpus/`](corpus/) | Small curated Rust corpus suitable for Git |
| [`fixtures/`](fixtures/) | Rules and parity fixtures |
| [`public/`](public/) | Deployable static assets and published lab snapshots |
| [`db/`](db/) and [`drizzle/`](drizzle/) | D1 access code and schema migrations |
| [`docs/`](docs/) | Architecture, workflows, data policy, and experiment guidance |

## Current decisions

- The canonical research target is 7x7 with 14 reserves per player.
- Rust is the high-throughput rules/search authority for native self-play and
  training. TypeScript remains the browser reference and coaching
  implementation used for regression checks.
- The GNN and CNN are research candidates, not automatically promoted browser
  opponents. Promotion requires disjoint evaluation, color balance, and
  recorded configuration.
- Leaderboard games come from imported or offline matches. The web UI displays
  results; it is not the authoritative game generator.
- Large replay archives remain external or ignored. Git stores code, schemas,
  manifests, compact corpora, selected checkpoints, and reviewable reports.

## Core documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime ownership and data flow
- [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) — development, self-play, training, and evaluation
- [`docs/DATA.md`](docs/DATA.md) — archive, dataset, checkpoint, and Git policy
- [`docs/ENGINE.md`](docs/ENGINE.md) — Rust/native/WASM engine details
- [`docs/LEARNING.md`](docs/LEARNING.md) — learner families and promotion status
- [`docs/GAME_ARCHIVE.md`](docs/GAME_ARCHIVE.md) — D1 archive and offline imports
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Sites, builds, and authentication

## Deployment

The site is built with Vinext/Vite and deployed through Sites. The normal
checkpoint path is to edit source, run the relevant tests, commit a coherent
milestone, and deploy from the pushed commit. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for environment-specific details.

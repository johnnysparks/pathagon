# Pathagon

Pathagon is a monorepo for preserving and advancing Mark Fuchs's two-player
wooden strategy game. The default view of the repository is deliberately the
current product: deployable apps, the tested Rust game system, promoted
opponents, and durable game data. Exploratory work is retained separately as a
dated research history.

## Start here

```bash
npm install
npm run dev
npm test
```

The root package is an orchestration workspace; application dependencies and
deployment configuration belong to the app that uses them.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`apps/`](apps/) | Deployable products. `apps/web` contains the game and leaderboard lab. |
| [`pathagon/`](pathagon/) | Stable rules, search, opponents, contracts, and runtime code. |
| [`data/`](data/) | Small, durable, strictly validated datasets and fixtures kept in Git. |
| [`research/`](research/) | Date-first research paths, including dead ends and disposable local artifacts. |
| [`docs/`](docs/) | Project index, rules, contribution guidance, policies, and active direction. |
| [`scripts/`](scripts/) | Shared maintenance, evaluation, and migration tools. |

Read [`docs/README.md`](docs/README.md) for the documentation index and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the ownership boundaries.

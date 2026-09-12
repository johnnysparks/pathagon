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

## Current project goal

The long-term AI goal is an opponent that gets better through play by learning
useful intuition about positions and actions. The intended design is a
rules-authoritative Rust search with a learned advisory layer: action intuition
should improve root choices or tie breaks without replacing legality, tactical
safety, or alpha-beta search. State-value learning is evaluated separately from
action ranking, and every candidate must pass held-out, whole-game, legality,
and search-cost gates before it can become supported.

Start with [`docs/RESEARCH.md`](docs/RESEARCH.md) before opening historical
experiments. It records the current workflow, the evidence behind it, and the
mistakes that future work should avoid.

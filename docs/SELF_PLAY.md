# Pathagon self-play

The self-play layer treats the browser game engine as the reference rules implementation. Every match is seeded and records every action, capture, node count, result, and termination reason.

## Run an arena

```bash
npm run selfplay -- --mode arena --games 20 --seed 20260822
```

This alternates the tracked champion between light and dark against a seeded random baseline. Two random opening plies diversify otherwise deterministic matchups.

Use `--opponent surveyor` or `--opponent pathfinder` for a named regression arena. Pathfinder is deliberately outside the promotion pool: evaluator mutations should not be permanently blocked merely because a candidate is tested with a smaller search budget than the expert browser opponent.

## Train evaluation weights

```bash
npm run selfplay:train -- --generations 5 --population 8 --games 12 --seed 20260822
```

Each candidate mutates six interpretable weights: path distance, material, immediate captures, connected structure, capture threats, and edge control. Candidates play paired-color matches with shared opening seeds against both the incumbent and recent historical champions.

A candidate is promoted only when it:

- beats the incumbent head-to-head;
- does not lose a matchup against any member of its evaluation pool; and
- wins at least 55% of decisive games across the pool.

Generated match logs live under `selfplay/progress/runs/` and are ignored by Git. The promoted `selfplay/progress/champion.json` is tracked so strategy progress is reviewable and reproducible.

Training never changes the live opponent automatically. A promoted champion must pass the tactical regression suite and a named evaluation arena before its weights are copied into the browser opponent.

## Run the historical league

```bash
npm run selfplay:league -- --games 8 --seed 20260823
```

This runs a paired-color round robin over tracked champions and produces provisional within-pool ratings. These numbers compare agents in this league only; they are not human Elo ratings.

The browser now exposes four compute levels: random Coin Flip, the intentionally naive one-ply Lunatic pattern heuristic, the novice two-ply Surveyor, and the four-ply Pathfinder. Search uses iterative deepening, alpha-beta pruning, a transposition table, tactical move ordering, and a strict node budget. That makes strength tunable without making mobile response time unbounded.

## Reproducibility contract

- Same engine version, configuration, and seed produce the same game record.
- Light and dark are alternated during evaluation.
- Repeated positions are draws after the third occurrence.
- Games are also capped by `--maxPlies` to prevent endless movement cycles.
- Elo is not assigned until enough champion-versus-pool games exist for calibration.

## Next engine milestones

1. Accumulate enough promoted champions for a meaningful league.
2. Add puzzle-derived tactical gates and adversarial opening suites.
3. Move browser search into a Web Worker so larger budgets never block touch input.
4. Run large batches in a Rust headless engine after parity tests match the TypeScript reference engine.

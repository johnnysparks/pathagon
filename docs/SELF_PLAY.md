# Pathagon self-play

The self-play layer treats the browser game engine as the reference rules implementation. Every match is seeded and records every action, capture, node count, result, and termination reason.

## Run an arena

```bash
npm run selfplay -- --mode arena --games 20 --seed 20260822
```

This alternates the tracked champion between light and dark against a seeded random baseline. Two random opening plies diversify otherwise deterministic matchups.

## Train evaluation weights

```bash
npm run selfplay:train -- --generations 5 --population 8 --games 12 --seed 20260822
```

Each candidate mutates the champion's path, material, and capture weights. Candidates play paired-color matches against the incumbent using shared opening seeds. A candidate is promoted only when it wins more games than it loses.

Generated match logs live under `selfplay/progress/runs/` and are ignored by Git. The promoted `selfplay/progress/champion.json` is tracked so strategy progress is reviewable and reproducible.

Training never changes the live opponent automatically. A promoted champion must pass the tactical regression suite and a named evaluation arena before its weights are copied into the browser opponent.

## Reproducibility contract

- Same engine version, configuration, and seed produce the same game record.
- Light and dark are alternated during evaluation.
- Repeated positions are draws after the third occurrence.
- Games are also capped by `--maxPlies` to prevent endless movement cycles.
- Elo is not assigned until enough champion-versus-pool games exist for calibration.

## Next engine milestones

1. Add tactical and capture-vulnerability features.
2. Maintain a pool of historical champions to prevent overfitting to one incumbent.
3. Add iterative deepening, transposition tables, and strict time budgets.
4. Run large batches in a Rust headless engine after parity tests match the TypeScript reference engine.

# Pathagon self-play

The self-play layer treats the browser game engine as the reference rules implementation. Every match uses contract v1 metadata and records every action, capture, node count, result, and termination reason.

## Run an arena

```bash
npm run selfplay -- --mode arena --games 20 --seed 20260822
```

This alternates the tracked champion between light and dark against a seeded random baseline. Two random opening plies diversify otherwise deterministic matchups.

Use `--opponent surveyor` or `--opponent pathfinder` for a named regression arena. Pathfinder is deliberately outside the promotion pool: evaluator mutations should not be permanently blocked merely because a candidate is tested with a smaller search budget than the expert browser opponent.

## Promotion training ownership

The TypeScript promotion trainer has been retired. TypeScript remains the
browser/reference self-play and league runner; evaluator-weight promotion is
owned by the Rust trainer (`npm run rust:train`), with the Python GNN league
handling checkpoint-based experiments. Historical TypeScript champion and
league manifests remain readable for replay and arena comparison, but no new
TypeScript promotion artifacts are produced.

## Run the historical league

```bash
npm run selfplay:league -- --games 8 --seed 20260823
```

This runs a paired-color round robin over tracked champions and produces provisional within-pool ratings. These numbers compare agents in this league only; they are not human Elo ratings.

The browser exposes four opponent levels: random Coin Flip, the intentionally
naive one-ply Lunatic pattern heuristic, the novice two-ply Surveyor, and the
four-ply Pathfinder. Hover coaching is one fixed, bounded reference search;
deeper iterative search belongs in the Rust engine so touch input does not
trigger an unbounded browser refinement loop.

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

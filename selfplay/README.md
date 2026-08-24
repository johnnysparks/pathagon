# Offline self-play

[`cli.ts`](cli.ts) and [`core.ts`](core.ts) provide the TypeScript regression
arena and historical league runner. Rust and Python have separate high-volume
and neural workflows documented in [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).

## Current role

- generate deterministic local regression matches;
- compare reference/search agents;
- write replayable records for validation and archive import;
- support historical league comparisons.

The browser leaderboard does not generate matches. New leaderboard records must
come from an offline runner or an explicitly imported archive.

## Runtime files

`progress/champion.json` and `progress/league.json` are TypeScript runner
configuration/state. `progress/runs/` contains disposable generated output and
is ignored by Git. These paths are retained for compatibility while the
artifact directory migration is planned.

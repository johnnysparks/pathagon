# Q-Arbiter deep-exploration handoff

Date: 2026-08-25

## Corpus

- 3,208 games and 185,044 positions after action-sequence deduplication.
- 3,208 unique trajectories; 0 duplicates removed in this rebuild.
- 185,044 / 185,044 positions include `mcts-root-q-v1` action values.
- Train split: 2,586 games / 149,592 positions.
- Held-out split: 622 games / 35,452 positions.
- Sources combine the shipped Q-complete backfill with 128-simulation multi-model exploration and a 256-simulation Q-Arbiter wave.

The full JSONL benchmark is intentionally kept in the ignored `work/` area because it is about 1.1 GB. The authoritative local manifest is `work/benchmark-qadv-expanded-20260825/manifest.json`.

## Checkpoint

`qadv-arbiter-7x7-v0.1.0-exploration-20260825.pt`

- Architecture: residual mean message passing with `dueling-transition-qadv-v1`.
- Training: 2,000 steps, learning rate `0.0002`, symmetry augmentation enabled.
- Q target source: `mcts-root-q-v1`.
- SHA-256: `dd984097af768c46f6abbdba343ea266b9a7ab552164deb62c6b9c13ccacfd68`.

## Held-out signal

On a 4,000-position / 63-game held-out sample:

- Predicted pairwise accuracy: `0.5394`.
- Q MAE: `0.0924`.
- Target-policy pairwise accuracy: `0.7711`.
- Predicted action was target-Q-max on `9.55%` of positions.
- Relocation pairwise accuracy: `0.5549`; placement pairwise accuracy: `0.4776`.

The ranking and Q regression improved directionally over the earlier sample, but top-1 action selection is still weak. Keep Q-Arbiter as a search ranking prior rather than promoting this checkpoint as a standalone player.

## Fresh roster cross-play

The 24-game direct-selector run is in `qadv-arbiter-7x7-v0.1.0-exploration-standings.json`:

- Q-Arbiter: 3-21-0 overall.
- Pathfinder: 0-4 for Q-Arbiter.
- Surveyor: 0-4.
- Re-evaluated GNN: 0-4.
- Re-evaluated CNN: 0-4.
- Lunatic: 0-4.
- Coin Flip: 3-1.

The deterministic Pathfinder, Surveyor, and Lunatic matches repeat by color. The neural and Coin Flip matches provide more varied trajectories; do not treat repeated deterministic lines as independent learning evidence.

## Reproduce

The training invocation was:

```text
python -m research.gnn.train qadv --data work/benchmark-qadv-expanded-20260825/all.jsonl --resume research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-qadv-backfill-20260825/qadv-arbiter-7x7-v0.1.0-backfilled.pt --out research/runs/gnn/benchmark-7x7/generated/batch-20260825-qadv-deep-exploration-20260825/qadv-arbiter-7x7-v0.1.0-exploration-20260825.pt --size 7 --steps 2000 --learning-rate 0.0002 --heldout-fraction 0.2 --seed 2026086600 --device auto --agent-id qadv-arbiter-7x7-v0.1.0 --agent-name 'The Q-Arbiter'
```

The local archive artifact is ready, but the Sites archive API still needs its shell credential (`PATHAGON_ARCHIVE_TOKEN`); the signed-in browser page exposes no supported import control. Upload authorization is already granted, so this is an authentication handoff rather than a confirmation gate.

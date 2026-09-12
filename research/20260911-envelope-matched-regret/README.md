# Envelope-matched relative regret

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

Relabel the same audited quiet roots with depth-4/beam-256/32k search, matching
the arena envelope that the 8k teacher failed to represent. Continuation traces
and the game-disjoint split were preserved; only action scores and their
ordering changed.

## Outcome

The relabeler performed 2,051 native action analyses. Action-head-only training
raised heldout top-1 from 7.3% to 52.1% and pairwise agreement from 46.8% to
71.7%; value MAE was 0.3431 and unchanged by training. In a fresh 48-game 32k
arena the candidate scored 42.7% versus 45.8% for control. All six logs passed
replay audits. Budget-matched labels improved offline fit but did not transfer
to whole-game strength.

## Decision

Retain the relabeler as a teacher-quality tool. Do not promote the checkpoint;
the next useful change is the runtime integration, not another optimizer sweep.

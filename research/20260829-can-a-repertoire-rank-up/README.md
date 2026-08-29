# 20260829 Can a repertoire rank up?

Status: running

## Idea

Aggregate the five most recent research results into a conservative new
opponent rank. Instead of another global evaluator mutation, always-on sorter,
or larger per-turn search budget, learn a compact opening repertoire from the
durable game corpus and new stronger-teacher labels. The repertoire may choose
only a legal, tactical-safe move on a covered early position; every other
position falls back exactly to the promoted v0.5 Rust Pathfinder.

This tests whether training data is most useful as narrow, auditable memory
rather than as a global policy. The candidate must earn a stable identity in a
source-disjoint arena before it becomes a supported opponent.

## Starting point

The five inputs are the completed 20260829 product-budget, v0.5 evolution,
root-regret, curriculum, and gated-sorter paths. Together they show that:

- depth 6 / 100,000 nodes was responsive but scored 49.9% against v0.5;
- local six-weight evolution regressed to 44.2% in its untouched arena;
- a small exhausted root-regret teacher produced no meaningful held-out gain;
- mixed phase roots increased coverage without increasing strength; and
- the compact learned sorter reduced held-out teacher agreement at every gate.

The incumbent remains `pathfinder-v0.5.0-trained-evaluator`, using the
tactical-safe root filter at depth 4 / 2,000 nodes / beam 8. The durable v1
corpus contains 38,547 unique games and 128,953 provenance observations. Its
historical outcomes are useful as priors and coverage evidence, but they mix
many agent generations, so new labels from a frozen stronger teacher remain
authoritative for move selection.

## What happened

Running. The protocol freezes corpus-derived opening statistics, teacher-label
seeds, selection seeds, and final arena seeds before promotion. Generated
games, labels, reports, and logs stay under `workspace/` until a compact,
versioned repertoire qualifies for `data/`.

## Data and artifacts

The experiment will keep one-time aggregation and labeling code beside this
narrative. Bulk labels and arenas belong in ignored `workspace/`. A promoted
repertoire must live in a strict versioned path under `data/`, include source
and teacher provenance, and be consumed by the Rust engine with focused tests.

## Project impact

None yet. No opponent identity or runtime behavior is supported until the
candidate passes legality, tactical, coverage, latency, and whole-game gates.

## Next decision

Promote only if the frozen candidate improves on v0.5 in an untouched paired
arena, remains positive in both colors, passes all existing regression suites,
and adds negligible browser latency. Otherwise retain the narrative and retire
the candidate without changing the supported ladder.

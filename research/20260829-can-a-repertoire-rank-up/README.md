# 20260829 Can a repertoire rank up?

Status: inconclusive

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

The protocol froze corpus-derived opening statistics, teacher-label seeds,
selection seeds, and final arena seeds before promotion. Aggregation covered
38,322 games and produced 238,223 raw state/action records, consolidated to
222,753 canonical records. The strongest tactical-safe teacher book covered
609 roots, but only 106 roots reached the selected depth-6 quality threshold.

The resulting repertoire was tested in paired, color-balanced arenas. The
opening-only candidate scored 109-122-9 in 240 games and 199-191-10 in the
untouched 400-game arena (52.0% game points); the wider ply-2-8 candidate
scored 111-122-7 in its 240-game screen. Coverage and legality were sound, but
whole-game strength did not clear the promotion threshold. No repertoire was
promoted.

## Data and artifacts

The experiment keeps one-time aggregation and labeling code beside this
narrative. Bulk labels and arenas remain in ignored `workspace/`; no strict
versioned repertoire was added under `data/`, and no runtime book remains in
the supported Rust path.

## Project impact

No opponent identity or runtime behavior was promoted. The negative result is
retained as evidence against repeating a sparse opening-book approach without
stronger contextual features.

## Next decision

Retain the narrative and retire the candidate without changing the supported
ladder. The next effort should spend the larger teacher budget on richer state
representations rather than expanding this sparse book.

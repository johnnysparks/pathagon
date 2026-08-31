# 20260829 Super-deep contextual evaluator

Status: inconclusive

## Idea

Train a phase-aware evaluator from a materially larger, source-disjoint set of
Pathagon roots labeled by super-deep Pathfinder search. The evaluator will
learn separate six-feature weight vectors for opening, placement, movement,
and late-game contexts, while the bounded Rust search remains the decision
authority. The target profile is depth 7 / 1,000,000 nodes / beam 32, with a
wall-clock policy of roughly 1–3 seconds per turn.

This is the follow-up most directly supported by the last five studies: their
sorter, curriculum, and root-regret paths suggest that signal quality and
context matter, while their 32–144-root pilots do not test this at scale.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator` with weights
`241,112,887,40,154,74` and the tactical-safe root filter. The durable corpus
contains 38,547 unique games and 128,953 observations. The preceding repertoire
path replayed 38,322 games and generated 268,254 early-position observations,
but its best lookup candidate scored only 52.0% in its untouched 400-game
arena. No temporary repertoire runtime code remains supported.

## Protocol

1. Select 480 source-disjoint 7×7 canonical games and one root per game across
   opening, placement, movement, and late-game phases. Split by source game
   before labeling into training and held-out partitions.
2. Label every root with tactical-safe depth 7 / 1,000,000-node / beam-32
   Pathfinder. Persist legal actions, six unit evaluator features, teacher
   action, completed depth, node use, and exhaustion state.
3. Fit phase-specific six-weight vectors on training roots with deterministic
   seeds, shrinkage toward v0.5, and a held-out action-ranking gate. Do not use
   final arena seeds for selection.
4. Run paired 120-game screens, then an untouched 400-game arena against v0.5
   at the same 100k+ node / 1–3 second policy. The decisive arena uses the
   full 2.8-second cap; earlier 1.5-second batches are low-budget ablations.
   Require positive margins in both colors, no tactical regressions, legal
   replay, and measured latency.

## Promotion criteria

Promotion requires every gate: at least 53% game points in the 120-game screen,
at least 55% in the untouched 400-game arena, positive win-loss margin in each
color, no immediate-win/forced-block/human regression, all games replaying
legally, and candidate timing within the 1–3 second product envelope. A failed
gate leaves the candidate research-only.

## What happened

The 480-root set was frozen before labeling, with one source game per root and
388 train / 92 held-out roots. The original 500k-node labels reached depth 6+
on 312 roots; raising the teacher budget to 1,000,000 nodes improved that to
417 roots (338 train / 79 held-out), with depth counts 1/62/295/122 at depths
4/5/6/7. All higher-budget labels use the same depth-7, beam-32,
tactical-safe teacher identity.

The first contextual fit was unstable: its 500k-node, 1.5-second ablation
scored 9-14-7 over 30 games. At the full 2.8-second cap, that same fit scored
13-4-3 over 20 games, showing why the cap should not be used as a proxy for
label quality. Retraining on the 1M-node labels produced a low-regularization
fit with held-out teacher top-1 33/79 (41.8%) versus v0.5's 32/79 (40.5%), but
its 1M-node arena was only 7-10-3 over 20 games and failed the Dark-color
margin. Conservative and intermediate variants scored 4-4-2 and 3-5-2 in
additional 10-game full-cap batches.

The strongest controlled blend learned only opening/placement weights and kept
the incumbent for movement/late-game. It scored 11-8-1 over 20 games at depth
7 / 1,000,000 nodes / beam 32 with a 2.8-second cap, but the color split was
7-2-1 as Light and 4-6 as Dark. It therefore fails the required positive
margin in both colors. The arena logs averaged roughly 364k node visits per
ply under the hard cap; all sampled games replayed legally.

The planned 120-game screen was stopped at this point under the predeclared
two-color gate: a negative Dark margin after 20 games means the candidate
cannot qualify for the untouched 400-game arena. This preserves compute for a
new representation rather than spending another 100 games on a known failed
candidate.

No candidate cleared the promotion gate. The 1M-node labels are retained as
the best current supervision, while the six-weight contextual model and all
arena variants remain research-only.

The follow-up [`20260829-turn-balanced-contextual-evaluator/`](../20260829-turn-balanced-contextual-evaluator/)
corrected a sampling flaw (the original roots were mostly Light-to-move) and
scaled to 1,920 balanced roots. Separate Light/Dark fits still failed the
held-out pre-gate and a depth-7-only screen was 9-10-1, so the avenue was
retired without a 100-game promotion arena.

## Data and artifacts

The path generated 480 frozen roots, 480 500k-node targets, 480 1M-node
targets, trainer reports, and multiple color-balanced arena batches. The
turn-balanced follow-up generated 1,920 additional roots and 1,920 1M-node
targets. Roots, targets, reports, generated weight files, and game exports are
ignored under `workspace/`; the harnesses and narratives are the durable
research artifacts. Nothing was promoted under `data/`.

## Project impact

The Rust engine now contains a research-only phase-conditioned Agent variant
and replay-audit test, but no supported opponent ID, browser entry, or deployed
weights changed. The contextual hook is intentionally not in the supported
roster because the Dark-color gate failed.

## Next decision

Retain the 1M-node labels and negative evidence. The next attempt should add
state features or a policy/value model (and likely condition on player/turn),
then repeat the same source-disjoint, full-cap two-color arena. Do not repeat
the six-weight phase sweep without changing its representation.

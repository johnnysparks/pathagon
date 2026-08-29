# 20260829 Can v0.5 evolve further?

Status: completed

## Idea

Hypothesis: the promoted v0.5 evaluator is already in a better region of the
weight landscape than the handcrafted evaluator, so local evolution starting
from v0.5 can find another fixed-budget improvement that the prior restart from
handcrafted weights could not reach.

This is the highest-priority learning path after the latency-calibrated search
budget is selected because it changes the smallest possible learning variable:
the learner's starting point. Rules, tactical filtering, and the selected
product envelope remain frozen within the experiment.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator` with weights
`path=241, material=112, capture=887, structure=40, threat=154, edge=74`.
The frozen control is `pathfinder-v0.4.0-tactical-filter`. Both use depth 4,
2,000 nodes, beam 8, paired colors, two randomized opening plies, and the
tactical-safe root filter.

The sibling compute-budget path is expected to select a larger product profile.
That profile should be applied identically to incumbent and candidate for the
primary arena; the 2,000-node profile remains a secondary historical cross-check.

The immediate parent path is
[`../20260828-pathfinder-boundary-evolution/`](../20260828-pathfinder-boundary-evolution/).
That run started from the handcrafted evaluator, tested 24 candidates, and
promoted none. Its result argues for changing the initialization rather than
merely adding generations.

## Proposal

1. Extend the Rust training entry point to load an initial evaluator from the
   supported opponent manifest or explicit initial weights. Record the exact
   starting identity and weights in every report.
2. After the product search envelope is selected, run at least three
   deterministic seeds around v0.5. Use a coarse mutation
   pass followed by a narrower pass around any independently positive candidate.
3. Keep training openings, held-out openings, and the final arena seeds
   disjoint. Do not tune after observing the final arena.
4. Keep incumbent and candidate search configurations identical at the selected
   latency-qualified product profile. Only evaluator weights may vary. Repeat a
   smaller screen at 2,000 nodes to expose budget-specific overfitting.
5. Audit representative wins, losses, long games, captures, repetitions, and
   both colors before considering promotion.

## Promotion criteria

All gates must pass:

- A final 400-game paired arena against v0.5 at the selected product envelope
  reaches at least 55% game points, has a positive win-loss margin in each
  color, and uses untouched final seeds.
- At the historical 2,000-node envelope, a separate 120-game paired screen
  against v0.5 remains positive and a 120-game v0.4 control screen reaches at
  least 55% game points, exposing budget-specific exploitation or regression.
- Every game replays legally; tactical and human-derived regression fixtures
  remain fully passing; node and browser latency limits do not regress.
- The exact weights, stable agent identity, protocol, representative game
  audit, and rerun command are durable and reproducible.

Anything weaker is research evidence, not a new supported opponent.

## What happened

The Rust training entry point was made initializer-aware. `pathagon-train` now
accepts `--initial-manifest` for a supported opponent or prior champion output,
and `--initial-weights` plus `--initial-id` for explicit weights. The loader
preserves the exact starting identity, generation, and weights in `report.json`.
The self-play CLI also gained `--candidate-id` so research weights cannot be
silently recorded under a supported opponent identity. Focused initializer
tests passed, and the command-line smoke test loaded
`pathfinder-v0.5.0-trained-evaluator` with weights
`241,112,887,40,154,74`.

The sibling latency path had not yet selected a fully validated product
profile, so depth 7 / 500,000 nodes / beam 32 was used as a provisional
working profile because its earlier browser probe was approximately 1.40 s on
the reference position. A bounded native calibration with two candidates, one
training pair, one held-out pair, 120-ply games, and four total games ran for
18 minutes without producing a report and was stopped. This profile is not
promotion evidence and remains a separate product-envelope issue.

The historical 2,000-node screen then ran with depth 4, beam 8, the tactical
root filter, two randomized opening plies, 120-ply maximum games, and paired
colors. Each coarse run used two generations, population 4, two training pairs,
six held-out pairs, mutation scale 200 per mille, and a distinct seed:

| seed | best held-out candidate | held-out result | promoted |
| ---: | --- | ---: | ---: |
| 2026082901 | `244-121-798-44-126-61` | 8–4–0 (666/1000) | no |
| 2026082902 | `226-122-1036-37-160-86` | 6–5–1 (541/1000) | no |
| 2026082903 | `211-111-895-44-156-77` | 5–7–0 (416/1000) | no |

The seed-1 positive screen was independently narrowed with mutation scale 75
per mille. The narrow champion was
`rust-evo-g2-c3-238-112-798-45-126-65`; it scored 13–10–1 (562/1000) in its
24-game held-out screen and was frozen for the untouched arena.

The final 120-game historical arena rejected it: against v0.5 it scored
52–66–2, or 44.2% game points, with a negative margin in both colors (28–30–2
as light and 24–36 as dark). The protocol control reproduced the known v0.5
advantage over v0.4 at 69–50–1, or 57.9% game points, with positive margins in
both colors (37–23 as light and 32–27–1 as dark). This isolates the failure to
the evolved candidate rather than the arena protocol.

The candidate and control corpora each contain 120 games with disjoint seeds
and unique two-ply openings. All 240 games replayed legally through the shared
rules adapter, all 240 recorded outcomes matched replay, all games contained
captures, and the candidate corpus included 118 path terminations, one repetition draw,
and one max-ply draw. The candidate used 4,698 plies and 6,070,387 search
nodes; the control used 4,778 plies and 6,313,094 nodes.

The required 400-game product-envelope arena was not run after the candidate
failed the independent historical screen; spending that cost could not rescue
the failed promotion gate. The evidence therefore rejects a v0.6 evaluator
promotion from this path while leaving the product-envelope selection for its
own research path.

## Data and artifacts

Preserve:

- Generic Rust support for loading a declared initial evaluator, with tests.
- The final aggregate reports, seed partitions, candidate weights, and concise
  narrative for any independently screened candidate.
- Only replay fixtures with durable regression value, promoted under a strict
  versioned location in `data/`.
- A promoted agent manifest and Rust implementation only if every gate passes.

Discard or keep ignored in `workspace/`:

- Rejected population members, optimizer/evolution state, repeated full replay
  exports, verbose logs, and ad hoc summaries.
- One-off candidate manifests and duplicate game archives that do not become
  reusable fixtures.

This path retained the final reports, seed partitions, candidate weights, and
compact arena corpora in its ignored `workspace/` for local reproducibility.
No candidate agent, manifest, replay fixture, or corpus was promoted under
`pathagon/opponents/` or `data/`.

## Project impact

The generic initializer capability was promoted into
`pathagon/engine-rs/src/training.rs` and `pathagon/engine-rs/src/train_main.rs`,
and the explicit candidate identity boundary was added to
`pathagon/engine-rs/src/main.rs`. No runtime behavior or supported opponent was
promoted. The research hypothesis was not supported: a v0.5-seeded local
mutation that looked positive on a small held-out screen regressed materially
on the untouched 120-game arena. This points to screen noise/overfitting, not
an actionable new evaluator blind spot.

## Kick-off prompt

> Execute the research brief in
> `research/20260829-can-v05-evolve-further/README.md` through a defensible
> promotion or rejection decision. Start from the promoted v0.5 evaluator,
> preserve the v0.4 control, use the selected latency-qualified product envelope
> identically for incumbent and candidate, retain depth-4 / 2,000-node / beam-8
> as a historical cross-check, and keep final arena seeds untouched until the
> candidate is frozen.
> You have explicit liberty to rename identities or files and to make
> appropriate edits outside this research folder—including under `pathagon/`,
> `apps/`, `data/`, `docs/`, and `scripts/`—when the evidence and implementation
> justify it. Update all references, tests, manifests, generated browser assets,
> and documentation affected by those edits. Keep disposable outputs under this
> path's ignored `workspace/`; do not promote a candidate unless every criterion
> in this README passes.

## Next decision

Reject and retire this local linear-weight evolution attempt. Do not create a
v0.6 opponent from the frozen candidate. Continue with root-regret supervision
or curriculum experiments, and complete the separate latency/product-envelope
path before using a multi-second profile as a learning target.

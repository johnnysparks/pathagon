# 20260828 QAdv root-sorter held-out evaluation

Status: `completed · not promoted`

## Idea

The recent QAdv ladder mostly evaluated QAdv as a direct move selector. That
does not establish whether the separate Rust QAdv root-sorter generalizes when
it only reorders Pathfinder's candidate moves. This path isolates that role and
measures it in two independent ways: held-out action ranking and paired
whole-game play.

## Starting point

The frozen search authority is the Rust Pathfinder with the tactical-safe root
filter. The candidate is the QAdv ONNX root-sorter, which supplies action-value
ordering while Pathfinder's legal move generation, evaluator, alpha-beta
search, and root limit remain authoritative. Existing QAdv checkpoints are the
0%, 25%, and 50% seeded curriculum variants under the prior experiment's
ignored workspace; the exact QAdv export and target metadata will be recorded
with each run here.

## Protocol

For ranking, split each QAdv target corpus by complete game seed and report
target-Q top-1, pairwise agreement, selected-action rank, and phase/seed
coverage. The checkpoints do not preserve their original training split seed,
so this path uses a reproducible post-hoc split with seed `20260828`; it is a
held-out screen, not a claim about the exact original training partition.

For whole-game play, use the native Rust runner with 7x7/14-reserve rules,
paired colors, two randomized opening plies, a 120-ply cap, and the same
depth-4/beam-8/2,000-node search envelope used by the prior sorter study.
Compare QAdv root sorting with the frozen Pathfinder control on identical seed
sets. Validate every replay and inspect representative tactical, capture,
relocation, win, and draw records.

## What happened

The three exported QAdv models loaded successfully in the inference-enabled
Rust binary and completed the native arena. The ranking screen covered 3,501
held-out positions from 61 game seeds:

| curriculum | positions / games | predicted Q pairwise | predicted target-Q top-1 | predicted target-Q rank | target-policy pairwise |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 1,320 / 18 | 57.94% | 24.24% | 18.47 | 53.51% |
| 25% | 979 / 20 | 56.77% | 17.26% | 17.35 | 54.68% |
| 50% | 1,202 / 23 | 58.18% | 23.63% | 17.59 | 54.91% |

The paired native arena used the Rust tactical-filter Pathfinder as the
opponent, with paired colors, two randomized opening plies, depth 4, beam 8,
2,000 nodes, root limit 16, and QAdv top-k 2:

| model / screen | games | QAdv wins | Pathfinder wins | draws |
| --- | ---: | ---: | ---: | ---: |
| 0% | 120 | 51 | 69 | 0 |
| 25% | 120 | 65 | 54 | 1 |
| 25% repeat | 120 | 64 | 55 | 1 |
| 50% | 120 | 46 | 73 | 1 |

The first matched 25% screen was positive at 65–54–1, but the disjoint repeat
was negative at 43–77–0; combined, the 25% model was 108–131–1. The primary
three-variant screen was negative overall at 162–196–2, and adding the
disjoint repeat gives 205–273–2. A focused disjoint 25% ablation went 32–28
with top-k 8 and 24–36 when QAdv scored all legal root actions. The narrow
top-k 2 screen is therefore the least-bad configuration, not evidence that
the sorter generalizes.

The native archives passed structural review: all 600 retained games had valid
terminal records; the only non-path endings were one threefold draw and one
max-ply draw. Export hashes and complete replays remain in the ignored
`workspace/` for auditability. Direct-selector results from the prior ladder
remain background context, not evidence for this sorter role.

## Data and artifacts

Generated reports, exports, replays, and logs belong in this path's ignored
`workspace/`. No repeated replay archive or implementation-shaped tensor will
be promoted. The retained QAdv export hashes are:

- 0%: `sha256:64ad45b3dcd11f518647bc7de77f0ace37d061136e7e4c2fb4f81905b859149c`
- 25%: `sha256:b32fbfbabc1cc4888a2f63c2a0fe9368557beca2791bc98555f760a34b186fe7`
- 50%: `sha256:605273dbbdef763eca234064fc6d2dbad4a09000a3c6f513ff0c8f9e1c9eebd6`

Durable evidence is summarized here and linked to existing canonical corpus
sources when applicable.

## Project impact

The stale `research.gnn` imports in the QAdv ranking evaluator were repaired so
the documented held-out command runs against the current lab package layout.
No production Rust, browser, registry, or canonical-data behavior changed.
The QAdv models are not promoted, and no deployment was attempted.

## Failures and limits

- The original training split seed was omitted from the checkpoint metadata;
  future training must record it alongside the held-out counts.
- The first repeat and ablation attempt reused overlapping contiguous game
  seeds. Those duplicate outputs were deleted; the retained repeat and
  ablations use disjoint seeds `2026083000`, `2026083200`, and `2026083260`.
- The 25% win is not stable across curriculum variants, and the broader root
  pools underperformed the narrow pool.
- These are 7x7 native screens against one frozen Pathfinder authority. A
  promotion would still require a longer, independently seeded ladder and
  representative tactical review after the provenance gap is fixed.

## Next decision

Retire this QAdv sorter candidate from promotion. Retain the evaluation path,
the exporter, and the ignored artifacts as research infrastructure; retrain
only after restoring exact split provenance and revising the target or pool
design.

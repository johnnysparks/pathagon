# 20260829 Next-generation action-transition opponent

Status: completed — scorer packaged, selective deep labels retained, production promotion deferred

## Idea

Scale the strongest prior signal rather than repeat the failed board-only
policy/value fit. A larger source-disjoint root set, deeper teacher labels,
and explicit placement-vs-relocation afterstate features should let the compact
nonlinear scorer learn a more stable root ordering while Pathfinder remains the
rules and search authority.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator`. The prior
`20260829-action-transition-policy` candidate scored 320–210–110 from its own
perspective across 640 untouched, paired games at depth 7 / 1,000,000 nodes /
2.8 seconds, with all 29,141 plies replaying legally. Its 1,920 source games
are excluded from this generation. Board-only GNN/QAdv variants failed their
held-out gate and are not receiving more arena budget in this path.

## Protocol

Select thousands of new 7×7 roots from the canonical corpus, excluding every
source game used by the prior turn-balanced set. Balance Light/Dark-to-move
roots and keep the source game as the split unit. The first lane contains
6,000 roots; a second lane adds 4,000 more from the remaining source games so
the generation can use all available local search capacity. Label each root
with the same tactical-safe Pathfinder teacher at depth 7 / 1,000,000 nodes / beam 32;
raise the budget or depth only as a targeted ablation where it changes label
quality rather than merely increasing volume. Train explicit typed action
features with multiple deterministic seeds and compare against the prior
checkpoint, the linear baseline, and a virtual off-board encoding.

All candidates must preserve exact legal action generation, the tactical-safe
root filter, iterative-deepening search, and the 1–3 second product envelope.
Selection proceeds through source-disjoint held-out action ranking, legality and
tactical fixtures, latency parity, then larger untouched color-balanced arenas.

## What happened

Root selection and labeling completed for 10,000 source-disjoint roots (5,000
Light-to-move / 5,000 Dark-to-move), excluding the prior 1,920-root source
set. The primary lane has 6,000 roots and the expansion lane 4,000. Every
root received the depth-7 / 1M-node / beam-32 teacher label; a balanced 256-
root ablation also received depth-8 / 2M-node / beam-32 labels. The deeper
teacher selected a different action on 51/256 roots (80.1% agreement), which
is large enough to justify targeted deep labels but not a wholesale deep-label
replacement yet.

Four compact explicit action-transition variants were trained. The broad
cross-entropy fit (`workspace/model-v3-xent/`) was the best held-out model:
594/2,038 exact teacher actions (29.15%) and 949/2,038 top-three actions
(46.57%), versus the incumbent baseline's 423/2,038 (20.76%) and 626/2,038
(30.72%). It selected no unsafe actions. Ranking-loss variants were weaker on
this split; the depth-8-only fit reached 230/366 (62.84%) on its narrow deep
subset, so it remains an ablation rather than a standalone model.
The finalist artifact SHA-256 is
`4f08a5a68057051e99c469aaf4a6e839885ebdcb167e6b82b076836c0b24b7f4`.

The deeper-label ablation was then made selective. Comparing the frozen
depth-7 and depth-8 labels produced 51 disagreement roots (19 Light, 32 Dark)
out of 256, so `workspace/targets-depth8-disagreement.jsonl` contains only
those 51 depth-8 replacements and
`workspace/targets-10000-selective-depth8.jsonl` applies them to the full
10,000-row view. All 51 roots are heldout by design; this is a teacher-quality
diagnostic and leakage-safe evaluation view, not a claim of additional training
signal. The selective labels passed the target audit with 51/51 unique legal
rows, depth-8/2M-node/beam-32 metadata, and completed-depth distribution
5:3, 6:18, 7:28, 8:2.

The finalist then played 800 untouched, paired games against
`pathfinder-v0.5.0-trained-evaluator` at depth 7 / 1M nodes / beam 32 / 2.8s,
with alternating colours. It scored 427 wins, 335 losses, and 38 draws
(55.75% points; Light 55.63%, Dark 55.88%; 95% Wilson interval 52.29–59.16%).
All 800 games replayed legally with zero capture mismatches. Candidate search
telemetry was close to the incumbent (318,583 vs 314,194 mean nodes/move;
5.060 vs 5.104 mean completed depth), so the strength gain did not come from a
material search-budget increase. The arena took roughly six hours wall time
on eight local workers, making it the dominant cost of this generation.

## Data and artifacts

This path stores one-time outputs in `workspace/`, including the frozen roots,
10,000 labels, the selective 51-row depth-8 view, four checkpoints, depth
comparison, arena JSONL, replay audit, and telemetry summary. The reusable
model is promoted as
[`data/models/pathfinder-action-transition-v3-xent/`](../../data/models/pathfinder-action-transition-v3-xent/)
with a manifest and SHA-256 identity; the browser copy is
`apps/web/public/models/pathfinder-action-transition-v3-xent.json`. The
generated Rust/WASM bundle exposes the same model through
`PathagonTransitionPolicyModel` and the web adapter's
`loadTransitionPolicyEngine()`.

## Project impact

The existing Rust research hook and rules-authoritative tactical-safe root
ordering were reused. Default and WASM-feature Rust tests (57 each), inference
feature tests (63), release all-target tests, the data-policy check, the web
build, all 42 browser tests, and 24-case cross-runtime parity pass. A direct
WASM smoke test loaded the versioned model, ranked seven legal roots, and
returned a legal searched action. The supported v0.5 opponent and browser
roster remain unchanged; the new model is packaged for inspection and opt-in
experiments, not promoted as the default opponent.

## Next decision

The strength, legality, tactical, latency, packaging, and browser-regression
gates passed for the research candidate. Promotion is deferred because the
selective deep rows are all heldout and the model has not yet been tested as a
rostered browser opponent. Keep v0.5 as the supported opponent. For the next
generation, first label a small source-disjoint training pool at depth 8, keep
only teacher disagreements, retrain without contaminating evaluation, and
measure whether the extra labels move heldout top-1/top-3 or arena strength;
then scale the winning data/search lever rather than paying for deeper labels
uniformly.

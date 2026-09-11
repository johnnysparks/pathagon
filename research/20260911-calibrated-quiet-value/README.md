# Calibrated quiet-position regret/value intuition

Status: inconclusive — completed on 2026-09-11; no promotion

## Idea

The preceding quiet-regret run improved action ranking but degraded its scalar
value head and had only 23 heldout roots. This follow-up tests the missing
calibration step directly. It enlarges the game-disjoint holdout, forces equal
neutral and decisive targets, records short/mid/long continuation values, blends
those values with the replay outcome into an explicit state target, and runs a
preregistered multi-seed arena.

## Starting point

The rules, replay parsing, legality, and search remain in the Rust engine. The
learner resumes from the prior QAdv checkpoint
`research/20260901-strong-teacher-10k-games/workspace/retraining-mixed-final/gnn-qadv-replay.pt`.
The architecture is the residual mean message-passing GNN with hidden size 64,
eight layers, dynamic place/relocate action features, and the dueling QAdv head.
The deployment control is the frozen Rust
`pathfinder-v0.4.0-tactical-filter` agent.

## What happened

The Rust exporter scanned a broader deterministic corpus sample and selected
roots by partition, phase, turn, and target class. Quiet roots have at least two
legal actions, no immediate win, no multi-capture, and no forced immediate loss.
The teacher scores up to 12 legal actions at depth 4 / 8,000 nodes and requires a
100-point best-versus-second margin. Each action then follows a tactical-filter
continuation to plies 4, 8, and 12. The blended continuation value is
`0.50*v4 + 0.30*v8 + 0.20*v12`; the calibrated state target is
`0.75*expectedContinuationValue + 0.25*sourceOutcome`, clipped to [-1, 1].

The final target set contains 192 roots from 100 source games: 96 neutral and
96 decisive, with exactly 48 roots in each target class for train and heldout.
The partition is 96/96 roots from 52/48 disjoint games. All phases are covered:
opening 53, placement 48, movement 45, and late 46; turns are light 99 and dark
93. There are 183 roots with distinct continuation values. Opponent profiles
include heuristic, search, neural, random, and other agents.

The joint learner ran 4,000 AdamW steps with symmetry augmentation. On the 96
heldout roots, teacher top-1 action agreement increased from 6.3% to 41.7%, and
pairwise agreement from 53.1% to 64.3%. Value calibration failed its registered
gate: value MAE increased from 0.349 to 0.380, RMSE from 0.478 to 0.502, and the
outcome-top metric was 23/90 (25.6%). A value-only pass on the old checkpoint
reduced MAE only to 0.344 and left action ranking unchanged; calibrating the
joint checkpoint worsened MAE to 0.388. The action and board-value signals are
still not learning the same abstraction.

The protocol in [`protocol.json`](protocol.json) was committed before the arena.
It fixes three seeds, 16 games per seed, alternating candidate colors, four
random opening plies, the evaluator weights, and 2k/8k/32k node budgets. The
candidate is the calibrated QAdv root sorter; the control is the same tactical
filter without learned ordering. Aggregate candidate game points are:

| Nodes | Candidate | Control | Difference |
| ---: | ---: | ---: | ---: |
| 2,000 | 18–22–8 (45.8%) | 18–21–9 (46.9%) | −1.0 pp |
| 8,000 | 19–26–3 (42.7%) | 22–20–6 (52.1%) | −9.4 pp |
| 32,000 | 25–17–6 (58.3%) | 23–25–0 (47.9%) | +10.4 pp |

These are 48 games per cell across three independent seeds. The high-budget
point is favorable, but the lower-budget points regress and the curve is not
monotonic. The registered strength gate therefore fails, as does the value gate.
All 18 arena files passed native replay auditing: 288 games, 13,271 plies,
2,380 captures, no duplicate sequences or illegal transitions.

The durable aggregate is [`results.json`](results.json). Full targets,
checkpoints, ONNX, arena games, audits, and run logs remain in the ignored
`workspace/` directory.

## Data and artifacts

The Rust exporter, calibrated trainer, analyzer, registered arena runner,
protocol, lockfile, and this narrative remain in Git. Generated artifacts are
ignored under `workspace/`. The protocol SHA-256 is
`acda376e8e2a7b2f82811faf18eb7c8a1f038c03c646e27d1ae8723c78cd56a5`; the final
target, checkpoint, and ONNX hashes are
`15d7ef17e5fa9b5833df45db5243767dc4379839b212a3085a69888d1c099ab0`,
`858bfda3a084d92ed9766e09e65697aceb6ad64dc4f1872c79fc93983d666f4c`, and
`6f7c45c25cffdb4062c3a41364fea2ac407d1b9e91bd1b3469c9fa1e35bf9fde`.
Nothing was promoted into `data/` or the user-facing opponent roster.

## Project impact

This path supplies reusable infrastructure for class-balanced quiet targets,
calibrated multi-horizon continuations, game-disjoint evaluation, and
preregistered multi-seed screens. It confirms the model runs natively through
Rust ONNX QAdv sorting, but it does not establish a stronger opponent. The v4
transition-policy default and Rust/WASM integration remain unchanged.

## Hiccups and limits

The first 192-root attempt labeled 16× too many roots before balancing and was
stopped after live profiling. The corrected run labels 4× the requested roots,
which still leaves a sizeable CPU cost but preserves the class and partition
quotas. Value targets are heuristic leaf evaluations except when a continuation
reaches a terminal result; they are calibrated targets, not game-theoretic
values. The source outcome comes from prior replay agents, so it is useful as a
weak long-horizon anchor but can encode their bias.

The arena is preregistered and multi-seed, but 48 games per budget is still a
screen rather than a rating. The 32k gain does not survive at 2k or 8k. The
heldout roots are disjoint by game key within this export, not proven disjoint
from every historical training archive. The value-only ablation's small MAE
change is not evidence of reliable calibration.

## Next decision

Retire this candidate as a negative promotion result. Keep the exporter, trainer,
and protocol as infrastructure. A future attempt should calibrate the value head
on a much larger, independently sampled set before joint action training, include
explicit movement roots in every heldout fold, and use more seeds or games per
budget. Do not promote a model until label quality, action ranking, board-value
calibration, and fixed-budget strength all pass together.

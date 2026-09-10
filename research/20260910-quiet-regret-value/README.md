# Quiet-position regret/value intuition

Status: inconclusive — completed on 2026-09-10; no promotion

## Idea

The earlier learning runs mostly copied the action chosen by a stronger search.
That teaches the network to imitate a trajectory, but it does not teach what a
quiet move is worth when several legal actions are plausible. This path changes
the supervision: collect replay positions after the opening, remove immediate
wins and forced losses, use an independent bounded search to score a candidate
action set, and attach short continuation outcomes. Train the existing
action-conditioned QAdv GNN on those soft values and test the resulting root
ordering under the low-budget Pathfinder envelope.

## Starting point

The rules, replay parsing, legality, and search remain in the Rust engine. The
model starts from the prior QAdv checkpoint
`research/20260901-strong-teacher-10k-games/workspace/retraining-mixed-final/gnn-qadv-replay.pt`.
The checkpoint uses the existing residual mean message-passing GNN (hidden 64,
eight layers) with the dynamic place/relocate action head and dueling QAdv
head. The learned sorter is evaluated by the native Rust
`pathagon-selfplay` binary; the opponent is the frozen
`pathfinder-v0.4.0-tactical-filter` control.

## What happened

The Rust exporter scanned the 7×7/14-reserve unified corpus with a deterministic
game-key sample, stratified reservoirs, and phase/turn round-robin selection.
Each retained root passed the quiet-state filter: at least two legal actions,
no immediate root win, no multi-capture, and no position where every action
allows an immediate reply win. Teacher labels are independent depth-4 searches
with an 8,000-node ceiling over up to 12 legal candidates. Roots need at least
100 points between the best and second-best teacher actions. Each candidate also
gets a six-ply continuation from depth-2 tactical-filter searches. Soft action
probabilities use a temperature of 0.15; teacher scores are normalized before
training.

The final target set has 96 roots from 29 source games. It covers opening (18),
placement (29), movement (15), and late (34) phases; light/dark turns are 44/52;
and the game-key split is 73 train roots from 23 games versus 23 heldout roots
from six disjoint games. Forty-four roots have distinct continuation outcomes.
The most common source profiles are heuristic-vs-search (29),
heuristic-vs-neural (22), and heuristic-vs-other (21), with neural-vs-neural,
neural-vs-heuristic, neural-vs-other, and heuristic-vs-random also present.

Training resumed from the prior checkpoint for 3,000 AdamW steps with symmetry
augmentation. On the 23 heldout roots, action ranking improved: teacher top-1
went from 0/23 to 10/23 (43.5%), and pairwise agreement from 45.8% to 66.5%.
The scalar continuation value moved the other way (MAE 0.056 to 0.205), and the
outcome-top metric was only 1/4 roots after training. This separates “can fit
the action labels” from “has learned a reliable board value.”

The model was exported to ONNX and loaded by Rust. The matched 32-game screens
use the same seed, openings, weights, and search settings at each budget; only
learned root ordering changes. Game points are from the candidate's perspective
and count a draw as half a point.

| Nodes | Quiet QAdv sorter | Tactical-filter control | Difference |
| ---: | ---: | ---: | ---: |
| 2,000 | 14–15–3 (48.4%) | 12–15–5 (45.3%) | +3.1 pp |
| 8,000 | 18–13–1 (57.8%) | 19–11–2 (62.5%) | −4.7 pp |
| 32,000 | 13–15–4 (46.9%) | 14–13–5 (51.6%) | −4.7 pp |

All six 32-game arenas passed the native replay audit (192 games, 9,077 plies,
1,421 captures, no illegal transitions or duplicate sequences). The curve is
not monotonic and does not beat the frozen control across budgets. With only 32
games per point, this is a screen rather than a precise rating estimate, but it
is enough to reject promotion of this checkpoint as a strength improvement.

The durable summary is [`results.json`](results.json); the full generated JSONL and audit files remain under the ignored `workspace/` directory.
The exporter and trainer are reproducible with:

```bash
cargo build --release --manifest-path research/20260910-quiet-regret-value/Cargo.toml
research/20260910-quiet-regret-value/target/release/quiet-regret-export \
  --max-games-scan 3000 --game-sample-mod 32 --game-sample-bucket 0 \
  --per-bucket 6 --max-roots 96 --candidate-limit 12 \
  --teacher-depth 4 --teacher-nodes 8000 --teacher-beam 32 \
  --continuation-plies 6 --continuation-depth 2 --continuation-nodes 2000 \
  --continuation-beam 32 --min-margin 100 \
  --output research/20260910-quiet-regret-value/workspace/quiet-targets.jsonl

.venv-pathagon-gnn/bin/python research/20260910-quiet-regret-value/train_quiet_value.py \
  --targets research/20260910-quiet-regret-value/workspace/quiet-targets.jsonl \
  --output research/20260910-quiet-regret-value/workspace/quiet-value.pt \
  --resume research/20260901-strong-teacher-10k-games/workspace/retraining-mixed-final/gnn-qadv-replay.pt \
  --steps 3000 --device cpu
```

## Data and artifacts

The research harness, Python trainer, analyzer, lockfile, and this narrative
remain in Git. Generated targets, checkpoints, ONNX, arenas, audits, and JSON
summaries are ignored under `workspace/`; the target, checkpoint, and ONNX hashes
are `cc66eb01a90c13920eab8dad684ebc142b4bd130236f46b4e141d5f9885dcdca`,
`821f6164784ba927132b6df06e6bc329899bfa41600b240ca9d0a081b58026f7`, and
`5112f8dec9e857e501863afe1933650766eb1379eb3efa921f6bdea4290f7b81`. No
labels or model were promoted into `data/`.

## Project impact

This path establishes an end-to-end way to train action-conditioned intuition
from quiet regret and short outcomes, and it confirms native ONNX QAdv sorting
works in the Rust opponent. It does not improve the low-budget strength curve,
so the supported transition-policy v4 default and its Rust/WASM integration are
unchanged.

## Hiccups and limits

An unbounded first export spent minutes scanning and labeling irrelevant roots;
its process was stopped, and an explicit game scan cap plus over-sampled,
phase-balanced reservoirs fixed the protocol. A stale `--help` probe also ran
the default exporter because this one-off binary has no help mode; that process
was stopped before the final run.

The strict 500-point pilot admitted almost no opening roots. Lowering the gate to
100 and retaining more roots restored all phases, but it also lets very large
terminal search scores dominate the mean margin. The source corpus is not a
fresh population: it is replay data from prior agents, and the opponent profile
labels are inferred from observation names. Heldout roots are disjoint by game
key within this export, not guaranteed disjoint from every historical training
archive. The 32-game arena points are noisy, and the continuation target covers
only six plies. The heldout split contains no movement roots, so it cannot test
phase-generalization for relocation. The value-head degradation shows that action
regret and state outcome need separate calibration and likely more balanced,
independently sampled positions.

## Next decision

Retire this candidate as an inconclusive strength result. Keep the exporter and
trainer as research infrastructure, but do not promote the checkpoint or alter
the user-facing default. A follow-up should first enlarge the heldout position
set and calibrate continuation/value targets (including balanced neutral roots),
then run a registered multi-seed arena. The transferable lesson is to measure
label quality, action ranking, board-value calibration, and fixed-budget playing
strength separately; fitting a stronger teacher's choices is not evidence of
learned intuition until all four agree.

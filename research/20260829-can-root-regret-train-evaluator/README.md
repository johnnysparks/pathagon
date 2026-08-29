# 20260829 Can root regret train the evaluator?

Status: inconclusive · not promoted

## Idea

Hypothesis: whole-game win/loss evolution is too sparse and noisy to teach the
evaluator which locally plausible moves are costly. Training against
counterfactual root-action regret from a stronger teacher should improve move
ranking while preserving Pathfinder's bounded Rust search authority.

The important change is the supervision signal, not necessarily the model. A
better linear evaluator is preferred if it can absorb the signal; a compact
learned evaluator is justified only after the linear path is measured.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator` at depth 4 / 2,000 nodes
/ beam 8 with the tactical-safe root filter. Existing native target emitters
can score all legal root actions, and the repository already contains canonical
replay corpora, adversarial fixtures, seeded-position work, and random
midgame/late-game generators.

Prior sorter and Q/advantage work improved selected offline metrics without
reliable whole-game promotion. This path therefore treats root-regret metrics
as a training and diagnostic tool; whole-game strength remains the authority.

Before scaling this work, the compute-budget path should select the intended
few-seconds-per-turn product envelope. Teacher labels may use more offline
compute, while incumbent and candidate arenas must use the same selected product
profile. The 2,000-node profile remains a historical diagnostic.

For this effort, the requested product profile is fixed provisionally at depth
7 / 500,000 nodes / beam 32. The existing WASM probe measured about 1.40 s
median on one 22-ply reference position at that profile. It replaces the
2,000-node envelope for any arena; 2,000 nodes is retained only for historical
comparison and the bounded pilot teacher below.

## Proposal

1. Build a leakage-resistant position set from canonical games, adversarial
   fixtures, randomized phase roots, and representative v0.4/v0.5 self-play.
   Split by source game and seed family before labeling.
2. Label every legal root action with a stronger, explicitly budgeted native
   teacher. Store best-action identity, score gaps, and regret relative to the
   teacher's best action; record teacher configuration with every shard.
3. Train a pairwise or regret-weighted evaluator that prioritizes expensive
   ordering mistakes. Begin with the existing explainable weight family before
   considering a compact model.
4. Freeze candidates using held-out root metrics, then run paired whole-game
   arenas at the selected latency-qualified product envelope, with a smaller
   2,000-node cross-check. Never select a candidate using final arena seeds.
5. Inspect where offline regret and game outcomes disagree, and preserve those
   positions only when they become useful regression fixtures.

## Promotion criteria

All gates must pass:

- On a source-disjoint held-out set, the candidate reduces mean root regret by
  at least 20% versus v0.5 and does not reduce immediate-win or forced-block
  accuracy.
- A final 400-game paired arena against v0.5 at the selected product envelope
  reaches at least 53% game points with a positive win-loss margin in each
  color; a smaller 2,000-node arena must not reveal a severe regression.
- A 120-game v0.4 control screen reaches at least 55% game points, and all
  tactical/human regression fixtures replay and select legally.
- The deployed candidate remains within the selected latency/deadline envelope.
  Any learned runtime must have an explicit size, latency, fallback, and
  reproducibility budget before promotion.

Offline agreement or regret reduction alone is never promotion evidence.

## What happened

The small leakage-audited pilot completed, then stopped at the pre-arena
decision gate.

### Pilot protocol

- The native collector sampled 32 roots: 24 source-disjoint 7×7 canonical
  replay games, 6 source-disjoint seeded phase roots, and 2 human tactical
  fixtures. It preserved the full runtime position at every root.
- The target emitter labeled all 3,500 legal actions with teacher
  `pathfinder-teacher-v1-depth5-2k-beam16`, using v0.5 weights and the d5 /
  2,000-node / beam-16 native search. The target rows also contain the six
  unit evaluator features, capture count, immediate-win flag, safe-root flag,
  teacher score, and teacher exhaustion state.
- The source-group split was frozen before training: 24 roots in training and
  8 roots held out. Human fixtures were always held out. Three deterministic
  3,500-iteration mutation runs trained only the explainable six-weight root
  evaluator.
- The best candidate was
  `path=115, material=112, capture=887, structure=0, threat=115, edge=0`.
  All three seeds reached equivalent training objectives, so there was no
  stable independent improvement to carry into a game arena.

From the repository root, the pilot is reproducible with:

```bash
cargo run --release --manifest-path research/20260829-can-root-regret-train-evaluator/rust/Cargo.toml -- \
  --command collect \
  --games-dir data/corpora/games-v1/games \
  --seeded data/corpora/games-v1/sidecars/seeded-position-20260828-v1.jsonl \
  --human data/fixtures/human-tactical-suite-v1.jsonl \
  --canonical-limit 24 --seeded-limit 6 \
  --output research/20260829-can-root-regret-train-evaluator/workspace/roots.jsonl
cargo run --release --manifest-path research/20260829-can-root-regret-train-evaluator/rust/Cargo.toml -- \
  --command label \
  --roots research/20260829-can-root-regret-train-evaluator/workspace/roots.jsonl \
  --output research/20260829-can-root-regret-train-evaluator/workspace/targets.jsonl \
  --teacher-depth 5 --teacher-nodes 2000 --teacher-beam 16
python3 research/20260829-can-root-regret-train-evaluator/scripts/train_root_regret.py \
  --targets research/20260829-can-root-regret-train-evaluator/workspace/targets.jsonl \
  --output-dir research/20260829-can-root-regret-train-evaluator/workspace/training \
  --iterations 3500 --seeds 20260829 20260830 20260831
```

### Held-out evidence

| partition | v0.5 mean regret | candidate mean regret | reduction | median regret | teacher top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| training, 24 roots | 83,334,865.25 | 83,334,817.50 | 0.00006% | 87.5 → 4.0 | 45.8% → 50.0% |
| held out, 8 roots | 125,006,508.38 | 125,005,098.50 | 0.00113% | 10,203 → 9,915 | 12.5% → 12.5% |

The promotion gate requires at least 20% held-out mean-regret reduction. The
candidate misses it by several orders of magnitude. The held-out set happened
to contain no immediate-win or forced-block roots, so those two tactical gates
were not estimable there; across the full 32-root audit both baseline and
candidate retained 100% immediate-win and forced-block accuracy. Teacher
search was exhausted for 74.8% of held-out actions (43.1% overall), which is a
second reason not to scale this label set as reusable data.

### Promotion gate decision

| gate | decision |
| --- | --- |
| ≥20% source-disjoint held-out regret reduction | Failed: 0.00113% |
| immediate-win and forced-block accuracy | Not estimable on held-out; no full-audit regression |
| 400-game d7 / 500k / beam-32 arena against v0.5 | Not run after the offline pre-gate failed |
| 2,000-node cross-check | Intentionally omitted as a product arena; 2,000 is too low |
| 120-game v0.4 control screen | Not run after the offline pre-gate failed |
| legality, tactical fixtures, latency, deployable runtime | No candidate was deployed; no production change was justified |

Decision: no-go for scaling and no promotion. The result is inconclusive about
the general hypothesis because the bounded teacher exhausted too often and the
held-out pilot was small, but it is strong enough to reject this candidate and
avoid an expensive product-budget arena.

## Data and artifacts

Preserve:

- Generic native target-emission and evaluation code, target schema, split
  manifest, and tests in [`rust/`](rust/) and [`scripts/`](scripts/).
- Reusable labeled positions only when promoted into a strict versioned path
  under `data/` with source provenance and teacher configuration.
- Final candidate weights or the single selected deployable model, its hash,
  stable identity, aggregate arenas, and representative regression fixtures.

Discard or keep ignored in `workspace/`:

- Bulk teacher traces, materialized intermediate datasets, repeated labels,
  optimizer state, rejected checkpoints, and exploratory notebooks/logs.
- Any implementation-shaped tensors that are not a stable interchange asset.

## Project impact

No reusable target shard, candidate weights, opponent manifest, browser asset,
or Rust runtime identity was promoted. The pilot code remains research-only;
the generated roots, labels, split report, and training report remain in this
path's ignored `workspace/`. The existing
`pathfinder-v0.5.0-trained-evaluator` remains the incumbent. The result gives a
small negative signal for the six-weight evaluator but does not distinguish
limited evaluator capacity from an underpowered teacher.

## Kick-off prompt

> Execute the research brief in
> `research/20260829-can-root-regret-train-evaluator/README.md`. Build the
> smallest leakage-audited root-regret pilot that can falsify the hypothesis,
> then scale only if its held-out evidence is useful. Keep v0.5 and the
> tactical-safe v0.4 control frozen, and treat whole-game paired arenas as the
> promotion authority. Use the selected latency-qualified product profile for
> incumbent/candidate arenas and retain 2,000 nodes as a historical diagnostic.
> You have explicit liberty to rename schemas, tools,
> identities, or files and to make justified edits outside this research folder
> under `pathagon/`, `apps/`, `data/`, `docs/`, and `scripts/`. Update every
> affected reference, test, manifest, and generated runtime asset. Preserve
> reusable labels only under a strict versioned `data/` path; keep disposable
> labels, traces, checkpoints, and logs in this path's ignored `workspace/`.
> Finish with a documented promotion or rejection decision against every gate
> in this README.

## Next decision

Revisit only after the product-budget benchmark is durable and a stronger
teacher can label without frequent exhaustion. If revisited, enlarge the
source-disjoint held-out set, keep the d7 / 500k / beam-32 arena profile, and
use the current pilot only as a schema/protocol starting point—not as promoted
training data.

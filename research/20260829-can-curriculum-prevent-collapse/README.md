# 20260829 Can curriculum prevent collapse?

Status: completed

Decision: reject the tested portfolio as the default training source. The
curriculum improved root and phase coverage, but it did not improve whole-game
strength, and the frozen portfolio failed multiple predeclared structural and
product gates. No reusable games, roots, labels, or checkpoints were promoted.

## Idea

Hypothesis: evaluator and policy learning are overfitting a small number of
deterministic trajectory families. A frozen, measurable portfolio of
opponents, openings, and game phases should produce candidates that generalize
better than candidates trained on a narrow self-play stream.

This path promotes a training protocol only if it creates stronger agents under
a common final evaluation. Diversity statistics alone are diagnostic evidence.

## Starting point

The preceding seeded-position study showed that reachable, parity, and
asymmetric roots add meaningful coverage, but its short candidates did not beat
the ordinary-root parent. The random-phase study showed that movement and
late-game roots can be generated, but it did not establish downstream
strength. The available product benchmark also did not select a
latency-qualified envelope; the only measured production-like point was the
historical depth-4, 2,000-node, beam-8 profile.

This completion therefore froze a measurable 0% control versus 50% mixed-root
comparison using the existing matched checkpoints. It is a controlled test of
the available curriculum signal, not a claim that every possible
multi-opponent curriculum has been exhausted.

## What happened

### Frozen protocol

The versioned manifest was frozen before the final arenas in
[`portfolio-v1.json`](portfolio-v1.json). It fixes the 7x7/14-piece rules,
the learner, the seed namespaces, and the held-out boundaries:

- Training control: 96 ordinary-root games, seeds 2026082800–2026082895.
- Training portfolio: 96 games, seeds 2026083000–2026083095, with 48 ordinary,
  19 reachable, 14 parity, and 15 asymmetric roots.
- Selection: 96 untouched games, seeds 2026082900–2026082995.
- Final arenas: seeds 2026083200–2026083839, with paired colors and randomized
  two-ply openings.
- Same residual mean-message-passing learner, 32 hidden units, 4 message
  layers, 400 training updates, symmetry augmentation, and 8 PUCT simulations.

### Structural audit

[`audit_curriculum.py`](scripts/audit_curriculum.py) replayed all three
partitioned source corpora successfully. There was zero seed or root-family
overlap between training and selection partitions, and each partition had 96
unique trajectories. The proposed 50% source had 1,310 movement positions out
of 5,194 (25.22%), so the movement-coverage gate passed.

Two gates failed before strength was considered:

- Only three root families reached the required 15% share: ordinary 50.0%,
  reachable 19.79%, and asymmetric 15.63%. Parity was 14.58%, just below the
  required fourth family.
- Root turns were unbalanced: 71 light versus 25 dark, with 20 of 96 roots
  near terminal positions.

The older 220-game ranked ladder remains diagnostic only; its legacy records
contain 41.8% duplicate action signatures and are not part of the promotion
partitions. The random midgame and late-game files likewise remain diagnostic
root coverage, not promoted training data.

### Strength evidence

Both checkpoints exported to ONNX with passing model-parity checks. The native
historical-2k arena used the identical 400 seeds, colors, opening randomization,
search settings, and v0.5 evaluator weights. The runtime identifies that
opponent as `pathfinder-v0.4.0-tactical-filter`; the weights are the promoted
v0.5 trained-evaluator weights, so this is recorded as a historical tactical
filter cross-check rather than a clean v0.5 product-envelope result.

| Candidate | Wins | Losses | Draws | Game points |
| --- | ---: | ---: | ---: | ---: |
| 0% ordinary-root control | 20 | 380 | 0 | 5.0% |
| 50% mixed-root curriculum | 22 | 378 | 0 | 5.5% |

The required paired 240-game common-seed learner arena also favored the
control: ordinary control 136–104, with no draws. Colors were balanced at 120
games per side for each learner. The mixed-root candidate therefore did not
beat the old-protocol candidate, and neither learner approached the required
53% final strength threshold. The product-envelope gate also remains false
because no latency-qualified production profile was selected.

The complete machine-readable evidence is in the ignored
`workspace/portfolio-v1/` directory: the two ONNX exports, both native 400-game
JSONL arenas, the 240-game common-seed JSON result, and `audit.json`. The audit
reports `promotionEligible: false`.

## Data and artifacts

Kept in Git:

- [`portfolio-v1.json`](portfolio-v1.json), the frozen, rejected protocol
  manifest.
- [`scripts/audit_curriculum.py`](scripts/audit_curriculum.py), the source
  boundary, replay, provenance, diversity, and arena audit.
- [`scripts/run_common_gnn_arena.py`](scripts/run_common_gnn_arena.py), the
  reproducible paired learner arena runner.

Kept only in ignored `workspace/portfolio-v1/`:

- Exported control and portfolio ONNX files and their manifests.
- Native 400-game arenas, the 240-game common-seed arena, and the aggregate
  audit report.

No files were promoted under `data/`: the candidate failed structural,
strength, and product gates. The source games and checkpoints remain owned by
their earlier research path and were not copied here.

## Project impact

This path establishes a reproducible negative result: better root and movement
coverage did not prevent collapse for the tested learner and budget. It also
leaves behind a frozen split manifest and audit/arena tooling that can be
reused when the learner or product envelope changes.

The result narrows the immediate problem but does not prove that curriculum is
useless. In particular, the tested candidate was a seeded-root mixture rather
than a newly generated multi-opponent portfolio, and the product-qualified
arena was unavailable. The evidence says that coverage alone is insufficient
under this protocol; it does not identify whether target semantics, learner
capacity, search budget, root balance, or their interaction is the limiting
factor.

## Next decision

Retire this portfolio version and do not promote it. Revisit only after a
latency-qualified product envelope is selected and the learner/root-regret
work clarifies whether the training target can use the added coverage. A
follow-up should balance root turns and give parity at least 15% of the source
portfolio before spending on a larger multi-opponent generation run.

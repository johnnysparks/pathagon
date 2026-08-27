# Data and artifact policy

The repository contains source code, reproducibility evidence, and large
research outputs. They should not all be managed the same way.

## Source-of-truth rules

| Data | Source of truth | Git policy |
| --- | --- | --- |
| Rules and UI code | `app/`, `engine-rs/`, `research/gnn/` | Track normally |
| Contract and parity fixtures | `contracts/`, `fixtures/` | Track; changes require tests |
| Small curated corpus | `research/corpora/` | Track and review diffs |
| Experiment records | `research/experiments/` | Track serious successes, failures, lineage, and artifact references |
| Large self-play archives | D1, local archive, or external storage | Ignore unless explicitly promoted |
| Active run output | `research/runs/`, `training/` | Ignore; promote only durable evidence |
| Selected small checkpoints | `research/experiments/` | Track with manifest, hash, and retention rule |
| Dataset manifests and hashes | `research/experiments/`, `research/corpora/` | Track |
| Browser WASM/model artifacts | `public/engine/`, `public/models/` | Generated release artifacts with recorded hashes |

## Canonical game corpus

Historical replay archives are normalized into
[`research/corpora/games-v1/`](../research/corpora/games-v1/). A game is
content-addressed by its rules/configuration and action sequence. Seeds, model
identities, outcomes, and source provenance are separate observations keyed to
that game, so copied archives and train/held-out exports do not duplicate the
game itself.

The game table excludes policy tensors, Q-target arrays, visit counts, and
search diagnostics because they are not part of game identity. Rust reconstructs
game states by replaying the canonical actions. Architecture-independent labels
that remain useful for training may be promoted into a separate, versioned
game-keyed sidecar under `research/corpora/`; implementation-shaped intermediates
belong in experiment storage instead.

Do not partition or copy canonical games by agent, opponent, or model. Those
relationships belong in keyed observations and experiment manifests. The full
experiment and external-artifact policy is in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Dataset requirements

Every promoted dataset needs:

- board size and reserve configuration;
- source archive list;
- duplicate-removal policy;
- train/held-out split seed;
- model and optimizer seeds;
- human-game consent/provenance status;
- SHA-256 hashes for ignored inputs and outputs when practical.

Training and held-out records must be disjoint by game or seed group. A model
score without its split manifest is a diagnostic, not a promotion result.

## Human games

Human games are valuable training and evaluation data only when their use is
covered by an explicit privacy and consent policy. They are archived separately
from machine self-play and should be labeled in dataset manifests.

## Cleanup rule

Do not erase the record of an old run merely because it is weak. Mark the
experiment as failed, inconclusive, historical, or retired; retain its lineage,
protocol, completed game keys, decisive result, and artifact manifest. Bulky
checkpoints, repeated replay payloads, temporary datasets, and verbose logs may
be removed once important content is canonicalized or externally stored with a
verified hash and stable location.

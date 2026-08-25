# Data and artifact policy

The repository contains source code, reproducibility evidence, and large
research outputs. They should not all be managed the same way.

## Source-of-truth rules

| Data | Source of truth | Git policy |
| --- | --- | --- |
| Rules and UI code | `app/`, `engine-rs/`, `research/gnn/` | Track normally |
| Contract and parity fixtures | `contracts/`, `fixtures/` | Track; changes require tests |
| Small curated corpus | `research/corpora/` | Track and review diffs |
| Large self-play archives | D1, local archive, or external storage | Ignore unless explicitly promoted |
| Selected checkpoints | `research/runs/` | Track when small and named in a report |
| Dataset manifests and hashes | `research/runs/` | Track |
| Published lab snapshots | `public/lab/` | Build/deployment output; do not hand-edit |
| Browser WASM/model artifacts | `public/engine/`, `public/models/` | Generated release artifacts with recorded hashes |

The current `public/lab/` files mirror training archives for static display.
That duplication is transitional: the training artifact should be canonical,
and the public copy should be generated from it.

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

Do not delete an old run merely because it is weak. Mark it as historical,
retain its report and manifest, and remove only bulky replay payloads whose
hashes and source location are recorded elsewhere.

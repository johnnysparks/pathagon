# `<experiment-id>` · `<short title>`

Status: `planned | running | succeeded | failed | inconclusive | abandoned`

## Hypothesis

State the expected improvement and the one primary variable changed.

## Lineage

- Parent experiment:
- Parent/baseline agent:
- Candidate agent:
- Candidate model hash:
- Git commit:

## Protocol

- Commands/runner:
- Rules and board configuration:
- Dataset/corpus version:
- Split and seed policy:
- Held-constant settings:
- Opponents, model hashes, and color balancing:
- Predeclared success/failure boundary:

## Canonical games

Reference `research/corpora/games-v1` game keys through `games.tsv`, observation
source IDs, or a deterministic selector and result hash. Do not copy moves into
this directory.

## Results

Report outcomes by opponent and color, relevant training metrics, latency or
resource cost, and representative game keys/counterexamples.

## Decision and learning

Record whether the candidate advances, what failed or remained inconclusive,
and which follow-up should inherit from this experiment.

## Artifacts

| Role | Git path or stable external URI | SHA-256 | Bytes | Keep because |
| --- | --- | --- | ---: | --- |

Do not include credentials or temporary signed URLs.

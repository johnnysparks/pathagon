# Rust Pathfinder ladder matchup

## Question

How do the current learned/trained Pathfinders compare with the supported Rust
baseline opponents that had no live leaderboard games after the v4 promotion?

## Protocol

Run the promoted Transition v4 model and the trained evaluator control against
The Pathfinder v0.4 tactical filter, The Surveyor, Lunatic, and Coin Flip using
the native Rust self-play engine. Use the production 7×7 search envelope,
randomize the first two plies, alternate colors, and generate 100 total games:
13 per v4 pairing and 12 per v0.5 pairing. Import only records whose agent
identities are in the supported Rust leaderboard roster.

Generated JSONL and corpus files belong in this path's ignored `workspace/`;
the live D1 archive is the durable game store for the matchup evidence.

## Outcome

The Rust runner produced and the live cross-play importer accepted all 100
games under run `pathagon-rust-pathfinder-ladder-20260830`. Replay validation
passed before import; the live API reported `inserted: 100`.

| Learned/trained Pathfinder | v0.4 tactical filter | Surveyor | Lunatic | Coin Flip |
| --- | ---: | ---: | ---: | ---: |
| Transition v4 (13 games each) | 7–6 | 6–7 | 12–1 | 13–0 |
| v0.5 trained evaluator (12 games each) | 8–4 | 1–11 | 12–0 | 12–0 |

The first number in each cell is the learned/trained Pathfinder's wins; there
were no draws in this batch. The new live totals are 3,245 archived games and
1,100 ranked games. This small batch is useful coverage, but it is not a new
promotion signal: both learned/trained Pathfinders were vulnerable to the
deeper Surveyor profile in this sample, so the existing v4 promotion and v0.5
rollback/control status remain unchanged.

## Project impact and promotion decision

This is an evaluation/coverage batch for the existing promoted roster. It does
not promote a new model or change the user-facing default. The prior v4
promotion evidence and rollback control remain authoritative until a larger,
pre-registered comparison warrants a research decision.

## Failures and follow-up

Record any runner, import, legality, or replay-audit failures here. Keep
disposable logs and replay exports out of version control.

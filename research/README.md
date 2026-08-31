# Research history

Research is a date-first project archive, not a second production tree. Each
path is named `YYYYMMDD-short-question` and should make sense when read on its
own. Start with its `README.md`; code and small evidence may sit beside it.

A good archive README records the idea, approach, outcome, generated data,
project impact, failures, and promotion decision. Research code is allowed to
be awkward, partially tested, or tied to a one-time format. Large games,
checkpoints, targets, and logs belong in an ignored `workspace/` inside the
dated path. If a large output cannot be reconstructed perfectly and has no
lasting product value, it may be discarded.

## Promotion boundary

An experiment does not become supported merely because it worked. Port useful
agents and opponent behavior to Rust, define strict contracts for promoted
data, add high coverage, inspect representative game outputs, and retain only
high-value artifacts in Git. Canonical reusable games and labels belong in
[`../data/`](../data/), not duplicated across research paths.

Use [`TEMPLATE.md`](TEMPLATE.md) for a new path. The current direction is in
[`../docs/RESEARCH.md`](../docs/RESEARCH.md).

## Archive

- [`20260824-4x4-endgame-tactics/`](20260824-4x4-endgame-tactics/)
- [`20260824-gnn-cnn-lab/`](20260824-gnn-cnn-lab/)
- [`20260824-separated-value-action-policy/`](20260824-separated-value-action-policy/)
- [`20260825-selfplay-corpus-audit/`](20260825-selfplay-corpus-audit/)
- [`20260826-adversarial-self-play/`](20260826-adversarial-self-play/)
- [`20260827-pathfinder-rust-sorter/`](20260827-pathfinder-rust-sorter/)
- [`20260828-seeded-position-curriculum/`](20260828-seeded-position-curriculum/)
- [`20260829-superdeep-contextual-evaluator/`](20260829-superdeep-contextual-evaluator/)
- [`20260829-turn-balanced-contextual-evaluator/`](20260829-turn-balanced-contextual-evaluator/)
- [`20260829-action-transition-policy/`](20260829-action-transition-policy/)
- [`20260829-board-aware-policy-value/`](20260829-board-aware-policy-value/)
- [`20260829-nextgen-action-transition/`](20260829-nextgen-action-transition/)

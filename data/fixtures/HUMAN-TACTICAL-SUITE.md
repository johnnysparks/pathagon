# Human tactical suite

`data/fixtures/human-tactical-suite-v1.jsonl` is a small, held-out suite
of positions mined from a user-provided human-vs-Pathfinder replay. It is an
evaluation and search-debugging artifact, not a claim that human move volume
is representative of the full game distribution.

## Current fixture

Source replay: `08de361a-de4b-425a-8a98-801408c49dee` (`pathfinder-v0`, light
won, 33 plies).

The suite captures two adjacent positions:

1. Before ply 31, light to move. The recorded `D7` placement captures `E7`
   and creates the winning fork setup.
2. After ply 31, dark to move. Every legal dark action leaves an immediate
   light win. The complete winning-reply union is `C3` or `D3`; the recorded
   line is `D3`, followed by `C3`.

The fixture deliberately stores rule-relevant positions and outcome labels,
not a Pathfinder slider setting. Human replay metadata is not required for
this evaluation case.

## Verification

Run the focused Rust checks with:

```bash
cargo test --manifest-path pathagon/engine-rs/Cargo.toml --test human_tactical
```

The integration test reconstructs both snapshots, verifies the mined capture,
and enumerates all legal dark replies to prove the forced `C3`/`D3` finish.

## How it is used

Keep these positions out of ordinary training batches. Use them to compare
Pathfinder variants at matched budgets: baseline alpha-beta, wider beams,
transposition-table ordering, selective tactical extensions, and QAdv root
sorting. Promotion requires improvement here without regressing the existing
solver-labelled tactical suite or paired-game gates.

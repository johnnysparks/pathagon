# 20260901 JEPA afterstate representation

Status: running

## Idea

Use a compact joint-embedding predictive objective to learn “what changed and
why it matters” from exact Pathagon transitions. The Rust engine remains the
world-model authority: it enumerates legal actions and applies them to produce
the target afterstate. The learned representation is an auxiliary signal for
the existing policy/value learner and, later, a candidate root action ordering
signal. Rust search remains the planner and tactical safety boundary.

This is the narrowest useful application of the JEPA/H-JEPA ideas to Pathagon.
The game is fully observable and deterministic, so the first version has no
latent stochastic variable and does not attempt to predict pixels or replace
the rules engine.

## Starting point

The user-facing default is the Rust/WASM `pathfinder-action-transition-v4-xent`
model. The current GNN learner already has dynamic place/relocate action heads,
while the Q/Advantage path consumes explicit afterstate features. The latest
strong-teacher replay path provides a useful source of positions but is capped
at 20 plies and does not provide complete root-Q vectors for every move.

The new `pathagon-jepa-export` binary writes exact
`(state, legal action, nextState)` rows. The research Python module trains an
EMA-target encoder and action-conditioned predictor in embedding space. It is
not a production opponent and must pass the normal source-disjoint, legality,
paired-arena, latency, and replay gates before any promotion.

## Protocol

1. Generate transition rows with `pathagon-jepa-export`; the binary is the
   authority for legal actions and successor states.
2. Validate the rows against the mirrored Python decoder only as an audit; do
   not use the Python rules implementation to generate targets.
3. Train a 64-dimensional context embedding with one-ply prediction, VICReg
   variance/covariance regularization, and optional policy/value replay loss.
4. Add four-ply targets only after one-ply prediction is stable. Treat that as
   the first H-JEPA-like temporal hierarchy; do not invent abstract actions yet.
5. Export the shared policy/value trunk and use it only to order all legal
   actions before the existing Rust search. Keep the tactical-safe root guard
   and full-root fallback.
6. Compare against frozen v4 with source-disjoint positions, paired colors,
   held-out phases, representative replay audits, and browser-cost checks.

## What happened

The initial end-to-end smoke passed. Eight Rust-generated games at 12 plies
produced 96 sampled positions and 1,536 exact transition rows. The Python
loader validated every row against its mirrored decoder, and the JEPA training
run completed on 1,152 train rows with 384 rows held out by game. Held-out
invariance loss was `0.03034`; the variance/covariance regularizers remained
finite.

The online GNN trunk was then fine-tuned for two replay steps, exported through
the existing GNN ONNX exporter, and loaded by the inference-enabled Rust
`pathagon-selfplay` binary. A two-game, 12-ply PUCT smoke arena completed with
24 legal plies and two capped draws. This proves the data/training/export/
search wiring, not a strength improvement; the tiny capped arena is not a
promotion gate.

One integration failure occurred when the legacy exporter was invoked as a
file rather than as a package module. The smoke script now invokes
`python.export_gnn` with the correct package path.

## Data and artifacts

Generated JSONL transitions, checkpoints, reports, and logs belong in this
path's ignored `workspace/`. No generated artifact is promoted yet. A reusable
corpus may be promoted under `data/` only after its schema, provenance,
source-disjoint split, and value are established.

## Project impact

The Rust exporter establishes the exact-engine data boundary. The JEPA learner
is intentionally research-only until it demonstrates that shared transition
representations improve the existing search-backed player rather than merely
improving an offline embedding loss.

## Next decision

The next experiment should use a larger source-disjoint transition corpus,
retain a frozen v4 comparison, and add four-ply targets only if the one-ply
representation remains stable. A promotion decision still requires a paired
arena, full replay legality audit, and browser-cost measurement.

# Scale-invariant GNN AlphaZero lab

This is the first implementation of the proposed GNN learner. It is a
research pipeline, not the browser opponent.

## What is implemented

- Variable-size orthogonal Pathagon graphs for 5x5, 7x7, and larger boards.
- Four typed virtual goal nodes for the two connection objectives.
- Residual mean-message-passing layers with LayerNorm.
- A dynamic policy head: node logits for placement and source/destination
  pair logits for relocation.
- A graph-level value head using mean/max pooling plus normalized game state.
- Replay validation and policy/value warm-start training from schema-v2 JSONL.
- PUCT search and neural-guided self-play example generation.
- Compact AlphaZero-style generation/training loop.

The model is scale-compatible, but scale-compatible weights are not proof of
zero-shot playing strength. The learner receives normalized coordinates,
reserves, turn state, capture state, and boundary roles; board size is still
provided as a feature so it can adapt rather than confuse 5x5 and 7x7.

## Local setup

```bash
python3 -m venv .venv-pathagon-gnn
.venv-pathagon-gnn/bin/python -m pip install -r learning/gnn/requirements.txt
```

Warm-start from the complete local Rust archive:

```bash
.venv-pathagon-gnn/bin/python -m learning.gnn.train warmstart \
  --data /tmp/pathagon-rust-selfplay-100-20260823.jsonl \
  --out training/gnn/pathagon-warmstart.pt \
  --steps 200
```

Run a small neural self-play generation:

```bash
.venv-pathagon-gnn/bin/python -m learning.gnn.train alphazero \
  --resume training/gnn/pathagon-warmstart.pt \
  --out training/gnn/pathagon-generation-0.pt \
  --games 8 --simulations 64 --updates 200
```

Use `--size 5` with a fresh model to exercise the dynamic graph path. The
5x5 rules adapter uses a scaled reserve of `2 * size` by default; it is a
curriculum environment, not a claim that the historical 7x7 game had a
different reserve count.

## Important boundaries

The 100/120 archived games are a warm-start signal and a replay-validation
fixture. They are far too small to justify AlphaZero conclusions. The old
tabular book remains useful as a baseline, but this GNN should be evaluated
against search and random with disjoint seeds and multiple color-balanced
batches before any promotion.

The Python rules adapter is tested against the shared move semantics during
development. The Rust/TypeScript engines remain the production authorities;
the next hardening step is a generated parity corpus across both board sizes
before using 5x5 games as curriculum data.

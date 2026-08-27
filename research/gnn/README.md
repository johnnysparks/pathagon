# 7x7 GNN/CNN AlphaZero lab

This is the Python research and model-tooling layer, not the automatic browser
promotion path. The canonical comparison target is 7x7 with 14 reserves. See
[`docs/LEARNING.md`](../../docs/LEARNING.md) for model status and
[`docs/WORKFLOWS.md`](../../docs/WORKFLOWS.md) for the end-to-end workflow.

The GNN remains the scale-compatible research baseline; the CNN path is
intentionally fixed to 7x7 so it can be compared under the same PUCT and replay
workflow. Smaller boards are supported for curriculum and regression work, not
as competing league targets.

## What is implemented

- Variable-size orthogonal Pathagon graphs for 4x4, 5x5, 6x6, 7x7, and larger boards.
- Four typed virtual goal nodes for the two connection objectives.
- Residual mean-message-passing layers with LayerNorm.
- A dynamic policy head: node logits for placement and source/destination
  pair logits for relocation.
- A graph-level value head using mean/max pooling plus normalized game state.
- Replay validation and policy/value warm-start training from contract-v1 JSONL (with schema-v2 compatibility).
- PUCT search and neural-guided self-play example generation.
- Compact AlphaZero-style generation/training loop.
- Rules-preserving D4 symmetry augmentation during replay and self-play-target
  optimizer updates.
- A small residual CNN alternative with the same dynamic policy/value heads.

The model is scale-compatible, but scale-compatible weights are not proof of
zero-shot playing strength. The learner receives normalized coordinates,
reserves, turn state, capture state, and boundary roles; board size is still
provided as a feature so it can adapt rather than confuse 4x4, 5x5, 6x6, and 7x7.

## Local setup

```bash
python3 -m venv .venv-pathagon-gnn
.venv-pathagon-gnn/bin/python -m pip install -r research/gnn/requirements.txt
```

Warm-start from the complete local Rust archive:

```bash
.venv-pathagon-gnn/bin/python -m research.gnn.train warmstart \
  --data /tmp/pathagon-rust-selfplay-100-20260823.jsonl \
  --out research/runs/gnn/pathagon-warmstart.pt \
  --steps 200
```

Warm-start the fixed-size CNN alternative:

```bash
.venv-pathagon-gnn/bin/python -m research.gnn.train warmstart \
  --architecture cnn --size 7 --hidden 32 --cnn-blocks 4 \
  --data /tmp/pathagon-rust-selfplay-100-20260823.jsonl \
  --out research/runs/gnn/pathagon-cnn-7x7-warmstart.pt \
  --steps 200
```

Run a small neural self-play generation:

```bash
.venv-pathagon-gnn/bin/python -m research.gnn.train alphazero \
  --resume research/runs/gnn/pathagon-warmstart.pt \
  --out research/runs/gnn/pathagon-generation-next.pt \
  --games 8 --workers 4 --selfplay-device cpu \
  --simulations 64 --updates 10000 --max-plies 196 \
  --replay-limit 100000
```

Generate a larger, provenance-stamped 7x7 data batch from the three current
neural players:

```bash
./.venv-pathagon-gnn/bin/python scripts/generate-7x7-selfplay.py \
  --games-per-player 1000 --players scout,learner,cnn \
  --workers 8 --simulations 4 --temperature-moves 32 \
  --max-plies 196 \
  --output-dir research/runs/gnn/benchmark-7x7/generated/<batch-id>
```

The batch runner gives each player a disjoint seed range and writes separate
JSONL archives plus a manifest containing checkpoint hashes and result counts.
Each game is same-model self-play (the selected player controls both colors),
so cross-player matches should be generated as a separate evaluation slice.
The runner uses `--updates 0`: it creates data only and does not adapt the
checkpoints during generation.

For a 7x7 run, use the full 196-ply board cap so move-cap draws are not
introduced by the learner. Training targets should use 32-64 PUCT simulations;
the defaults retain 64 simulations, 10,000 optimizer updates, and 100,000
replay positions. A 1,000-game pilot is the first meaningful data milestone;
scale toward 10,000 games only after its held-out 7x7 evaluation results improve.

The small graph operations used during search are typically faster on CPU;
checkpoint updates can still use MPS through the default `--device auto`.

Training samples a random one of the eight square-board symmetries by default.
The four axis-preserving transforms keep player identities unchanged; rotations
by 90 degrees and diagonal reflections also exchange Light and Dark so the
vertical and horizontal connection objectives remain equivalent. This is
augmentation rather than a fixed architectural guarantee, and
`--no-symmetry-augmentation` is available for ablation runs.

The CNN requires `--size 7`; it is a deliberately focused comparison model.
The GNN can still use `--size 5` with a fresh model to exercise the dynamic
graph path. Smaller boards remain curriculum and regression environments, not
part of the canonical 7x7 training distribution.

## Browser CNN deployment

Export a fixed 7x7 CNN checkpoint as a single-file ONNX artifact for the
Rust/WASM inference session:

```bash
./.venv-pathagon-gnn/bin/python -m research.gnn.export \
  --checkpoint research/runs/gnn/benchmark-7x7/cnn-warmstart.pt \
  --output public/models/pathagon-cnn.onnx
npm run build:engine
```

The native Rust parity harness can load either the shared policy/value trunk or
the complete QAdv export from the same checkpoint. The second form includes
the deterministic 24-feature transition tensor and direct Q/A action ranking:

```bash
./.venv-pathagon-gnn/bin/python -m research.gnn.export_gnn \
  --checkpoint research/runs/gnn/benchmark-7x7/generated/<qadv-batch>/qadv-arbiter-7x7-v0.1.0-exploration-20260825.pt \
  --output work/rust-qadv-spike/qadv-gnn-policy-value.onnx
cargo run --release --manifest-path engine-rs/Cargo.toml \
  --features inference --bin pathagon-selfplay -- \
  --onnx work/rust-qadv-spike/qadv-gnn-policy-value.onnx \
  --opponent neural --simulations 128 --workers 2 --jsonl

./.venv-pathagon-gnn/bin/python -m research.gnn.export_gnn \
  --checkpoint research/runs/gnn/benchmark-7x7/generated/<qadv-batch>/qadv-arbiter-7x7-v0.1.0-exploration-20260825.pt \
  --output work/rust-qadv-spike/qadv-gnn-qadv.onnx --include-qadv
cargo run --release --manifest-path engine-rs/Cargo.toml \
  --features inference --bin pathagon-selfplay -- \
  --eval-only --qadv-onnx work/rust-qadv-spike/qadv-gnn-qadv.onnx \
  --eval-sequence P24,P0,P1,P2,P3,P4

# Full native exploration controls: Pathfinder blend, temperature schedule,
# opening uniform mix, and two low-priority workers for development headroom.
cargo run --release --manifest-path engine-rs/Cargo.toml \
  --features inference --bin pathagon-selfplay -- \
  --onnx work/rust-qadv-spike/qadv-gnn-policy-value.onnx --guided \
  --simulations 128 --temperature-moves 48 --policy-temperature 1.15 \
  --opening-moves 16 --opening-temperature 1.8 --opening-randomness 0.30 \
  --pathfinder-guidance 0.45 --placement-guidance 0.30 \
  --pathfinder-temperature 1.15 --pathfinder-depth 2 \
  --pathfinder-beam 8 --pathfinder-nodes 512 --workers 2 --jsonl
```

The build expects the pinned `wasm-bindgen` CLI to be available on `PATH`.

The small rules/search module is emitted to `public/engine`. The learned CNN
runtime is emitted separately to `public/engine-inference` and is loaded only
when the CNN opponent is selected. Its PUCT API uses Rust-generated legal
actions, the exported policy logits as priors, and the exported value head at
leaf nodes.

## Important boundaries

The 100/120 archived games are a warm-start signal and a replay-validation
fixture. They are far too small to justify AlphaZero conclusions. The old
tabular book remains useful as a baseline, but this GNN should be evaluated
against search and random with disjoint seeds and multiple color-balanced
batches before any promotion.

The Python rules adapter is tested against the shared move semantics during
development. Rust is the production rules/search authority; TypeScript remains
the browser reference/coaching implementation for regression tests. The
Rust/WASM and CNN inference boundaries require the same contract and parity
checks before browser promotion.

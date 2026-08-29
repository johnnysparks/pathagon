# 7x7 adversarial and regression cases

This directory is the durable registry for positions and trajectories that
teach us something unusual. It is deliberately separate from the bulk
self-play corpus and from the held-out evaluation pool.

`cases.json` contains seed-level references into generated game archives. The
registry is not a second corpus: training code should load a referenced record
at most once and cap its sampling weight. Every registry seed should remain
outside the held-out evaluation split when it is used for hard-example
training.

The first critical case, `2026201789`, crossed the old internal 180-ply limit
while the match contract allowed 196 plies. It is now a native-engine
regression test for the max-ply propagation fix. Other cases are mined from
the 17,500-game campaign by long horizon, capture density, low action variety,
small Q top-gap, and the placement-to-relocation phase transition.

Refresh the registry after a corpus or backup batch changes:

```bash
python3 research/20260826-adversarial-self-play/scripts/mine-adversarial-cases.py
```

The registry also records four generation profiles for future targeted
rollouts. Those games should be emitted into a dated adversarial batch, audited
separately, and then merged into training through an explicit capped manifest;
they should not be copied into the primary campaign directory by hand.

Generate the current four-profile targeted batch locally with the corrected
Rust engine and the exact deployed QAdv model:

```bash
python3 research/20260826-adversarial-self-play/scripts/generate-adversarial-selfplay.py \
  --model research/20260826-adversarial-self-play/workspace/qadv-arbiter.onnx
```

The default batch is 16 games (four each for placement exploration, ranking
ambiguity, capture pressure, and long horizon). It is intentionally small and
reviewable; increase it only after checking the manifest and targeted audit.

# Tile Driver · GNN policy/value v1

This is the real graph policy/value artifact used by the Tile Driver opponent
card. Rust owns the 7×7 graph features, legal action order, PUCT search, and
WASM boundary; the browser only supplies controls and renders the aligned
results.

The artifact passed native Rust loading, ONNX export parity, and legal-action
alignment checks. It is browser-playable and intentionally labelled
provisional: a strength ladder against the Pathman default remains separate
from the artifact promotion gate.

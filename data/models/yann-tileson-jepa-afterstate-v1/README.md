# Yann Tileson · JEPA afterstate rank/value v1

This artifact is the first playable JEPA path for the roster. It is not the
embedding-only smoke checkpoint: the trained model contains independent
afterstate action-ranking and bounded action-value heads. Rust owns the graph
and action ABI plus every successor transition used by the bounded browser
beam search.

The artifact passed native Rust loading, export parity, and legal-action
alignment checks. It is browser-playable and provisional pending a longer
paired arena and replay audit.

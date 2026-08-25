# Research run artifacts

This directory contains model checkpoints, replay datasets, evaluation
archives, and dated experiment reports. Python source code lives in
[`research/gnn/`](../gnn/); this directory is for its inputs and outputs.

## Current entry points

- [`gnn/README.md`](gnn/README.md) — current GNN/CNN checkpoint and benchmark index
- [`gnn/benchmark-7x7/`](gnn/benchmark-7x7/) — deduplicated 7x7 benchmark
- [`gnn/benchmark-7x7-full-20260824/`](gnn/benchmark-7x7-full-20260824/) — full-data split manifest
- [`gnn/round-full-20260824/`](gnn/round-full-20260824/) — dated full-model training report
- [`rust-v1/README.md`](rust-v1/README.md) — Rust evaluator training result

## Artifact rule

Large JSONL archives are research data, not source code. Keep them external or
ignored unless a manifest identifies them as a curated reproducibility input.
Keep selected checkpoints, manifests, hashes, and reports in Git when they are
small enough to review and reproduce.

The dated directories are intentionally stable because scripts and reports refer
to them directly. Add new artifacts under the closest existing category and keep
large generated data external or ignored.

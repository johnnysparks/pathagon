# GNN/CNN training artifacts

This directory contains outputs from the Python learning pipeline. Source code
lives in [`learning/gnn/`](../../learning/gnn/); workflow commands live in
[`docs/WORKFLOWS.md`](../../docs/WORKFLOWS.md).

## Current entry points

| Path | Meaning |
| --- | --- |
| `benchmark-7x7/` | Earlier deduplicated 7x7 CNN/GNN benchmark and held-out results |
| `benchmark-7x7-full-20260824/` | Full-data 7x7 split manifest; large replay files are ignored |
| `round-full-20260824/` | Dated compact-GNN, CNN, and full-GNN training report |
| `league/` | Historical and current pairwise/league archives |
| `pathagon-*.pt` | Warm-start and generation checkpoints |
| `*.jsonl` | Replay archives or generated self-play data |

The dated reports are the primary index for model claims. A checkpoint without
its dataset manifest, training configuration, and evaluation record is only an
initialization artifact.

## Current hardware note

On the current Apple Silicon development hardware, small unbatched replay
warm-starts are faster on CPU than MPS because launch and scalar-synchronization
overhead dominates. Re-benchmark before changing that default for larger,
batched, or self-play workloads.

## Model history

Older generations and smaller-board curriculum runs remain available for
regression and historical comparison. They are not part of the canonical 7x7
strength claim unless a report explicitly says so.

The full-data snapshot is documented in
[`round-full-20260824/README.md`](round-full-20260824/README.md), and the
earlier matched benchmark is documented in
[`benchmark-7x7/RESULTS.md`](benchmark-7x7/RESULTS.md).

## Artifact policy

Large replay files should remain ignored or external unless they are promoted
to a curated dataset with a manifest and hash. Selected checkpoints and compact
reports may remain in Git when they are small enough to review.

Generated 7x7 batch JSONL archives and self-play logs are ignored by default;
their durable Git record is the adjacent manifest, report, and any explicitly
selected checkpoint artifacts.

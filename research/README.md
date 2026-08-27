# Research

The canonical historical game dataset is the content-addressed, Rust-readable
corpus under [`corpora/games-v1/`](corpora/games-v1/). Large legacy run files
remain archival inputs rather than the durable game representation.

This directory is the home for Pathagon research experiments and their
reviewable artifacts.

| Path | Responsibility |
| --- | --- |
| [`gnn/`](gnn/) | Python GNN/CNN research code, tests, and dependencies |
| [`corpora/`](corpora/) | Curated, reproducible datasets suitable for Git |
| [`runs/`](runs/) | Dated checkpoints, replay data, manifests, evaluations, and reports |

Keep source code in `gnn/`, keep generated data inside a named run bundle, and
retain a manifest or report whenever an artifact is used as evidence. Large
replays remain ignored or external; selected checkpoints and compact reports
may be committed when they are reproducible and reviewable.

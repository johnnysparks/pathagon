# 7x7 GNN/CNN AlphaZero lab

Status: historical research platform; useful engine behavior was promoted to
Rust, but the Python agents are not supported opponents.

## Idea

Test whether graph and convolutional policy/value learners, PUCT self-play,
and replay-derived targets could produce stronger 7×7 play while supporting
smaller-board curriculum and symmetry augmentation.

## What was built

The [`python/`](python/) implementation contains variable-size graph features,
a fixed 7×7 CNN alternative, dynamic placement/relocation policies, value and
Q/advantage training, PUCT, replay validation, symmetry transforms, league
evaluation, and ONNX export. Historical one-off runners are in [`scripts/`](scripts/).

The work generated checkpoints, replay archives, target datasets, league
results, and browser ONNX exports. Most large outputs were disposable local
work. Reusable move histories and labels were later consolidated under
[`../../data/corpora/`](../../data/corpora/).

## Outcome

The lab established useful model/export contracts and exposed tactical and
action-ranking weaknesses, but offline loss and ranking improvements did not
translate into stable playing strength. The Python runtime was too
research-shaped to become a supported opponent.

## Project impact

Cross-runtime contracts, corpus replay, native target generation, inference
adapters, tactical filtering, and several search controls moved into
[`../../pathagon/engine-rs/`](../../pathagon/engine-rs/). Browser release assets
remain under `apps/web/public`, but the Python agents and their training state
were not promoted.

## Running historical code

The code is retained for archaeology and follow-up work, not guaranteed as a
supported workflow. Its package can be addressed by adding this research path
to `PYTHONPATH` and running modules beneath `python`, for example:

```bash
python3 -m venv .venv-pathagon-gnn
.venv-pathagon-gnn/bin/python -m pip install -r research/20260824-gnn-cnn-lab/python/requirements.txt
PYTHONPATH=research/20260824-gnn-cnn-lab \
  .venv-pathagon-gnn/bin/python -m python.train --help
```

Generated checkpoints and games should go under this path's ignored
`workspace/` rather than a new top-level data tree.

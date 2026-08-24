# Full archive compact-GNN runs

These three compact GNN checkpoints use the same 7x7 replay split and 2,000
symmetry-augmented warm-start updates. The large JSONL replay files are kept
locally and ignored by Git because the full `all.jsonl` file exceeds the
repository's per-file limit.

Dataset: `training/gnn/benchmark-7x7-full-20260824/`

- 6,141 unique games / 617,240 positions
- 4,911 train games / 497,464 positions
- 1,230 held-out games / 119,776 positions
- 15 validated human games, split 10 train / 5 held out
- imported/offline cross-play export: 1,104 records; legacy live-generator records excluded
- split seed: `20260824`; optimizer seed: `20260825`, `20260826`, or `20260827`

The dataset manifest records the full source list and the ignored local files
have SHA-256 hashes recorded below. Rebuild the corpus with:

```sh
python3 scripts/build-7x7-benchmark.py \
  --root training/gnn \
  --output training/gnn/benchmark-7x7-full-20260824 \
  --heldout-fraction 0.2 \
  --seed 20260824
```

## Local artifact hashes

These files remain on the working machine and are intentionally ignored by
Git:

| file | SHA-256 |
| --- | --- |
| `archive/cross-play-db-20260824-0000.jsonl` | `2b9f5d3dce0926f74c3be431009cfbadebf2d1c1fe2ad6c1cbb99e96968a9306` |
| `archive/cross-play-db-20260824-0500.jsonl` | `fda5c9816abdc697487ac1b8a30cd9f54138857d90c12fa8ae454e6fea01411e` |
| `archive/cross-play-db-20260824-1000.jsonl` | `b78715d2af20c2b09b7d6148c81f842b82594d390d63bb648eec9d5bfc13271f` |
| `archive/human-games-web-20260824.jsonl` | `fa5da4c1aeb74061ecc8d86589ed744d9b1a9a6fdfe94e0a91135423917e0820` |
| `benchmark-7x7-full-20260824/all.jsonl` | `fbe2b6c3147b0cfabd320754f3fa7c2281bcb80f7c8545f1fd61b654c95e7060` |
| `benchmark-7x7-full-20260824/train.jsonl` | `5d115f3ff4e1083664d47dfb4948b449c412d16a95fee7bb93aba6672798ecee` |
| `benchmark-7x7-full-20260824/heldout.jsonl` | `83838192499afdfcf383df3eb78b0cebdd6403251126be5fb76a2c883de8660b` |

The compact checkpoints are small enough to keep in Git:

| checkpoint | SHA-256 |
| --- | --- |
| `compact-gnn-seed20260825.pt` | `b33bceea9e6959437f992fac7e43b1f4bb2807012a39dbf8a9956de438387758` |
| `compact-gnn-seed20260826.pt` | `785dc7eb6c5a5b8ed0697287597a8176f44984e7873e64e08dab7258ad346b17` |
| `compact-gnn-seed20260827.pt` | `b3cc114a8fae0874b6a47cb5accb1566c3a1365f87b038b8f490e97619502a9a` |

## Full-model warm starts

The larger comparison models were trained from the same 4,911-game training
split for 2,000 symmetry-augmented updates on MPS. The held-out split was not
used during training.

| model | architecture | policy loss | value loss | checkpoint |
| --- | --- | ---: | ---: | --- |
| CNN | 32 channels × 4 residual blocks | 2.4928 | 0.3025 | `cnn-full-seed20260828.pt` |
| Full GNN | 64 channels × 8 message layers | 2.2681 | 0.3220 | `full-gnn-seed20260829.pt` |

| checkpoint | SHA-256 |
| --- | --- |
| `cnn-full-seed20260828.pt` | `c5dca2e7e69c01469169ba4e428e8b27c616a08e529f7ebee93d9cbb9d767056` |
| `full-gnn-seed20260829.pt` | `c399a4cbaf07777ffb2ec50b7caa03f90299fc65c9a86c3ac5cc782441ef50f6` |

## Hardware note: Apple Silicon

On this Apple Silicon / PyTorch environment, CPU is the preferred device for
the current warm-start implementation. The loop performs one small graph
forward/backward pass per example and synchronizes scalar losses every update,
so MPS launch and synchronization overhead outweighs its accelerator benefit.

| model | CPU updates/sec | MPS updates/sec | CPU speedup |
| --- | ---: | ---: | ---: |
| CNN, 32 × 4 | 103.9 | 12.5 | 8.3× |
| Full GNN, 64 × 8 | 92.7 | 13.1 | 7.1× |

These are 300-update measurements on the same full 7×7 training split, not a
claim about all Apple Silicon workloads. Use `--device cpu` for these small
unbatched replay warm starts; re-benchmark before switching to MPS for larger
models, batched training, or self-play search.

## Held-out scoring snapshot

These scores use the same seeded 10,000-example sample from the 119,776-example
held-out split (`--seed 20260825`, 873 games), evaluated on CPU. The sample
contains 6,935 draw, 1,469 loss, and 1,596 win targets. NLL is lower; top-1
and top-5 are higher; value MSE is lower. This is a progress snapshot, not a
replacement for a complete held-out pass or an arena strength test.

| model | policy NLL | top-1 | top-5 | value MSE |
| --- | ---: | ---: | ---: | ---: |
| CNN full | 2.0564 | 55.8% | 82.0% | 0.3066 |
| Full GNN | 1.6868 | 62.5% | 86.8% | 0.3199 |
| Compact GNN seed 20260825 | 1.9360 | 51.9% | 81.5% | 0.3071 |
| Compact GNN seed 20260826 | 1.9456 | 52.5% | 83.2% | 0.3062 |
| Compact GNN seed 20260827 | 1.9830 | 52.4% | 81.6% | 0.3072 |

The full-model phase split is:

| model / phase | policy NLL | top-1 | top-5 | value MSE |
| --- | ---: | ---: | ---: | ---: |
| CNN / placement | 2.0016 | 55.6% | 80.7% | 0.4830 |
| CNN / relocation | 2.1064 | 55.9% | 83.2% | 0.1454 |
| Full GNN / placement | 1.6346 | 63.5% | 84.5% | 0.5062 |
| Full GNN / relocation | 1.7345 | 61.7% | 88.8% | 0.1497 |

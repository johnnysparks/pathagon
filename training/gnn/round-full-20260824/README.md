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

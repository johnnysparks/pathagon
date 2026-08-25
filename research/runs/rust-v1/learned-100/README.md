# Provisional learned book

Agent: `rust-learned-tabular-v0.1.0`

Source: uploaded Rust self-play run `rust-selfplay-100-20260823`.

This artifact is an exact-state replay book, not a general-strength model.
Unseen states fall back to the Rust search agent. It was built with:

```bash
python3 scripts/jsonl-to-rust-games.py \
  --input /tmp/pathagon-rust-selfplay-100-20260823.jsonl \
  --output /tmp/pathagon-rust-selfplay-100.games.tsv
npm run rust:learn -- \
  --games /tmp/pathagon-rust-selfplay-100.games.tsv \
  --out research/runs/rust-v1/learned-100
```

Smoke evaluation against `rust-surveyor-v0.1.0` at `--seed 20261001` for 20
games: 9 wins, 11 losses, 0 draws, with 21 learned-book moves. This candidate
has not been promoted to the browser game.

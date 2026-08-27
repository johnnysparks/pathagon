# Compact 7×7 game history

This directory contains the Git-sized history slice for the 20k-game campaign.

`game-history-20260826.p1` contains 20,000 deduplicated, lossless replay records:

- 17,500 games from the Rust Lambda campaign (`2026200000`–`2026217499`)
- 2,500 clean QAdv cross-play games (`2026220000`–`2026222499`)

The `p1` format is one tab-separated record per game:

```text
p1  seed64  light-agent  dark-agent  winner  reason  2-char-actions
```

Each action is encoded in two base-64 characters. The action stream is enough to
replay the game and regenerate board states, captures, final-state badges, and
playback views. The verbose JSON transcripts remain local/cold storage because
they include derived engine diagnostics and full per-action Q/visit vectors.

The canonical conversion is implemented by
`scripts/jsonl-to-rust-games.py`. The source manifests, audits, and the QAdv
held-out evaluation report remain alongside the generated campaign artifacts.

# Unified compact game corpus v1

This corpus is generated from historical Pathagon archives by
`scripts/compact_game_corpus.py`. It is deterministic, content-addressed, and
split into small shards for reviewable Git diffs.

## Identity

A `g1_...` key is the SHA-256 digest of the rules version, board size, reserve,
repetition limit, and 12-bit encoded action sequence. Seeds, agents, engines,
outcomes, source paths, and search/training annotations do not affect identity.

`games/` stores each unique game once. `observations/` associates source and
run metadata with a game key. `sources.tsv` maps compact source IDs back to the
original archive paths.

Observations retain the agent ID and model/checkpoint hash for each color when
available, plus outcome and minimal provenance. Policy tensors, Q arrays,
visits, and search scores are not part of the game table. Useful universal
targets may be promoted separately into versioned, game-keyed corpus sidecars.
Every game state can be reconstructed by replaying the canonical action
sequence in Rust.

## Rebuild

```bash
python3 scripts/compact_game_corpus.py   --input research/runs   --output research/corpora/games-v1
```

The command refuses to replace an existing output directory unless
`--replace` is provided. When the output exists, it is loaded as the durable
base before new inputs are scanned; use `--no-base` only for an intentional
from-scratch rebuild. Parse or normalization errors produce `errors.jsonl` and
a nonzero exit unless `--allow-errors` is explicitly selected.

# Learning from the game archive

The first learning agent is intentionally simple and inspectable:
`rust-learned-tabular-v0.1.0`.

It is a replay-derived exact-state policy. For every observed position and
legal action, it stores visits, wins, losses, and draws from the perspective
of the player who made that move. At play time it chooses the action with the
best empirical points rate, requires a configurable minimum number of visits,
and delegates unseen or weakly supported positions to the existing search
agent.

That design is appropriate for the current “crazy small” dataset because it
cannot invent a broad strategy from a handful of games. Its weakness is also
visible: exact board positions rarely repeat, so most moves still use search.
The book is a useful retrieval and evaluation baseline, not evidence that the
agent has learned Pathagon in a general sense.

## Current candidate

`training/rust-v1/learned-100/` was built from the uploaded
`rust-selfplay-100-20260823` run:

- 100 games
- 3,719 replay moves
- 3,624 exact position/action entries
- 20-game held-out-style smoke evaluation versus `rust-surveyor-v0.1.0`
- result: 9 wins, 11 losses, 0 draws
- 21 learned-book moves; all other moves fell back to search

The evaluation is too small to establish a strength improvement. The
candidate is not wired into the browser game and should not be treated as a
promoted model.

## Rebuild from an archive export

For a complete local export:

```bash
curl -H "OAI-Sites-Authorization: Bearer $PATHAGON_ARCHIVE_TOKEN" \
  'https://pathagon-game.sparks-house-6466.chatgpt.site/api/selfplay?engine=rust&format=jsonl&limit=500' \
  > /tmp/pathagon-rust-archive.jsonl
python3 scripts/jsonl-to-rust-games.py \
  --input /tmp/pathagon-rust-archive.jsonl \
  --output /tmp/pathagon-rust-archive.games.tsv
npm run rust:learn -- \
  --games /tmp/pathagon-rust-archive.games.tsv \
  --out training/rust-v1/learned-latest
```

Run evaluation with a new seed range and report `bookHits` from the aggregate
JSON line. For a serious promotion gate, keep the evaluation games separate
from the games used to build the book, compare against the incumbent over
multiple color-balanced batches, and retain every evaluation replay.

## Next learning steps

1. Add more varied games, including human games only after their privacy and
   consent policy is settled.
2. Add canonicalization or feature buckets so related positions can share
   evidence without collapsing tactically different states.
3. Use the replay archive for offline evaluation and dataset versioning before
   attempting policy-gradient, value-network, or self-play reinforcement
   learning.
4. Promote a candidate only after it passes disjoint replay tests and repeated
   arena gates; the browser engine remains unchanged until then.

## GNN AlphaZero direction

The next learner is under `learning/gnn/`. It follows the scale-invariant
proposal with two Pathagon-specific changes: policy logits are defined over
the legal action list so relocations have source/destination heads, and the
value path receives explicit reserves, capture, turn, repetition-adjacent,
and virtual-goal features alongside pooled node embeddings.

The implementation currently supports dynamic 5x5 and 7x7 graph construction,
replay warm-start training, PUCT search, and a compact neural self-play loop.
The 7x7 archived games are used only to initialize and exercise the pipeline;
they are not enough to establish AlphaZero strength. Curriculum learning starts
after variable-size parity is expanded beyond the current unit cases.

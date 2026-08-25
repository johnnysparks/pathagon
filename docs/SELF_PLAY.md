# Self-play and evaluation

Self-play is an offline experiment workflow. It produces replayable records;
the leaderboard displays records imported from that workflow.

## Owners

- Rust provides high-throughput search, self-play, and evaluator tournaments.
- Python provides neural self-play, checkpoint evaluation, and learned-agent
  pairwise matches.

The browser-facing TypeScript rules and search implementation remain the
reference/coaching path, but the former TypeScript offline arena is retired.
Rust owns handcrafted evaluator promotion, while Python owns GNN/CNN checkpoint
experiments.

## Offline match generation

```bash
./scripts/run-rust-archive.sh macbook-lunatic-001 1000 20260824 lunatic
```

For neural checkpoint comparisons, use the dedicated Python match and
self-play scripts documented in [`docs/WORKFLOWS.md`](WORKFLOWS.md).

## Evaluation requirements

- use a fresh seed range for evaluation;
- alternate light and dark assignments;
- record engine, model, search, board, and reserve configuration;
- retain draws and termination reasons;
- keep training and promotion games disjoint;
- include the relevant playable opponents, not only self-play.

Elo-like ratings are provisional summaries. They are not human Elo and should
not be treated as promotion evidence by themselves.

## Leaderboard boundary

There is no live web cross-play generator in the official workflow. New
leaderboard games are generated offline, validated, imported into D1, and then
aggregated for display. This makes the source of every result inspectable and
prevents UI polling from becoming an implicit experiment runner.

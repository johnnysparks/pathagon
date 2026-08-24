# Self-play and evaluation

Self-play is an offline experiment workflow. It produces replayable records;
the leaderboard displays records imported from that workflow.

## Owners

- TypeScript provides browser-reference regression arenas and historical league
  comparisons.
- Rust provides high-throughput search, self-play, and evaluator tournaments.
- Python provides neural self-play, checkpoint evaluation, and learned-agent
  pairwise matches.

TypeScript promotion training is retired. Rust owns handcrafted evaluator
promotion, while Python owns GNN/CNN checkpoint experiments.

## TypeScript regression arena

```bash
npm run selfplay -- --mode arena --games 20 --seed 20260822
npm run selfplay:league -- --games 8 --seed 20260823
```

The runner alternates colors and uses deterministic seeds. Its output under
`selfplay/progress/runs/` is disposable until explicitly archived.

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

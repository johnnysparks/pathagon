# Checkpoint league archives

These JSON archives contain the exact games used to produce the provisional Elo standings in the learning lab. Every record includes the board size, reserve size, agent IDs, winner/draw reason, and move list.

The roster combines versioned GNN checkpoints with four fixed opponents:

- `pathfinder-v0.3.0`: tactical depth search with a wider connection-aware evaluation.
- `surveyor-v0.2.0`: shallower strategic search with a broader candidate beam.
- `lunatic-v0.1.0`: the browser-matched one-ply local-pattern heuristic.
- `coin-flip-v0.0.1`: uniform random legal action.

Each pair plays with both color assignments. Ratings start at 1000 and use online Elo updates with `K=24`; they are useful as a promotion signal, not yet a statistically stable strength estimate. The 5×5 pool uses four games per matchup and four MCTS simulations. The 7×7 pool uses two games per matchup and one MCTS simulation to keep the local CPU benchmark bounded.

`league-5x5-r8-generation-10.json` is the latest candidate evaluation. Generation 10 won its direct four-game matchup against Generation 9 (2 wins, 2 draws), but finished below the incumbent in the broader league and was not promoted.

The transfer diagnostics cover `league-4x4-r6-transfer.json` and `league-6x6-r10-transfer.json`. They reuse 5×5 checkpoints without fine-tuning to test whether the dynamic graph and rules remain tractable at those sizes.

The Lunatic-inclusive curriculum snapshots are `league-4x4-r6-lunatic.json`,
`league-5x5-r8-lunatic.json`, `league-6x6-r10-lunatic.json`, and
`league-7x7-r14-lunatic.json`. They add the browser Lunatic opponent to every
board-size competition with paired color assignments.

The Rust fast-path archive `rust-lunatic-7x7.jsonl` contains 100 Pathfinder vs
Lunatic games. Its matching compact, indexed corpus lives at
`training/rust-v1/lunatic-100-7x7/`.

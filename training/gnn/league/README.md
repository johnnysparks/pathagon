# Checkpoint league archives

These JSON archives contain the exact games used to produce the provisional Elo standings in the learning lab. Every record includes the board size, reserve size, agent IDs, winner/draw reason, and move list.

The roster combines versioned GNN checkpoints with three fixed opponents:

- `pathfinder-v0.3.0`: tactical depth search with a wider connection-aware evaluation.
- `surveyor-v0.2.0`: shallower strategic search with a broader candidate beam.
- `coin-flip-v0.0.1`: uniform random legal action.

Each pair plays with both color assignments. Ratings start at 1000 and use online Elo updates with `K=24`; they are useful as a promotion signal, not yet a statistically stable strength estimate. The 5×5 pool uses four games per matchup and four MCTS simulations. The 7×7 pool uses two games per matchup and one MCTS simulation to keep the local CPU benchmark bounded.

`league-5x5-r8-generation-10.json` is the latest candidate evaluation. Generation 10 won its direct four-game matchup against Generation 9 (2 wins, 2 draws), but finished below the incumbent in the broader league and was not promoted.

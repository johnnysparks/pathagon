# Active research direction

The canonical target is 7×7 with 14 reserves per player. Pathfinder with the
native tactical-safe root filter remains the default control, while the newly
promoted trained evaluator is the higher-ranked playable opponent.
The filter is the clearest recent promotion result: it substantially improved
paired play against the prior Pathfinder at the same game rules and bounded
search budget.

Learned policy, Q/advantage, root-sorter, and proof-guided experiments have
improved some offline metrics but have not produced repeatable promotion-grade
strength. Filter-aware evaluator evolution produced the first higher-ranked
candidate in this cycle.
The seeded-position curriculum increased near-terminal coverage but its short
ladder candidates remained below their parent. The next useful work should
change one major variable at a time, use paired colors and held-out positions,
and preserve the tactical filter as a frozen control.

Current questions:

1. Can the trained evaluator hold its advantage on a larger post-deployment
   ladder without increasing the browser search envelope?
2. Can better starting-position coverage improve move ranking without reducing
   ordinary whole-game strength?
3. Which opponent portfolio and opening policy produces diverse games without
   over-weighting deterministic trajectory families?
4. Which learned artifact, if any, improves the Rust player's strength enough
   to justify its inference and deployment cost?

Historical evidence and detailed outcomes live with the research paths:

- [`../research/20260827-pathfinder-rust-sorter/`](../research/20260827-pathfinder-rust-sorter/)
- [`../research/20260828-budgeted-pathfinder/`](../research/20260828-budgeted-pathfinder/)
- [`../research/20260828-proof-guided-pathfinder/`](../research/20260828-proof-guided-pathfinder/)
- [`../research/20260828-seeded-position-curriculum/`](../research/20260828-seeded-position-curriculum/)
- [`../research/20260825-selfplay-corpus-audit/`](../research/20260825-selfplay-corpus-audit/)
- [`../research/20260824-4x4-endgame-tactics/`](../research/20260824-4x4-endgame-tactics/)

New work starts as another dated research path. Promotion requires a Rust port,
focused coverage, strict reusable data, and representative game review.

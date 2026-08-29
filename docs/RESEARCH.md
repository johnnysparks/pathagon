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

## Pathfinder improvement line

The current strength line is:

1. The 20260827 sorter study found that learned root ordering was not reliably
   stronger, but a pure-Rust tactical-safe root filter was: 629–169–2 over 800
   paired games. That became `pathfinder-v0.4.0-tactical-filter`, the frozen
   control.
2. The 20260828 budgeted evaluator study evolved the evaluator under the same
   depth-4 / 2,000-node / beam-8 envelope and promoted
   `pathfinder-v0.5.0-trained-evaluator` after a 70–47–3 held-out screen.
3. Proof-guided and seeded-position follow-ups improved selected offline
   measures but did not beat the frozen control. A longer three-generation
   evaluator run from the handcrafted seed also produced no promotion; its best
   held-out result was 13–11–0 over 24 games.
4. A rerun after the alpha-beta sentinel-bound correction in `42e89299` scored
   70–48–2. This preserves the strength signal while explaining why the old
   summary is not bit-for-bit reproducible.

The durable product state is therefore a trained Pathfinder default, a
tactical-safe control for all future screens, and a single canonical search
envelope. The detailed protocols, negative results, and artifact provenance
remain in the linked dated paths below.

The depth-4 / 2,000-node / beam-8 envelope is now the frozen historical
comparison profile, not an assumption about the ideal product budget. A direct
WASM probe shows enough headroom to investigate a few-seconds-per-turn default
before more learning work is promoted.

The 20260829 three-second study is complete. Its deadline-bounded depth-6 /
100k / beam-16 candidate remained responsive in a cancelable Worker but scored
49.9% game points in the final 400-game paired arena, so no deeper profile was
promoted. The deadline export, Worker boundary, and durable benchmark fixture
remain reusable infrastructure while the supported v0.5 default stays frozen.

## Proposed next paths, ranked

1. [`What fits in three seconds?`](../research/20260829-what-fits-three-seconds/)
   — completed without a profile promotion; the responsive browser execution
   boundary and durable benchmark remain available for a later attempt.
2. [`Can v0.5 evolve further?`](../research/20260829-can-v05-evolve-further/)
   — restart evaluator evolution from the promoted weights at the selected
   product envelope.
3. [`Can root regret train the evaluator?`](../research/20260829-can-root-regret-train-evaluator/)
   — pilot complete; held-out regret gain was 0.00113%, so no candidate was
   promoted and the path is inconclusive pending a stronger teacher.
4. [`Can curriculum prevent collapse?`](../research/20260829-can-curriculum-prevent-collapse/)
   — test whether a frozen opponent/opening/phase portfolio produces more
   general candidates.
5. [`Can a gated sorter help?`](../research/20260829-can-gated-sorter-help/)
   — completed calibration audit; no useful activation region, so not promoted.

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
- [`../research/20260829-can-root-regret-train-evaluator/`](../research/20260829-can-root-regret-train-evaluator/)
- [`../research/20260825-selfplay-corpus-audit/`](../research/20260825-selfplay-corpus-audit/)
- [`../research/20260824-4x4-endgame-tactics/`](../research/20260824-4x4-endgame-tactics/)

New work starts as another dated research path. Promotion requires a Rust port,
focused coverage, strict reusable data, and representative game review.

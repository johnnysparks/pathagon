# Self-play corpus audit · 2026-08-25

## Finding

The broad local 7x7 replay tree is not a clean training corpus. A read-only
scan of 48 JSON and JSONL sources found 25,732 records but only 6,586 unique
complete trajectories: 19,146 records (74.4%) were exact repeats. The
canonical benchmark ingestion, which excludes browser-wrapper records, sees
25,392 records and 6,246 unique trajectories.

The worst concentration is in generated QAdv batches:

- `qadv-pathfinder-700`: 700 records collapsed to 2 trajectories;
- the 100-game CNN, GNN, and Surveyor QAdv batches each collapsed to 2;
- the combined Rust archive still contains 113 duplicates in 1,100 records.

The issue is deterministic action selection and tie-breaking, not an
effective random seed. Once two runs reach the same state, they repeatedly
choose the same action sequence.

## Data-quality artifacts

The scan also found 6,819 legacy records without `contractVersion`, about
2.2 million positions without policy targets, and only 6,822 positions with
action-value targets. About 2.0 million positions have zero search nodes and
zero completed depth. These are useful outcome archives, but they should not
be mixed into the Q/Advantage training pool as though they were ranked-action
examples.

Long-game termination is another visible concentration: 7,143 local records
end at `max-plies`, compared with 18,550 path wins and 39 threefold draws.
That distribution should be measured separately from decisive tactical
games.

## Cleanup policy

Historical local archives remain preserved. The training boundary should use a
deduplicated manifest and keep at most one record for an exact key of:

`rules/config + directed agent pair + winner + action sequence`

This preserves the difference between the same board sequence produced by
different opponents while preventing repeated deterministic games from
receiving thousands of times the training weight. The live cross-play archive
was cleaned with this policy on the same date: 2,256 records became 1,080,
with zero remaining exact duplicates under that key. Browser-generated
records were left untouched by the imported leaderboard filter.

## Exploration path

1. Keep three pools: Q/Advantage targets with valid ranked-action metadata,
   policy/value replays with validated outcomes, and a permanently held-out
   evaluation pool.
2. Vary the first 4–8 plies, use root noise or temperature sampling, and
   sample from a top-k action set instead of always taking argmax.
3. Keep Pathfinder as the primary opponent, but mix in Surveyor, neural
   candidates, Lunatic, and a controlled random/exploration policy. Balance
   colors and openings.
4. Mine positions where the top two actions have a small Q margin, action
   rankings disagree, a capture or relocation occurs, or the game is near a
   win. Cap each trajectory family so hard-example mining cannot recreate a
   single attractor.
5. Archive only unique trajectories per directed matchup/configuration, then
   retrain in measured batches. Compare unique-game rate, move-type mix,
   max-plies rate, ranking agreement, and Elo—not just total games.

A useful next experiment is four 100-game batches with randomized openings,
temperature/top-k action sampling during the opening, Pathfinder in roughly
half the games, and the remainder spread across the portfolio. Retrain after
each batch from the clean pools and keep evaluation games out of training.

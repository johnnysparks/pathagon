# Three generations of learned intuition

Status: running — all collection complete; generation three labeling

## Idea

Hold the authoritative Rust engine, the existing 32–32 tanh transition-policy
architecture, feature normalization, and deployment search budget fixed across
three consecutive learning generations. Generate complete affordable games,
selectively investigate decisions with deeper search, and fine-tune the parent
weights. Compare every generation with the starting v4 model and identical
search without learned root guidance.

## Starting point

Clean worktree at experiment start. The supported v4 transition scorer supplies
generation zero. No gameplay engine changes are intended. Runtime calibration
established sample sizes before training began. The production envelope is
d5/b256/256k; an affordable collection envelope can differ but must remain fixed
across generations. The normal 196-ply rules cap is preserved, never shortened
for collection; cap terminations are reported separately and never used as
outcome supervision. This is policy learning, with no value head.

## What happened

Created a Rust research runner using the supported engine. Calibration and
training/export smoke completed. The first 80-game collection passed native
replay and termination audit: 80 unique trajectories, 3,643 plies, 895
relocations, 75 path wins, four repetition draws, and one 196-ply draw.
Generation-one labeling and training completed. Of 640 selected roots, 403
reached depth five or six (317 train, 86 held out). Held-out teacher top-one
agreement rose from 10/86 to 12/86, top-three from 21/86 to 24/86, and cross
entropy fell from 3.535 to 3.416. These are modest imitation diagnostics; no
strength result has been inspected. Generation two is collecting games using
the generation-one weights, with the fixed architecture and normalization.

## Data and artifacts

Generated games, labels, models, telemetry, and logs stay in ignored workspace/.
The runner, frozen protocol, compact results, and narrative will remain here.

## Project impact

No production changes or promoted artifacts yet.

## Hiccups

Prior research used truncated outcomes and different search envelopes. This
experiment must preserve termination semantics, source-disjoint evaluation,
paired opening seeds, and explicit timing measurements. A negative result is
valid; three generations must be completed regardless of intermediate strength.

## Next decision

Finish all three generations and the two fixed-anchor comparisons, run
verify.py and summarize.py, review representative games, and decide whether
the evidence supports a promotion confirmation or a negative research result.

## Frozen protocol

See protocol.json. Eighty complete self-play games per generation; up to eight
selectively deeper-labeled roots each; 20 fine-tuning epochs from the parent
with cumulative replay. Deployment comparisons hold d5/b256/256k fixed, with
100 games against each anchor per generation and 100 starting-model/control
games. The deadline is disabled for deterministic node-budget comparisons;
wall time is recorded and product latency requires a separate check if a
candidate merits promotion. Training settings will not change based on arenas.

## Execution notes

The campaign runner is `python3 research/20260909-three-generation-intuition/run.py`.
It is resumable using successful subprocess completion markers, and its logs
are per-stage in workspace/. Inspect the actual live process/session before
resuming; never start a second copy while it is running. The initial active
exec session is 90344. Calibration sessions 49623 (production budget) and
55407 (label budget) are independent of the campaign.

Calibration d4 completed four distinct path-win games (32, 32, 35, 63 plies).
The native replay audit passed all 162 plies and 27 captures, including 28
relocations. Fine-tuning/export smoke passed; smoke weights are not included
in the campaign. Engine source is unchanged. The campaign was launched before
the final summarize invocation was added to run.py, so invoke summarize.py
manually after this first run finishes. Run the native `target/release/audit
--arena <file>` on each complete collection and evaluation file before final
interpretation, and record its compact outputs. Generation results are pending.

Final calibration evidence is in calibration.json. The production games both
reached path wins in 56 plies. All 32 selected calibration roots were labeled;
only completed-depth >=5 roots are eligible for the experiment learner.
The stronger native auditor now verifies termination reasons as well as moves.
After the campaign, run verify.py and summarize.py, then review representative
replays and finalize the narrative/promotion decision.

Generation-two collection completed with 80 fresh seeds and 80 unique
trajectories, using generation-one weights on both sides. The native audit
verified 3,786 plies, 640 captures, and 968 relocations: 75 path wins and five
threefold draws, with no ply-cap terminations. Generation-two selective
labeling is running at the unchanged d6/b256/512k teacher envelope.

Before any arena was started, confirmation selection was made explicit in
protocol.json: a candidate must screen at >=53% against both anchors and
above 50% in both colors for each matchup. The best minimum-anchor performer
then receives 400 fresh paired games against each anchor, with independent
confidence, legality, representative-game, and deployment-latency gates before
promotion. Screen and confirmation results will not be pooled. No model or
search/training setting changed as part of this decision.

Generation two trained from generation one after all 640 selected roots were
labeled (384 eligible). Cumulative replay supplied 622 training roots. On the
same 86 generation-one held-out roots, top-one agreement changed from 12/86
to 11/86, top-three from 24/86 to 26/86, and cross entropy from 3.416 to
3.398. This is a mixed offline signal, not a strength result. Architecture,
normalization, and parent lineage checks passed. Generation three is now
collecting complete games using generation-two weights on both sides.

Generation-three collection completed and passed the native audit: 80 unique
games, 3,851 plies, 594 captures, 1,064 relocations, 74 path wins, and six
threefold draws. It used generation-two weights and fresh seeds. Across all
three batches there are 240 disjoint seeds and 240 unique action trajectories,
covering 11,280 legal plies. The final selective-labeling pass is running.

## Superseded before arenas

The absolute completed-depth-five gate retained only 2 of 345 selected
relocation roots in generations one and two, despite the teacher actually
searching deeper on 341 of them. Placement roots had median 31–33 candidate
actions; relocation roots had median 273. This was an acceptance-rule flaw
that defeated the intended full-game coverage. Two models were trained; the
third label pass was stopped, and no arena was run. All original artifacts
are retained in workspace/absolute-depth-pilot/. The corrected campaign
restarts at generation zero, reusing only its unchanged first-batch games and
raw teacher labels. No old trained weights enter the corrected lineage.

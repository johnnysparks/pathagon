# Three generations of learned intuition

Status: completed on 2026-09-10 — all training, screening, and confirmation
finished; no candidate promoted.

## Idea

Test a closed learning loop for three consecutive generations while holding the
Rust engine, 32–32 tanh transition-policy network, normalization, evaluator
weights, and deployment search budget fixed. Collect complete affordable games,
selectively relabel positions with deeper search, update the current policy
using cumulative replay, and compare with both the starting policy and the same
search with neural root ordering disabled.

## Starting point

The starting model is the supported v4 transition scorer, SHA-256
`f11d7ddee101ccab35ee162e53c95ced076b1fb10242443ad562dbd51c1085d4`.
`protocol.json` records the frozen engine hash and settings. Each generation
collects 80 complete 7×7/14-reserve games with four randomized opening plies.
Collection uses depth 4 / beam 256 / 32,000 nodes; teacher labeling uses depth
6 / beam 256 / 512,000 nodes; evaluation uses depth 5 / beam 256 / 256,000 nodes.
Node-budget comparisons have no wall-clock deadline. The normal 196-ply cap
and threefold termination remain in force.

This is policy learning through search-generated action targets. Game outcomes
are not value labels. Neural guidance orders the root; exact tactical checks
and the recursive evaluator remain in Rust. The NN-off anchor retains the same
frozen evaluator weights, so it does not remove all historical learning.

## What happened

All three corrected generations completed: 240 distinct self-play trajectories,
11,219 legal plies, and 1,920 selected roots. Relative-depth eligibility retained
1,847 roots, including 482 of 491 relocation roots. Cumulative training used
491, 991, and 1,477 unique training positions. Each update starts from its
immediate parent, uses 20 epochs at learning rate 0.0003, and preserves feature
normalization, architecture, and parent-policy KL regularization.

On the fixed 122-root heldout diagnostic, top-one teacher agreement across
G0/G1/G2/G3 was 25/28/25/26 correct; top-three was 36/38/50/43. Cross entropy
was 3.616/3.337/3.278/3.338. These offline metrics are non-monotonic and are not
a playing-strength verdict. See `learning-summary.json`.

The complete screening scores are:

| Candidate | Against starting G0 | Against NN-off search |
|---|---:|---:|
| Starting G0 | — | 48.5% |
| G1 | 61.0% | 46.5% |
| G2 | 56.5% | 53.5% |
| G3 | 54.0% | 51.0% |

Each cell is 100 games with paired openings and swapped candidate colors.
`RESULTS.md` and `results.json` include draws, paired-bootstrap intervals, color
splits, repetition, trajectory diversity, and resource statistics. Scores do
not show successive improvement. G1's favorable G0 result does not carry over
to NN-off search. G2 is the only candidate reaching the registered 53% overall
threshold against both anchors, but its intervals against both anchors include
50%. Screening alone does not establish a reliable gain.

`verify.py` passed all checks: frozen engine/architecture/normalization/budgets,
sequential parent hashes, native feature and logit parity, paired opening
identity, disjoint training/evaluation seeds, and full native replay auditing of
940 games and 50,551 moves. `verification.json` contains the compact evidence.
A representative G1 screen review is in `REPLAY-REVIEW.md`; it does not replace
review of the selected finalist's independent games.

Generation two was selected by the registered rule in
`confirmation-selection.json`. It received 400 fresh paired games against each
anchor, with seed bases 59091000 and 59092000. Confirmation results were never
pooled with screening results:

| Independent G2 confirmation | W–L–D | Points | Paired bootstrap 95% |
|---|---:|---:|---:|
| Against starting G0 | 198–183–19 | 51.875% | 47.875–55.875% |
| Against NN-off search | 203–178–19 | 53.125% | 48.750–57.625% |

Both intervals include 50%, and G0 confirmation also misses the 53% point-rate
threshold. The registered strength gate failed against both anchors. This is
not evidence of a reliable gain, nor proof that the true gain is exactly zero.
`CONFIRMATION.md` and `confirmation-results.json` preserve the independent
results and explicit no-promotion decision.

The final verification reran the frozen-protocol checks and audited all 1,740
games and 97,567 moves, including both confirmation arenas. Representative
confirmation wins, losses, repetition draws, and cap draws were inspected with
the native replay tool and recorded in `REPLAY-REVIEW.md`. Neither the aggregate
nor those samples establish a new reliable strategic capability. Conditional
deployment-latency and promotion work were skipped after the strength failure.

## Data and artifacts

Generated games, raw/annotated labels, checkpoints, verbose logs, and one-time
snapshots stay in ignored `workspace/`. Research source, protocol, compact
reports, and narrative evidence stay here. No model or dataset has been
promoted to a supported location under `data/`.

The superseded absolute-depth pilot is preserved in
`workspace/absolute-depth-pilot/`, with its compact explanation in
`ABSOLUTE-DEPTH-PILOT.md`. The first batch's unchanged raw games and labels were
reused through symlinks with recorded hashes in `workspace/g1-source-reuse.json`.
No pilot-trained weights enter the corrected lineage; G2/G3 games and labels
were regenerated from their corrected parents.

`run.py` resumes completed training and screening stages. `confirm.py` first
verifies the completed campaign, selects the finalist, and resumes confirmation
stages. The campaign has finished. Check for a live process before invoking either
again; completed stages are verified and skipped rather than regenerated. Native `audit` checks every transition and termination; native
`inspect` produces readable board snapshots without reimplementing the rules.

## Project impact

The production engine, supported models, and user-facing default remain
unchanged. The experiment supplies a controlled, verified learning loop and
an explicit account of what has and has not improved. G2 failed the independent strength gate against both anchors. No model was
promoted and no conditional deployment-latency run was needed.

## Hiccups and limits

The first label filter required absolute completed depth five. It retained only
2 of 345 selected relocation roots from the first two pilot generations,
although the teacher searched deeper than the collector on 341 of them.
Relocation roots had a median 273 candidate actions versus 31–33 for placement.
That filter erased movement supervision. This was a protocol-design error,
not evidence that the network cannot learn movement.

Before any arena, all three updates restarted from G0 under one consistent
rule: teacher completed depth must strictly exceed the collector's completed
depth. The unchanged first batch was reused; all learned weights were restarted.
`absolute-depth-protocol.json` and `absolute-depth-phase-diagnostic.json`
preserve the discarded approach and evidence.

Also before any arena, an overly strict requirement to win above 50% in each
color was replaced with overall points in balanced pairs. Requiring a winning
Dark score can incorrectly reject improvement in a game with first-player
advantage. Color splits remain diagnostics; the superseded rule is retained in
`superseded-confirmation-color-rule.json`. No evaluation scores informed this
adjustment.

Teacher actions are bounded-search targets, not exact game-theoretic answers.
Extra completed depth does not prove every label is better. Root policy learning
can alter search behavior without producing a stronger opponent. Small paired
screens, repeated trajectories, and opponent dependence limit conclusions.
Equal node limits also do not imply equal latency; deployment cost needs its
own check. Full-game policy collection avoids inventing value labels for draws,
but this experiment does not test learning a value function from outcomes.

## Next decision and transferable lessons

Close this run and retain the current default. Do not continue this same
three-generation run with extra epochs, changed budgets, or post-hoc gates.
The result is a controlled negative promotion result with uncertainty about
small strength differences, not a claim that neural intuition is impossible.

The workflow lessons are concrete:

1. Audit supervision by game phase before training. A superficially reasonable
   quality filter can remove the very behavior that needs to be learned.
2. Measure the benefit of teacher decisions separately from the ability to fit
   them. Here deeper-than-collector labels were verified, but we did not
   independently establish that those decisions improve on the deployed
   depth-5 / 256k player. A stronger label budget alone is not that evidence.
3. Keep multiple fixed anchors and independent confirmation. A 61% screen
   against the starting model did not translate to NN-off strength, and the
   selected G2's favorable screen weakened on fresh games.
4. Treat generation number and offline ranking accuracy as diagnostics. Neither
   is a substitute for matched-budget playing strength.

Before another training campaign, the next useful diagnostic would isolate
teacher quality at the deployment envelope on held-out positions, using solved
or independently adjudicated positions where practical. That is a proposed
follow-up, not work performed in this experiment or a demonstrated explanation
of its failure. This run does not isolate architecture capacity, insufficient
data, moving-teacher noise, root-only guidance, or optimizer choice as the cause.

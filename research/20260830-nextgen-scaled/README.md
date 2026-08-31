# 20260830 Scaled next-generation transition opponent

Status: complete — v4 promoted as user-facing default

## Idea

Scale the strongest evidence from the packaged explicit transition scorer while
spending deep-search compute only where it changes the teacher. The prior
generation showed a useful 55.75% arena result at unchanged search telemetry,
but also showed that 51/256 calibration roots changed action under depth 8 / 2M
nodes. This path expands the source-disjoint sample, measures that instability
on a fresh pool, and uses the disagreement set as the deep-label budget.

## Starting point

The incumbent supported opponent remains `pathfinder-v0.5.0-trained-evaluator`.
The packaged research candidate is
`pathfinder-action-transition-v3-xent`, SHA-256
`4f08a5a68057051e99c469aaf4a6e839885ebdcb167e6b82b076836c0b24b7f4`, with a
rules-authoritative Rust/WASM scorer and tactical-safe root ordering. The
canonical corpus contains 38,547 unique games; after excluding the prior 1,920
roots and the 10,000-root generation, 26,402 eligible 7×7 games remain.

## Protocol

1. Select 4,000 fresh, source-disjoint 7×7 roots with exactly 2,000
   Light-to-move and 2,000 Dark-to-move positions, keeping source games as the
   split unit.
2. Label every fresh root at depth 7 / 1,000,000 nodes / beam 32 with the
   tactical-safe Pathfinder teacher.
3. Deep-label a balanced calibration slice at depth 8 / 2,000,000 nodes / beam
   32. Compare actions and retain only changed-action roots for selective deep
   replacements; keep all deep rows outside training evaluation until the
   selection rule is frozen.
4. Train the packaged explicit scorer and at least one selective-depth blend,
   then gate on heldout top-1/top-3, unsafe selections, and teacher rank before
   any arena budget.
5. Run a larger paired arena against v0.5 at the same 2.8-second policy cap,
   alternating colors and auditing every replay. Promote only a candidate that
   clears the two-color, legality, tactical, and latency gates.

## What happened

The fresh 4,000-root pool is frozen: 3,183 train roots and 817 held-out roots,
with exactly 2,000 Light-to-move and 2,000 Dark-to-move roots before the source
split. The depth-8 calibration changed the teacher action on 39/256 roots
(15.23%; 31 Dark and 8 Light), so the selective corpus contains 14,000 rows:
the 4,000-root depth-7 pool plus the prior generation's 10,000 source-disjoint
rows, with 39 current roots replaced by their deeper labels. The
explicit cross-entropy scorer is the finalist: held-out top-1 is 31.35%, top-3
is 47.95%, mean teacher rank is 14.87, and unsafe selections are 0. The
rank-weighted and virtual-source variants were weaker (29.07% and 27.08%
held-out top-1), so they are retained as negative controls.

As a bounded infrastructure check, the previously evaluated contextual
one-game Lambda package was staged from the existing verified Linux `bootstrap`
(SHA-256 `19ac4451b9c7760816df2ff4fa8120594e7ba1d974839146f6d02c23349ec182`)
and paired with the exact event in `workspace/cloud-sanity/event.json`: one
seeded game, depth 7, one million nodes, beam 32, 2.8-second move deadline,
and a 20-ply maximum. Creation of the temporary `provided.al2023` function in
the configured `us-east-1` Region was denied by the project's managed service
control policy (`lambda:CreateFunction`); no function or other cloud resource
was created, and no invocation took place. This is an infrastructure blocker,
not a game or legality result. A fresh Linux cross-build was unavailable on the
macOS host, so the previously verified Linux artifact was reused unchanged.
The worker exercises the prior contextual evaluator, not the new
transition-policy v4 finalist.

The project-level Lambda path was subsequently enabled in the `pathagon`
profile's selected `us-east-2` Region. The v4-specific worker and model are
preserved under `workspace/lambda-v4-sanity/`; the temporary private function
`pathagon-transition-v4-sanity-20260830` ran the exact first-arena event
(seed `2026083002`, candidate Light, 80 plies, depth 7, 1M nodes, beam 32,
2.8-second deadline) successfully in 126.1 seconds. It returned a
contract-valid 47-ply Light path win and used 33 MB. The function is configured
as `provided.al2023`, x86_64, 2,048 MB, with a 900-second timeout. Synchronous
invocation required `--cli-read-timeout 1000` and `AWS_MAX_ATTEMPTS=1`; the
default retry/timeout behavior caused duplicate 120-second failures.
CloudWatch confirms 126,056 ms duration, 126,085 ms billed duration, and 33 MB
peak memory. The deployed ZIP's AWS `CodeSha256` matches the local
`function.zip` byte-for-byte. The function remains deployed temporarily for
comparison and cleanup review; no public endpoint was created.

The complete arena then ran as a 64-way fan-out through that same private
function, preserving the deterministic seed/index mapping (`seed + index`,
candidate Light on even indices) and the exact depth-7 / 1M-node / beam-32 /
2.8-second configuration. The merged 1,000-game result is 565 wins, 401
losses, and 34 draws (58.2% points). By color, the candidate scored 57.5% as
Light (277–202–21) and 58.9% as Dark (288–199–13); the 95% Wilson intervals
are 53.1–61.8% and 54.5–63.1%, respectively. Candidate mean search work was
238,889 nodes versus 237,247 for the incumbent (about 0.7% higher) with mean
completed depth 4.91 versus 4.95. The native replay audit covered all 46,604
plies and 10,859 captures with no legality, ownership, or capture mismatch;
966 games ended by path and 34 by the 80-ply bound.

The first local replay of seed `2026083002` and the Lambda result both ended
in a Light path win, but their timed searches diverged at ply 6 (local 39
plies, Lambda 47). This is expected timing/platform sensitivity under a hard
2.8-second deadline, so the cloud arena is the authoritative strength sample;
the parity check confirms the outcome and contract, not byte-for-byte search
telemetry. The audit found 914 unique action sequences (86 repeated groups),
which is legal but identifies opening entropy as the next useful data lever.

## Data and artifacts

Disposable roots, targets, calibration labels, checkpoints, Lambda package and
response, and arena logs live in `workspace/`. The cloud arena and run metadata
are in `workspace/arena-next-xent-lambda-1000.jsonl` and
`workspace/lambda-arena-1000-run.json`; its summary, native audit, and the
seed-`2026083002` local/Lambda comparison are kept beside them. The native
replay auditor in `audit/` replays each arena move
through the Rust rules and compares captures and turn ownership. The v4 model,
manifest, and browser copy are promoted under
`data/models/pathfinder-action-transition-v4-xent/` and
`apps/web/public/models/`; v4 is now the default browser Pathfinder model and
v3 remains available as the prior version.

## Project impact

The scaled generation produced a material, color-balanced strength gain while
holding the search envelope effectively constant. The v4 model is now a
versioned, hashed user-facing default with browser assets and a stable opponent
identity; v3 remains available as a prior version.

## Next decision

Promotion is complete for user-facing v4. Keep v3 and v0.5 available as prior
baselines and rollback controls. The next budget should increase opening
entropy (more randomized opening plies or a wider seed schedule) to reduce the
8.6% repeated-action-sequence rate, then deepen only roots where teacher
disagreement or model uncertainty predicts value; a richer state/action
representation remains the next model-family experiment.

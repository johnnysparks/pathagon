# 20260830 Browser long-horizon feedback

Status: completed

## Idea

Give the browser Pathfinder explicit 20-, 50-, and 100-ply horizon targets,
while reporting useful search checkpoints quickly enough that a human can wait
for a strong move or take the current incumbent. The feedback must not turn
the node-expansion loop into a message loop.

## Starting point

The shipped browser dial stopped at 12 ply and the Rust/WASM worker reported
only after an iterative-deepening pass. A live read-only replay of game
`0eaacb6f-56fa-4703-820c-bbc725bc4348` confirmed a 51-ply Light win against
`pathfinder-action-transition-v4-xent`; the old record had no search metadata.

## What happened

The browser envelope now exposes 2 through 12 ply plus 20, 50, and 100 ply.
Long targets receive progressively larger node caps and a narrow beam, but
remain bounded by the selected wall-clock limit. The result records requested
and completed depth separately, so a timed-out 100-ply target remains honest.

Rust search budgets now emit cumulative progress at the first node count at or
above each 10,000-position boundary, or after 500 ms of compute. The time
check is sampled every 256 nodes and the node threshold is checked cheaply on
each expansion. The worker forwards these events to the page and still emits
depth-boundary updates. Browser cancellation terminates the synchronous WASM
worker and recreates it, allowing the current best move to be played without
waiting for the requested horizon.

The local production browser check observed checkpoints at approximately
4,407 positions / 532 ms and 40,011 positions during one 12-ply run, then a
legal completed result. A cache-keyed JS/WASM load fixed a stale-tab mismatch
where an older transition-policy wrapper lacked the new callback method.

## Data and artifacts

No generated games, replay exports, or verbose logs are retained. The browser
checkpoint values above are small validation evidence; durable game metadata
is stored in the bounded JSON field added to `human_games`. The migration is
checked in but was not applied to the remote database during this task.

## Project impact

The browser UI, Rust search/WASM ABI, worker, loose human-game metadata
contract, and generated engine bundles now support long-horizon search
experiments. The existing 4-ply Pathfinder default remains the control. Search
telemetry includes model card, dials, config, per-move elapsed time, positions,
completed depth, table hits, exhaustion/interruption, and checkpoints.

## Next decision

Use the 20/50/100-ply controls to identify the user's practical “hard”
threshold before spending effort on large self-play runs. Promote any later
benchmark labels or fixtures only after they clear the repository's legality,
cost, and replay gates. Apply the pending D1 migration and deploy separately
when the live environment change is explicitly approved.

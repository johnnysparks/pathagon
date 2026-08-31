# 20260830 Browser long-horizon feedback

Status: completed

## Idea

Give the browser Pathfinder explicit 20–100-ply horizon targets and convenient
21/22/23-ply checkpoints, while reporting useful search checkpoints quickly
enough that a human can wait for a strong move or take the current incumbent.
The feedback must not turn the node-expansion loop into a message loop.

## Starting point

The shipped browser dial stopped at 12 ply and the Rust/WASM worker reported
only after an iterative-deepening pass. A live read-only replay of game
`0eaacb6f-56fa-4703-820c-bbc725bc4348` confirmed a 51-ply Light win against
`pathfinder-action-transition-v4-xent`; the old record had no search metadata.

## What happened

The browser envelope now exposes every integer depth from 2 through 100, so
21-, 22-, and 23-ply experiments do not require jumping between coarse
presets. Long targets receive progressively larger default node caps and a
narrow beam, but the browser can now override the cap with 250k, 500k, 1M,
2M, 5M, or 10M positions. The Rust runtime independently clamps browser
configs to the 10M hard ceiling. The result records requested and completed
depth separately, so a timed-out 100-ply target remains honest.

The wall-clock control now extends to 60 seconds. Per-move telemetry records
the selected node cap and whether search stopped at the position cap versus
the time cap, alongside the existing checkpoint stream.

The first 50-ply follow-up (`f0193fab-aa78-4a06-9e0e-5412f3575554`) reported
that the 2M-position budget was reached almost every time within 30 seconds.
The long-horizon presets therefore start at 5M positions for 50–99 ply and
10M for the 100-ply horizon; the explicit control still supports lower caps.

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
is stored in the bounded JSON field added to `human_games`. The human-game
metadata migration was applied to the remote database during the preceding
deployment. A direct remote-D1 lookup for the newly requested game
`2099d8b9-abca-4350-a10c-0a2e697c6d18` returned no row, matching the Worker's
404 response; no replay metadata could be recovered for it. The same lookup
for `f0193fab-aa78-4a06-9e0e-5412f3575554` also returned no row.

## Project impact

The browser UI, Rust search/WASM ABI, worker, loose human-game metadata
contract, and generated engine bundles now support long-horizon search
experiments. The existing 4-ply Pathfinder default remains the control. Search
telemetry includes model card, dials, config, per-move elapsed time, positions,
completed depth, table hits, exhaustion/interruption, and checkpoints.

## Next decision

Use the 20–100-ply controls to identify the user's practical “hard”
threshold before spending effort on large self-play runs. Promote any later
benchmark labels or fixtures only after they clear the repository's legality,
cost, and replay gates. Deploy the browser change separately when the live
environment update is explicitly approved.

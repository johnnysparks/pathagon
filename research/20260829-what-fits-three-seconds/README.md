# 20260829 What fits in three seconds?

Status: completed — no profile promoted

## Idea

Hypothesis: the current depth-4 / 2,000-node / beam-8 Pathfinder profile is a
useful frozen research baseline but is far below the compute users will accept
for a strong default opponent. A latency-calibrated profile targeting roughly a
few seconds per turn should unlock substantially deeper search and become the
product default if it produces real whole-game strength.

This is now the highest-priority path because the chosen compute envelope
changes the right operating point for every later evaluator, curriculum, and
sorter experiment.

## Starting point

The current browser, Rust CLI, and supported opponent manifest share depth 4,
2,000 nodes, and beam 8. A refreshed Node/WASM probe on one legal 22-ply
midgame position measured:

| profile | searched nodes | completed depth | elapsed |
| --- | ---: | ---: | ---: |
| depth 4 / 2,000 / beam 8 | 1,042 | 4 | 10.8 ms median |
| depth 6 / 250,000 / beam 24 | 46,859 | 6 | 278 ms median |
| depth 7 / 500,000 / beam 32 | 332,715 | 7 | 1.40 s |
| depth 8 / 2,000,000 / beam 48 | 1,344,167 | 8 | 6.16 s |

This single-position probe is not a product benchmark, but it is enough to
reject 2,000 nodes as a latency-calibrated default. It suggests that depth 7
and hundreds of thousands of nodes are plausible starting points on the
reference machine. Slower devices, high-branching positions, UI responsiveness,
and whole-game strength still need direct measurement.

The fixed suite is now promoted as
`data/fixtures/pathfinder-browser-suite-v1.jsonl` and contains ten 7×7
positions: empty and four-ply openings, placement midgame, capture-heavy,
human tactical, dense movement, high-branching movement, repetition context,
near-terminal, and dense placement. The browser harness reconstructs those
positions from the durable fixture and calls the shipped WASM asset through
the actual `pathagon_search_best_action_with_tactical_filter` entry point.
Reference hardware was Mac17,3 with an Apple M5, 10 logical cores, 24 GiB
memory, Chrome 151 at 1280×720 with device pixel ratio 2.

## Proposal

1. Define the user-facing target before strength testing: approximately 1–2.5
   seconds median and no more than 3 seconds p95 on agreed reference hardware,
   with a visible thinking state and a responsive/cancelable browser UI.
2. Build and promote a fixed benchmark set spanning opening, placement
   midgame, captures, movement phase, tactical fixtures, high branching,
   repetitions, and near-terminal positions. Measure native and actual
   browser/WASM execution.
3. Sweep depth 6–8, beam 16–48, and node caps from roughly 100,000 to 1,500,000.
   Preserve iterative-deepening fallback so a deadline returns the last fully
   completed depth.
4. Compare the latency-qualified profiles in paired arenas against the current
   2,000-node v0.5 baseline. Freeze evaluator weights so the first strength gain
   is attributable to compute.
5. If multi-second search blocks the main thread, move opponent search behind a
   Web Worker or equivalent responsive boundary before considering promotion.
6. Once a default profile is selected, make it the fixed product envelope for
   later learning experiments while retaining 2,000 nodes as a historical
   comparison profile.

## Promotion criteria

All gates must pass:

- On the fixed browser benchmark, median search time is 1–2.5 seconds and p95
  is at most 3 seconds on the agreed reference machine. Results on at least one
  slower-device profile are reported, with an explicit fallback policy.
- The browser remains responsive during search, can abandon obsolete requests,
  and always returns a legal move from the last completed iteration before its
  hard deadline.
- A final 400-game paired arena against the current 2,000-node v0.5 baseline
  reaches at least 55% game points with a positive win-loss margin in each
  color. Evaluator weights and openings are frozen for this comparison.
- Tactical, human-derived, parity, replay, and legality suites remain fully
  passing; immediate wins and forced blocks do not regress.
- The selected profile, reference hardware, benchmark positions, latency
  distribution, node use, bundle impact, and fallback behavior are durable and
  reproducible.

Raw node count or depth alone is not promotion evidence.

## What happened

The browser screen confirmed that the movement phase, not placement, sets the
tail. All tested calls returned legal moves. The deadline-enabled screens were:

| surface / profile | measured searches | median | p95 | max | completed depth | mean nodes | illegal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| browser / v0.5 control, 2,000 nodes | 10 | 42 ms | 148 ms | 179 ms | 2–4 | — | 0 |
| browser / depth 5, 40k nodes, beam 16 | 10 | 505 ms | 1,992 ms | 2,017 ms | 4–5 | — | 0 |
| browser / depth 6, 50k nodes, beam 16, 2.8 s deadline | 20 | 427 ms | 1,135 ms | 1,138 ms | 4–6 | 41,342 | 0 |
| browser / depth 6, 100k nodes, beam 16, 2.8 s deadline | 20 | 835 ms | 1,935 ms | 1,952 ms | 4–6 | 67,863 | 0 |
| browser / depth 7, 500k nodes, beam 32, 2.8 s deadline | 10 | 2,800 ms | 2,800 ms | 2,800 ms | 3–6 | 208,279 | 0 |
| browser / depth 5, 40k nodes, beam 16, 1.8 s fallback | 20 | 250 ms | 1,039 ms | 1,052 ms | 4–5 | 25,717 | 0 |

The native cross-check used the same ten positions and weights. Its mean/max
times were 68/221 ms for control, 1,105/2,761 ms for depth 5 at 40k nodes,
1,222/3,165 ms for depth 5 at 50k nodes, 1,852/4,302 ms for depth 6 at 50k
nodes, and 2,234/4,386 ms for depth 6 at 100k nodes. This makes the browser
tail credible rather than a harness-only artifact. No physical slower device
was available in the browser capability set; the 1.8-second depth-5 profile
is therefore a measured lower-budget fallback on the reference machine, not a
claim about a specific slower handset.

The depth-6/100k profile is the closest latency fit: in the latest durable
fixture run its median was 835 ms and p95 was 1,935 ms, while an earlier run
reached 2,800 ms p95 under the same hard deadline. The depth-7 profile spent
the full deadline on every position, so it was rejected. The synchronous
depth-6/100k screen blocked the browser event loop for up to 1.972 seconds;
the Worker harness completed the same class of search in 188 ms while keeping
10 ms timer ticks alive with a 12.1 ms maximum gap. The Worker cancellation
screen resolved an obsolete request to null and returned a legal move for the
replacement request.

The final paired arena used the exact depth-6/100k/beam-16 candidate with a
2,800 ms native deadline against the frozen depth-4/2k/beam-8 v0.5 baseline.
Evaluator weights, colors, and two randomized opening plies were frozen; the
run used seed 2026082903, 400 games, a 60-ply cap, and ten workers. It scored
178 wins, 179 losses, and 43 draws: 199.5/400 game points (49.9%), below the
55% gate. By candidate color, light scored 84–88–28 (margin −4) and dark
scored 94–91–15 (margin +3), so the required positive margin in each color
also failed. The run produced 15,097 plies and 412,788,991 search nodes in
758.868 seconds.

Representative records reviewed were game 001 (candidate light win, 33
plies), game 002 (candidate dark loss, 29 plies), and game 382 (candidate dark
draw at the 60-ply cap). All 400 JSONL records replayed successfully through
the web validator, with no illegal actions or capture mismatches.

## Failures and limits

The first deadline implementation used Rust's standard `Instant::now()` and
failed in the browser with a WASM `RuntimeError: unreachable`; the final
implementation uses the browser-compatible clock and is covered by a Rust
deadline test plus the browser benchmark. The full app dev preview could not
start because the installed Miniflare binary only supports compatibility dates
through 2026-05-22 while the project configuration requests 2026-08-28, so
Worker responsiveness and cancellation were verified with a standalone Vite
harness against the same app modules and generated engine assets. An
unbounded 120-ply native screen was stopped after it proved impractical for a
research iteration; the final arena is the bounded, reproducible 60-ply run
reported above. No direct CPU-throttled handset profile was available, so the
fallback timing is explicitly not presented as a physical slower-device
measurement.

## Data and artifacts

Preserve:

- A compact versioned benchmark-position set under `data/` if it is replayable
  and broadly reusable, plus the native/browser benchmark harness.
- Generic deadline/cancellation support, worker integration, deterministic
  iterative-deepening behavior, and focused tests if implemented.
- The selected profile, aggregate arenas, latency distributions, reference
  hardware description, and representative game review.

Discard or keep ignored in `workspace/`:

- Raw timing logs, browser profiles, sweep tables, flamegraphs, repeated game
  exports, rejected configurations, and device-specific temporary traces.
- Any benchmark roots that duplicate durable fixtures without adding value.

The compact benchmark positions are promoted into
`data/fixtures/pathfinder-browser-suite-v1.jsonl` and covered by
`apps/web/tests/search-benchmark-fixture.test.ts`. Raw timing logs, browser
profiles, the native timing harness, rejected profiles, the 400-game JSONL,
and device-specific traces remain ignored under this path's `workspace/`.
The reusable implementation changes are the Rust deadline export, checked-in
WASM regeneration, Worker client/worker, cancellation behavior, and focused
tests; no rejected search profile was promoted.

The generated bundle size changed from 389,092 to 399,829 bytes for the normal
WASM asset (+10,737), from 14,687 to 15,973 bytes for its JS glue (+1,286),
from 21,778,903 to 21,789,480 bytes for inference WASM (+10,577), and from
19,259 to 20,476 bytes for inference JS glue (+1,217).

## Project impact

Success would redefine the supported Pathfinder default around the intended
few-seconds-per-turn experience and give every subsequent learning experiment a
realistic compute target. Failure would identify the actual browser or search
bottleneck rather than silently optimizing around an arbitrary 2,000-node cap.

The path changed the browser execution boundary and regenerated the WASM
assets, but intentionally did not change the supported opponent, evaluator,
default search envelope, or opponent manifest. The latency work is reusable;
the strength result says a deeper deadline-capped profile is not yet a safe
product promotion.

## Kick-off prompt

> Execute the research brief in
> `research/20260829-what-fits-three-seconds/README.md`. Treat a few seconds per
> turn as the intended default experience, define the reference hardware and
> latency gate first, then benchmark actual browser/WASM search across a fixed
> position suite and run paired strength screens for only the latency-qualified
> profiles. Preserve current v0.5 evaluator weights for the causal comparison
> and retain depth-4 / 2,000-node / beam-8 as a historical baseline, not a
> permanent constraint. You have explicit liberty to rename configurations,
> identities, files, or UI concepts and to make justified edits outside this
> research folder under `pathagon/`, `apps/`, `data/`, `docs/`, and `scripts/`.
> Update every affected reference, test, manifest, generated WASM asset, and
> document. Add a worker/deadline boundary if required for responsiveness. Keep
> raw sweeps, profiles, rejected arenas, and logs in this path's ignored
> `workspace/`; promote only a reproducible profile that passes every latency,
> strength, legality, and UX gate in this README.

## Decision

Do not promote a three-second Pathfinder profile. The depth-6/100k profile was
responsive and legal behind the Worker boundary, but missed the preferred
median window in repeated runs and failed the final strength gate at 49.9%
game points with a negative light-color margin. Keep the historical v0.5
depth-4/2k/beam-8 default and the new deadline/Worker infrastructure. A future
attempt should first improve move quality under the same deadline, then rerun
the durable suite and the 400-game paired arena; it should not begin another
evaluator or learned-runtime promotion attempt by simply increasing the node
cap.

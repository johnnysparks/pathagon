# 20260831 Pathfinder full shallow coverage

Status: `idea`

## Idea

Test whether Pathfinder benefits more from nearly complete child coverage at
three or four plies than from spending the same or greater budget on deeper
search with the current `b256` cap. The working hypothesis is that the legal
action space is wide enough on a meaningful fraction of 7×7 positions that
`b256` silently omits viable moves, and that this omission matters more than
the extra horizon on those positions. Node visits and wall time are secondary
to the strength of the shallow, high-coverage search, but both must remain
visible so any gain can be priced honestly.

This is a follow-up to
[`20260831-pathfinder-search-strategies`](../20260831-pathfinder-search-strategies/),
which established the corrected ordinary Pathfinder controls and promoted
depth 5 / beam 256 / 256k nodes as the current strength envelope. That path
did not isolate beams above 256 at shallow depth or measure how often the beam
actually truncates the legal action set.

## Starting point

The native Rust engine is authoritative on `pathagon-rules-v1`, 7×7 boards,
and 14 reserves per player. Ordinary Pathfinder is iterative-deepening
alpha-beta with deterministic move ordering, transposition storage, a
recursive `beam_width` child cap, and a `max_nodes` ceiling. The current
strength control is:

```text
depth 5 / beam 256 / 256,000 nodes
```

The cheaper cost control remains depth 4 / beam 256 / 32,000 nodes. This path
uses unfiltered, unbooked, non-neural Pathfinder for every side, preserving
the corrected root-bound implementation from the parent study. It does not
change the shipped Transition v4 identity or the tactical-safe product line.

## Research questions

1. At depth 3 or 4, does widening the recursive beam from 256 to 384, 512,
   768, 1024, or 4096 improve paired game points against the current control?
2. Is there a coverage knee below the nominal full-set ceiling, or does the
   candidate need a beam large enough to cover almost every legal action?
3. At matched node ceilings, does a wider shallow profile outperform depth 5 /
   beam 256, and how often does it complete its configured depth?
4. Do any gains survive a held-out seed schedule after accounting for the
   candidate's node and latency cost?

The `4096` beam is a practical full-set sentinel for this board. A 7×7 action
is encoded by a source/destination pair and the model's maximum action tensor
has 2,401 slots, so the run must still report observed legal-action counts
rather than assuming that the sentinel is mathematically exhaustive for every
future rules configuration.

## Protocol

The profile catalog is [`profiles.json`](profiles.json). It is consumed by
the reusable arena runner in the parent path:
[`../20260831-pathfinder-search-strategies/scripts/run_pathfinder_search_strategies.py`](../20260831-pathfinder-search-strategies/scripts/run_pathfinder_search_strategies.py).
The new catalog keeps the first screen small enough to interpret:

- current strength control: depth 5 / beam 256 / 256k nodes;
- shallow fixed-budget lane: depth 3 at beams 256, 384, 512, 768, 1024,
  and 4096, all at 256k nodes;
- one-step-deeper fixed-budget lane: depth 4 at beams 256, 512, 1024, and
  4096, all at 256k nodes;
- budget-relaxed confirmation lane: depth 3 / beam 512 and depth 4 / beam
  1024 at 1M nodes, to test whether the shallow coverage signal was merely
  node-starved.

The initial screen should use 100 games per profile against the strength
control, paired colors, disjoint two-ply randomized openings, a 120-ply game
cap, and one worker for stable cost accounting. All matches pass
`--no-tactical-root-filter` and `--opponent deep-search`; learned weights,
books, proof hooks, and neural sorters remain out of scope. The corrected
parent binary must be rebuilt before running the arena.

```sh
cargo build --release --manifest-path pathagon/engine-rs/Cargo.toml --bin pathagon-selfplay
python3 research/20260831-pathfinder-search-strategies/scripts/run_pathfinder_search_strategies.py \
  --profiles research/20260831-pathfinder-full-shallow-coverage/profiles.json \
  --out-dir research/20260831-pathfinder-full-shallow-coverage/workspace/arena \
  --games 100 --workers 1
```

Use `--games 2 --profile pf-d3-b512-n256k --profile
pf-d5-b256-n256k-control` for a smoke run. Use the parent analyzer on the
resulting arena directory for the initial strength/cost summary; this path
still needs a coverage-specific replay analyzer before any conclusion is
promotion-grade.

## Measurements and gates

Every archive must first pass the existing replay/legality checks. The
coverage-specific analysis should replay each move through the Rust rules and
record, at minimum:

- legal action count at every decision;
- `min(beam_width, legal_action_count) / legal_action_count` as nominal child
  coverage;
- the fraction of decisions where `legal_action_count > beam_width`;
- separate placement and relocation distributions;
- configured depth versus completed depth, node-ceiling saturation, table hits,
  and wall seconds;
- results by color, termination reason, and game points.

The key coverage metric is descriptive, not a claim that alpha-beta evaluates
every child equally: move ordering and cutoffs still determine which expanded
children influence the result. A beam sentinel can therefore have 100%
nominal legal-set coverage while the node budget prevents full recursive
exploration. The report must retain both metrics.

The primary comparison is paired game points against depth 5 / beam 256 /
256k. A candidate is interesting if it improves both colors on held-out seeds
without a large increase in node cost or an unacceptable completed-depth
drop. The first screen is exploratory; no candidate is promoted from it.
Finalists require at least 400 paired held-out games, representative replay
inspection, browser/WASM responsiveness checks, and a Rust implementation and
test decision if the traversal or supported opponent is changed.

## What happened

The plumbing smoke completed with two paired games for depth 3 / beam 512 /
256k against the depth 5 / beam 256 / 256k control. Both colors were legal and
the archive contained the expected node and completed-depth telemetry. The
candidate lost 0–2, using 97,457.5 nodes per game on average versus
2,939,205 for the control. This is not evidence for or against the hypothesis:
the sample is intentionally too small, and the smoke used the already-present
release binary while another release inference test held Cargo's build lock.

The full coverage curve, paired strength results, cost trade-offs, failed
profiles, and any protocol corrections will be recorded here. If the
hypothesis fails, retain the negative result rather than silently merging it
with the parent search-strategy evidence.

## Data and artifacts

The catalog and this protocol remain in Git. Game archives, reports, coverage
tables, logs, and temporary binaries belong under this path's ignored
`workspace/`. No repeated replay export, optimizer state, or implementation-
shaped tensor should be committed. Reusable fixtures or labels must be
promoted through [`../../data/`](../../data/) with a strict version and
provenance rather than copied into this path.

## Project impact

This setup makes no production change. It is intended to determine whether a
coverage-first shallow envelope is worth implementing or promoting. Any
successful supported opponent must retain a versioned rollback/control, be
ported or represented in Rust/WASM, pass focused search and replay tests, and
survive the normal strength, cost, and representative-game gates.

## Next decision

Run the smoke command, add the Rust-backed coverage replay report, then run the
fixed-budget screen. Continue only with profiles that show a repeatable
strength or coverage advantage; otherwise retire the hypothesis as an
interesting but unsupported explanation for the b256 boundary.

# 20260831 Pathfinder search strategies

Status: `results`

## Idea

Measure the strength and efficiency frontier of ordinary Pathfinder against
ordinary Pathfinder. The immediate question is whether search depth, beam
width, or a larger node ceiling is the more useful way to spend compute when
the opposing player is allowed to use a different envelope.

This fills a gap in the existing research. Earlier paths tested a few deeper
profiles against a fixed control, but did not run a symmetric, reusable
Pathfinder profile campaign. A profile may use more or fewer nodes than its
opponent; the match must preserve that asymmetry in the records so strength
can later be compared with cost.

## Starting point

The native Rust engine is authoritative. Its ordinary search is iterative
deepening alpha-beta with deterministic move ordering, a recursive beam cap,
transposition-table storage, a node ceiling, and an optional wall-clock
deadline. `SearchConfig` exposes `depth`, `beam_width`, `max_nodes`, and
evaluator weights; the self-play agent adds an optional per-move deadline. The
current research control is depth 4 / beam 8 / 2,000 nodes with the default
heuristic weights.

The existing CLI already supports independent configurations: `--depth`,
`--beam`, and `--nodes` configure the opponent, while
`--candidate-depth`, `--candidate-beam`, and `--candidate-nodes` configure the
champion. Every move record stores the actual nodes searched, completed depth,
and transposition-table hits; every game stores both agent specifications.

This path deliberately uses unfiltered, unbooked, non-neural Pathfinder for
the first causal screen. The promoted tactical filter, learned weights,
sorters, proof hooks, and opening books would add separate variables.

## Protocol

The native runner is wrapped by
[`scripts/run_pathfinder_search_strategies.py`](scripts/run_pathfinder_search_strategies.py).
The profile catalog is [`profiles.json`](profiles.json). The default screen
plays every non-control profile against the control, using one-variable lanes
where possible plus explicit wide-beam and timed interaction profiles:

- depth: 2, 3, 4, 5, 6 at beam 8 / 2,000 nodes;
- beam: 2, 4, 8, 16, 32 at depth 4 / 2,000 nodes;
- node ceiling: 500, 2,000, 8,000, 32,000 at depth 4 / beam 8;
- wide shallow: depth 3/4, beam 150, with 32,000–200,000 nodes;
- timed: depth 3/4, wide beams, and 1–2 second per-move deadlines;
- Full Power: depth 4, beam 256, 1,000,000 nodes, and a 5-second deadline.
- 64k beam boundary: depth 4, beam 150 versus beam 256, both with 64,000
  nodes and no deadline.
- fixed-32k beam curve: depth 4, beams 32, 64, 150, and 256, all with
  32,000 nodes.
- depth-vs-budget probe: depth 5 / beam 150 / 256,000 nodes with no deadline.

The Full Power beam is set to 256 as the explicit interpretation of “wider
beam” for this campaign. Its node ceiling and time ceiling are both recorded;
the first limit reached determines the actual search cost on each move.

The control appears in more than one catalog group only conceptually; the
runner de-duplicates it. `--round-robin` runs every unordered pair in the
catalog, again with paired colors. The first screen should use 100 games per
pairing, 7×7 boards, 14 reserves per player, two randomized opening plies,
and a 120-ply cap. The runner offsets seeds by pairing, so disjoint campaigns
can be assigned disjoint base seeds.

All matches pass `--no-tactical-root-filter` and use `--opponent deep-search`.
That makes both sides the same ordinary Rust search implementation while
allowing independent depth, beam, node, and time settings. A profile's node
ceiling and deadline are both active; whichever is reached first stops that
side's iterative-deepening search. “Max plies” in a profile means search depth;
the match-level `maxPlies` is only the game termination cap.

The special tournament mode is:

```sh
python3 research/20260831-pathfinder-search-strategies/scripts/run_pathfinder_search_strategies.py \
  --tournament --losses-to-eliminate 5 --games 40 --workers 1
```

Each round deterministically shuffles the active profiles, pairs adjacent
profiles, and gives a matchup loss to the lower game-point scorer after the
paired-color duel. Ties give neither profile a loss; an odd profile receives a
bye. A profile leaves after five matchup losses. Raw game losses and costs are
still retained in each duel report, so the elimination bracket is only a
selection mechanism, not the complete statistical result.

“Depth-first versus breadth-first” needs care here. The current recursive
alpha-beta implementation is depth-first; `beam_width` limits how many
ordered children are expanded at each node, but it is not a queue-based
breadth-first search. If the parameter sweep shows a useful frontier, the next
phase can add an explicit Rust `TraversalStrategy` (with separate correctness
and parity tests) and compare true depth-first, breadth-first, and beam-search
implementations under the same evaluator and replay protocol.

## Measurements and gates

The campaign report derives per-profile metrics from the game archives:

- wins, losses, draws, game points, and results by color;
- total and mean nodes per decision, per game, and per game point;
- completed-depth distribution and node-ceiling saturation;
- transposition-table hits and wall-clock seconds for each subprocess;
- the exact profile and engine configuration used for every pairing.

Nodes are the primary efficiency measure because the current game contract
records them per move for each side. Wall time is retained as a machine- and
worker-dependent secondary measure. The two players do not share a node
budget, and no result should be normalized by silently treating their
ceilings as equal.

Before interpreting a result, every archive must replay successfully with no
illegal actions, capture mismatches, missing moves, or malformed agent
specifications. A candidate that looks strong only because it wins one color,
hits the ply cap unusually often, or spends materially more nodes must remain
an efficiency tradeoff rather than an automatic promotion.

The held-out follow-up should use disjoint seeds and at least 400 paired games
for finalists. A promotion would additionally require the normal tactical,
replay, browser-cost, and representative-game review gates; this research
path does not promote a new opponent from a single screen.

To run the first screen:

```sh
cargo build --release --manifest-path pathagon/engine-rs/Cargo.toml --bin pathagon-selfplay
python3 research/20260831-pathfinder-search-strategies/scripts/run_pathfinder_search_strategies.py \
  --games 100 --workers 1
```

For a cheaper smoke or a focused lane, use `--games 2 --profile
pf-d2-b8-n2k`, `--group depth`, `--group beam`, or `--group nodes`. After
screening, `--round-robin` runs all selected profiles against each other.

## What happened

The release Rust arena completed 5,280 games across 80 reports with balanced
colors and two randomized opening plies. Every archive passed the analyzer's
structural checks: valid contract, winner/color fields, candidate identity,
and move-player fields. The durable analysis command is
[`scripts/analyze_pathfinder_search_strategies.py`](scripts/analyze_pathfinder_search_strategies.py);
the generated ranking and confidence intervals are in the ignored
`workspace/analysis-all-completed.{json,md}` files.

The 100-game fixed-control screen established the broad shape. The strongest
screen result was timed depth 3 / beam 150 / 500k / 1 second at 84–16, followed
by depth 3 / beam 150 / 200k at 82–18, depth 4 / beam 150 / 32k at 81–19, and
depth 3 / beam 150 / 32k at 78–21–1. Full Power (depth 4 / beam 256 / 1M /
5 seconds) beat control 64–36, but consumed about 686,560 nodes/game, roughly
29 times the control's 23,712 nodes/game.

The randomized five-loss selector then excluded Full Power at the user's
request and ran 18 rounds / 1,240 games among the remaining six profiles.
Depth 4 / beam 150 / 32k was the only survivor with four matchup losses;
every other profile reached five. A final 12-round / 840-game five-profile
selector added depth 4 / beam 256 / 32k and made it the only survivor with
zero matchup losses; the other four profiles each reached five. The earlier
partial bracket containing Full Power is retained as interrupted evidence but
is excluded from the final selector ranking.

The depth-5 probe did not displace the champion. Against the control it scored
65–35 over 100 games, completing depth 5 on 1,194 of 1,493 searched decisions
and saturating its 256k node ceiling on 299 decisions. In the paired-color
200-game head-to-head, depth 4 / beam 256 / 32k beat depth 5 / beam 150 /
256k by 141–59 with no draws. The depth-5 profile consumed about 2.80M
nodes/game versus 405.6k for the champion, so extra depth was not a good use
of this envelope despite often reaching the depth limit.

The targeted node and beam checks sharpened the tuning rule. At fixed depth 4
and 32k nodes, the control screen rose from 62% at beam 32 to 69% at beam 64,
81% at beam 150, and 76.5% at beam 256. Across 400 direct games including
the final selector, beam 256 / 32k beat beam 150 / 32k 55.9% to 44.1% in game
points, with only about 2–3% more nodes/game. At 64k nodes, beam 150 beat beam
256 by 52–48 over 100 games, so the wider-beam edge is not monotonic once the
node cap changes.

Node scaling is even clearer at beam 256: 32k beat 64k by 103.5–96.5 over 200
games, including seven draws, while using about 54.5% fewer nodes/game. The
same node increase at beam 150 was effectively even (98–102) over 200 games.
The current practical champion is therefore depth 4 / beam 256 / 32k, with
depth 4 / beam 150 / 32k as a useful lower-cost control; larger node ceilings
are not automatically stronger.

## Empirical tuning function

This is the current best basic-Pathfinder policy, not a mathematical optimum
over every possible engine configuration:

| Available envelope | Recommended search | Evidence and dropoff |
|---|---|---|
| Around 500 nodes | depth 4 / beam 8 | The 500-node profile is weak (32% versus control); do not spend depth beyond what the node cap can finish. |
| Around 2k nodes | depth 4 / beam 32 | Beam 32 scored 62% versus control, ahead of beam 2 at 56%, beam 8 at 40%, beam 16 at 46.5%, and beam 4 at 45%; the curve is non-monotonic because truncation changes the ordered search frontier. |
| Around 32k nodes | depth 4 / beam 256; retain beam 150 as the cost-control | Across 400 direct games, beam 256 led beam 150 by 55.9%–44.1% in game points at only ~2–3% extra nodes/game. The fixed-control beam curve was 62%, 69%, 81%, and 76.5% for beams 32, 64, 150, and 256, so direct validation matters near the top. |
| Around 64k nodes | usually keep depth 4 / beam 256 / 32k; do not assume the extra 32k helps | At beam 256, 32k beat 64k 103.5–96.5 over 200 games with ~54.5% lower cost. At the exact 64k beam-150/256 comparison, beam 150 was 52–48. |
| 200k–500k or a 1–2s latency budget | benchmark depth 3 / beam 150, but prefer depth 4 / beam 256 / 32k for general play | Timed depth 3 / beam 150 reached 84% versus control, but lost the direct selector to the depth-4 wide-beam family; the timed profile is a useful latency-specific alternative, not the general winner. |
| Around 256k nodes with no deadline | do not raise depth to 5 by default | Depth 5 / beam 150 scored 59/200 game points against the depth-4 / beam-256 / 32k champion while costing ~6.9× as many nodes/game. |
| 1M nodes / 5s | do not use as a default | Full Power was only 64% versus control and was removed from the competition for compute efficiency. |

The lever-level pattern is: widen beam aggressively through the 32k envelope,
with beam 256 currently best in direct play, but expect a non-monotonic curve
when the node cap changes. Raise depth to 4 only when
the beam/node envelope can complete it; deeper depth at beam 8 saturated over
90% of decisions in the depth-5/6 tests and produced no reliable gain.
The new depth-5 / beam-150 / 256k probe reached depth 5 on about 73% of
searched decisions, but still lost the direct comparison badly at roughly
6.9× the champion's node cost/game; this is another sign that “reaches the
depth limit” does not by itself imply a stronger search.
Increase nodes from 500 through 32k when beam is narrow, but treat 32k as the
current wide-beam knee; beyond it, validate a direct matchup before paying
for more search. Wall-clock deadlines are secondary because they change
completed-depth distributions and are hardware-dependent.

## Data and artifacts

The durable artifacts are the narrative protocol, the small profile catalog,
and the dependency-light runner. Generated game archives and reports are
ignored under `workspace/`. Reusable replay fixtures or labels must be
promoted into a strict versioned location under `data/` rather than copied
into this experiment.

## Project impact

This experiment does not change production manifests, browser behavior, or
canonical game data. It does add ordinary-search deadline support and the
research runner/analyzer. The research winner is currently `d4/b256/32k`, with
`d4/b150/32k` retained as a lower-cost control, but it is not
promoted to a user-facing default from this path alone: normal tactical,
legality/replay, browser-cost, and representative-game gates still need to be
run. Full Power remains a measured rollback/control profile in the catalog,
not a tournament entrant. The depth-5 / beam-150 / 256k profile is retained
as a measured negative depth-vs-budget result, not a tournament entrant. If a
true alternate traversal is added, it must be
implemented and tested under `pathagon/engine-rs`; the current deadline path
is still the same depth-first alpha-beta traversal.

## Next decision

Keep `d4/b256/32k` as the current research champion, `d4/b150/32k` as the
lower-cost comparison, and `d4/b8/2k` as the control. Before promotion, run
the normal supported-agent gates and inspect
representative games. For future budget changes, use the piecewise policy
above and add direct, paired-color tests around the proposed knee; do not
assume that more depth, nodes, beam, or seconds is monotonically stronger.
The new `d5/b150/256k` profile was added to the catalog and screened, but the
direct result does not justify adding it to the elimination bracket.
Do not call beam width “breadth-first” until a real breadth-first Rust
traversal is implemented and separately validated.

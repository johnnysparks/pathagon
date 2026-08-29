# 20260828 Luna vs Pathfinder depth boundary

Status: `complete`

## Idea

Measure whether Luna—the GPT-5.6 general player actively selecting moves—can
beat the specialist Pathfinder over a meaningful sample, then increase
Pathfinder's search depth/budget until it becomes the strong opponent. The
experiment must distinguish Luna's explicitly chosen strategic procedure from
any learned checkpoint or the existing Lunatic baseline.

## Starting point

The opponent is the current filter-aware Pathfinder configuration mirrored by
the research rules adapter: 7×7 board, 14 reserves, two randomized opening
plies, and a 160-ply cap. Its promoted evaluator weights are
`path=241, material=112, capture=887, structure=40, threat=154, edge=74`.
The initial screen starts at depth 2, beam 8, and a 1,000-node ceiling. The
current production Rust implementation remains the authority; this runner
uses a dependency-light Python mirror so Luna's move-selection policy can be
executed in bulk and every game can be archived locally.

Luna's policy is a transparent distillation of the GPT player's reasoning:
explicit own wins and opponent-win blocks, repetition avoidance, capture
awareness, and broader whole-board connection/structure scoring with a wider
adversarial beam. It is not presented as an independently trained agent.

## What happened

The tracked runner is
[`scripts/run_luna_vs_pathfinder.py`](scripts/run_luna_vs_pathfinder.py).
Every screen uses paired colors, fixed seeds, complete move records, and a
summary in the ignored `workspace/`. A 100-game screen is the minimum evidence
for each depth. Luna remains ahead when its game points exceed 50%; the first
screen where Pathfinder takes a majority of points is the strength boundary.

The corrected 100-game screen completed with Luna 21–79–0 against Pathfinder
(21.0–79.0 points). Luna was Light in 50 games, winning 12 and losing 38;
Pathfinder was Light in 50 games, winning 41 and losing 9. All 100 games
ended by connection, with 100 unique seeds and non-empty replays. The run
took 848.351214 seconds using eight workers.

The strength boundary is therefore the initial Pathfinder screen: depth 2,
beam 8, and a 1,000-node budget. Because Luna did not retain a majority of
points at this screen, no higher-depth Pathfinder screens were run under the
experiment's escalation rule.

## Data and artifacts

Generated game archives and reports belong in this path's ignored
[`workspace/`](workspace/). No checkpoints or repeated replay exports should
be committed. If a game, fixture, or label proves reusable beyond this
experiment, it must be promoted through the strict versioned `data/` path
instead of copied into this research directory.

## Project impact

This experiment does not promote a new opponent or alter production Rust,
browser behavior, or canonical game data. Its measured result says the
GPT-guided strategic procedure, as operationalized here, is well below the
current filter-aware Pathfinder control even at its initial shallow envelope.
The immediate roadmap focus should be improving Luna's tactical and
connection-preservation policy before spending more Pathfinder compute.

## Failures and limits

The comparison is not a claim that a short scripted distillation equals the
full internal reasoning process of GPT-5.6. It is a reproducible operational
definition of Luna's move selection for 100-game screens. A 100-game result is
stronger than an anecdotal match but still sensitive to openings and the
chosen compute envelope; color balance and fixed seeds are therefore required.

The first full report had a summary-only agent-ID accounting defect and was
discarded. The runner was then corrected and its Pathfinder mirror was brought
into line with the Rust control's full safe-root iterative loop and table/killer
ordering. The saved report is from that corrected run. Rust library tests also
pass (43/43); the runner remains a dependency-light research mirror rather
than a claim that Python executed the production binary directly.

## Next decision

Stop escalation at the initial Pathfinder-majority screen. The next useful
experiment is a Luna-policy improvement or a Rust-native Luna harness, followed
by a fresh 100-game screen; deeper Pathfinder budgets are not informative until
Luna can win this baseline.

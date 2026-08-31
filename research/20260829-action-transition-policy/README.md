# 20260829 Action-transition policy

Status: validated research candidate · promotion deferred

## Idea

The super-deep and turn-balanced evaluator paths established that the labels
are now materially better, but a phase-conditioned six-weight evaluator still
cannot absorb the signal. This path changes the representation instead of
simply adding more copies of the same features: a compact nonlinear action
policy scores each legal afterstate using the six existing transition values,
capture/tactical flags, and action geometry. Pathfinder remains the decision
authority; the policy only supplies root ordering inside the tactical-safe set.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator` with weights
`path=241, material=112, capture=887, structure=40, threat=154, edge=74` and
the tactical-safe root filter. The source-disjoint turn-balanced corpus has
1,920 roots and one-million-node/depth-7 labels (960 Light-to-move and 960
Dark-to-move). The labels are not equally deep: 448 completed depth 7, 1,188
depth 6, 271 depth 5, and 13 depth 4; 1,472 exhausted the one-million-node
budget. The six-feature contextual family tied or lost its held-out teacher
gate even after this scale-up, so this is a representation test rather than
another linear-weight sweep.

## Protocol

Train a small two-hidden-layer tanh scorer on the training roots only. Each
legal action is represented by the six unit evaluator afterstate features,
capture count, immediate-win/safety flags, placement-vs-relocation kind,
source/destination geometry, player-oriented progress, center distance, and
edge/corner indicators. Standardization statistics are fit on training action
rows only. The target is the million-node teacher's selected action; source
games remain the split unit.

The Rust research agent must preserve the incumbent's legal-action generation,
tactical-safe root filter, iterative-deepening search, and deadline. It may
only reorder the safe root list. Selection gates are held-out teacher top-1,
legality/tactical audits, then paired color-balanced arenas at the same
1M-node/depth-7/2.8-second envelope. No candidate is promoted from a screen.

## What happened

The first implementation is now trained and screened. The six-feature linear
baseline selects 104/385 held-out teacher actions (27.0%). With the same
two-layer tanh capacity and training schedule, the 20-feature action-only
encoding selects 126/385 (32.7%), while adding state context (piece/reserve
counts, mobility, phase, ply, and last-capture context) selects 128/385
(33.2%). This isolates a useful but modest state-representation gain; the
larger improvement comes from nonlinear action/transition modeling. A virtual
off-board source encoding for placements selects 119/385 (30.9%), so a unified
move syntax is not automatically easier to learn. None of these offline
models selected an unsafe action after the tactical-safe pool was applied.

The explicit state-aware model then scored 13–6–1 in a 20-game, paired
color-balanced screen at depth 7 / 1,000,000 nodes / 2.8 seconds (67.5% game
points; 6–3–1 as Light and 7–3 as Dark). All 769 plies replayed legally. This
clears the pre-gate, so the independent 120-game arena is the active decision
run. It completed at 61–42–17 (57.9% game points), with positive margins in
both colors: 32–19–9 as Light and 29–23–8 as Dark. Candidate and incumbent
search telemetry remained effectively matched (about 236k versus 238k nodes per
move and depth 4.92 versus 4.95), and all 5,231 plies replayed legally. This
is strong research evidence that action-specific ordering can improve a fixed
finite search budget; it is still not a supported v0.6 promotion without the
longer untouched gate and broader regression review.

To test whether that result was a lucky seed, the same model and configuration
were run again on 120 new paired games after adding a safety-only immediate-win
ordering guard (winning roots are always searched first; the legal-action set
and tactical-safe filter are unchanged). The repeat was 68–35–17 (63.8% game
points), with 33–17–10 as Light and 35–18–7 as Dark. Across both independent
120-game arenas the candidate is 129–77–34 (60.8% game points), 65–36–19 as
Light and 64–41–15 as Dark, over 10,614 legal plies. Candidate and incumbent
telemetry stayed matched in the combined runs (269,578 versus 269,937 mean
nodes per move; depth 5.00 versus 5.02). The second run therefore supports a
repeatable positive ordering signal, while still leaving the longer promotion
arena and broader regression review outstanding.

The untouched 400-game promotion gate then scored 191–133–76 from the
candidate's perspective (57.25% game points) over 18,527 plies. The color split
was 91–70–39 as Light (55.25% points) and 100–63–37 as Dark (59.25% points).
The combined 640-game evidence is 320–210–110 (58.59% game points), with
positive margins in both colors (156–106–58 as Light and 164–104–52 as Dark).
The candidate and incumbent again received effectively equal search resources:
298,939 versus 296,678 mean nodes per move and depth 5.05 versus 5.08 in the
400-game gate. The independent replay audit accepted all 400 games as legal.
This is a genuine fixed-budget strength signal, but it is not yet a shipped
opponent: the model artifact still needs a durable browser/WASM integration and
the broader tactical/regression review required for promotion.

Representative replay review covered a Light win (seed `2026084001`, path win
at ply 41), a Dark win (`2026084002`, path win at ply 32), a Light loss
(`2026084011`, opponent path win at ply 42), a Dark loss (`2026084010`,
opponent path win at ply 53), and draws for both colors (`2026084003` and
`2026084012`, both reaching the 60-ply cap). These samples showed ordinary
placements and relocations, immediate winning finishes, and capped draws; no
model-selected action bypassed the rules or tactical-safe root.

## Data and artifacts

The 1,920 roots and labels are retained in the ignored workspace of the
super-deep contextual path because they are the source corpus for this
experiment. Trained JSON weights, reports, arena records, and one-time logs
belong in this path's ignored `workspace/`. No generated checkpoint or replay
archive is a durable asset yet.

## Project impact

This path adds a research-only Rust action-policy hook and tests. It does not
change the supported opponent, browser roster, or canonical data. If the
candidate clears independent arena and regression gates, the stable model
contract and a versioned artifact can be promoted in a later step; otherwise
the feature audit and negative result remain as guidance.

## Next decision

The 400-game gate confirms that the explicit state-aware action scorer is the
strongest current avenue and earns a promotion attempt, not an automatic roster
change. Keep the incumbent v0.5 as the supported browser opponent while the
candidate is packaged as a stable Rust/WASM artifact and exercised against the
durable tactical suite. Retain the compact scorer as the cheap ordering
baseline; board-only policy/value variants failed their held-out gate and should
not consume more arena budget until their objective or batching is redesigned.

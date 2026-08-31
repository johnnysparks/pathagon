# 20260829 Board-aware policy/value

Status: inconclusive · board-aware variants failed the held-out gate

## Idea

The action-transition study found a reproducible gain from nonlinear
action-specific ordering, but only a modest gain from hand-added scalar state
features. This path asks whether the board itself is the missing state
representation: a small message-passing policy/value network should be able to
share spatial structure across placements and relocations while Pathfinder
continues to own legality, tactical filtering, and recursive search.

## Starting point

The incumbent is `pathfinder-v0.5.0-trained-evaluator` with weights
`path=241, material=112, capture=887, structure=40, threat=154, edge=74` and
the tactical-safe root filter. The best current research candidate is the
explicit 32-feature action-transition scorer, which reached 129–77–34 across
two independent 120-game arenas but is not promoted. The source-disjoint
turn-balanced corpus contains 1,920 7x7 roots and one-million-node/depth-7
labels, split into 1,535 training and 385 held-out roots.

## Protocol

Train the existing 7x7 residual message-passing graph model on the frozen
teacher targets. Board nodes encode occupancy, forbidden squares, relocation
markers, coordinates, boundaries, and player-to-move; typed dynamic action
heads score legal placements and relocations. The value head receives an
auxiliary normalized teacher-score target, while the policy head learns the
teacher-selected action. No arena seed participates in training or model
selection.

Export the selected checkpoint through the existing ONNX contract. In Rust,
use it only to reorder Pathfinder's tactical-safe root candidates; keep the
full legal action list, immediate-win guard, iterative-deepening search,
one-million-node ceiling, and 2.8-second deadline unchanged. Measure held-out
teacher top-1, value agreement, unsafe selections, model inference overhead,
and paired color-balanced arenas before considering promotion.

## What happened

The first board-only fit (48 hidden units, six message layers, eight augmented
epochs) reached only 55/385 held-out teacher top-1 (14.3%), below the incumbent
feature baseline's 103/385 (26.8%). Adding the existing deterministic
transition Q/advantage head improved the follow-up to 92/385 (23.9%) after
pairwise ranking fine-tuning, but still did not clear the held-out gate. These
fits were not sent to an arena. The board-only failure is useful evidence that
spatial message passing alone does not recover the teacher's action-specific
afterstate signal; the transition head recovers some of it but not enough at
this training scale.

The explicit action-transition scorer remains the only candidate with a
positive, reproducible arena signal. Its separate untouched 400-game gate
completed at 191–133–76 from the candidate perspective (57.25% game points),
with positive margins as both Light and Dark; all 18,527 plies replayed legally.
Training reports and checkpoint hashes for all board-aware fits are retained in
the ignored workspace.

## Data and artifacts

The frozen roots and labels remain in the ignored workspace of
`20260829-superdeep-contextual-evaluator`; this path stores only one-time
checkpoints, ONNX exports, reports, and arena logs under its ignored
`workspace/`. Nothing is promoted to `data/` until the full strength,
legality, regression, and latency gates pass.

## Project impact

The experiment reuses the existing Python graph architecture and native ONNX
inference ABI. It adds a research-only full tactical-safe-root sorter hook to
the Rust engine, but no board-aware model or supported opponent has been
promoted. Until a long untouched arena and tactical review pass, it does not
change the supported opponent, browser roster, or canonical corpus.

## Next decision

Retain the board-aware failures and do not spend arena compute on them. Use the
validated explicit action-transition scorer as the ordering baseline while it
goes through durable Rust/WASM packaging and tactical regression review. A
future board attempt needs a better teacher objective or a more efficient
batched action/value architecture, not merely more unstructured data volume.
